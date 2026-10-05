"""Independent synthetic checks; Python/NumPy only, reports to stdout.

No executor, models, Torch, real data, private assets or network is used.
"""
from __future__ import annotations
import sys
if not __debug__:
    sys.stderr.write('REFUSE_OPTIMIZED_MODE: assertions must remain enabled\n')
    raise SystemExit(2)

import ast
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time
import numpy as np
from selection_controls import make_mask_generators, uniform_stratified_mask, expected_support, mask_support
from post_seal_statistics import paired_report_comparison


def direct_counts(mask, signs):
    """Independent row enumeration, not the module's vectorized counter."""
    return np.array([[sum(bool(mask[i,j]) and bool(signs[i,j]) == sign
                          for i in range(len(signs))) for sign in (False, True)]
                     for j in range(signs.shape[1])], dtype=np.int64)


def rational_counts(signs, numerator, denominator):
    """Integer arithmetic oracle for the declared small-stratum fractions."""
    result = []
    for j in range(signs.shape[1]):
        row = []
        for sign in (False, True):
            n = sum(bool(signs[i,j]) == sign for i in range(len(signs)))
            row.append(max(1,(numerator*n)//denominator) if n else 0)
        result.append(row)
    return np.array(result, dtype=np.int64)


class EnumeratedPermutation(np.random.Generator):
    """Feed every stratum permutation through the ACTUAL mask implementation."""
    def __init__(self, permutation):
        super().__init__(np.random.PCG64(0))
        self.explicit_permutation = permutation
        self.calls = 0

    def permutation(self, values, axis=0):
        assert axis == 0 and len(values) == len(self.explicit_permutation)
        self.calls += 1
        return values[np.array(self.explicit_permutation)]


def numpy_state_equal(a, b):
    return a[0] == b[0] and np.array_equal(a[1],b[1]) and a[2:] == b[2:]


def auc_pairwise(labels, scores):
    """Independent concordant positive/negative pairs, with half-credit ties."""
    positives = [i for i,value in enumerate(labels) if value]
    negatives = [i for i,value in enumerate(labels) if not value]
    if not positives or not negatives:
        raise ValueError('unsupported synthetic class')
    return sum(float(scores[p] > scores[n]) + .5*float(scores[p] == scores[n])
               for p in positives for n in negatives)/(len(positives)*len(negatives))


def brute_bootstrap(y, peer, uniform, seed, replicates, alpha=.05):
    """Resample actual row indices, not sorted ranks or weighted tie reductions."""
    binary = y > .5
    generator = np.random.Generator(np.random.PCG64(seed))
    deltas, attempts = [], 0
    rejected = np.zeros(y.shape[1],np.int64)
    while len(deltas) < replicates:
        attempts += 1
        indices = generator.integers(0,len(y),size=len(y))
        sample_y = binary[indices]
        unsupported = np.array([not any(sample_y[:,j]) or all(sample_y[:,j])
                                for j in range(y.shape[1])])
        if unsupported.any():
            rejected += unsupported
            continue
        deltas.append([auc_pairwise(sample_y[:,j],peer[indices,j])
                       -auc_pairwise(sample_y[:,j],uniform[indices,j])
                       for j in range(y.shape[1])])
        if attempts > replicates*20:
            raise AssertionError('unexpected synthetic oracle rejection rate')
    deltas = np.array(deltas)
    return dict(attempts=attempts,rejected=rejected,
                macro_interval=np.quantile(deltas.mean(1),[alpha/2,1-alpha/2],method='linear'),
                finding_intervals=np.quantile(deltas,[alpha/2,1-alpha/2],axis=0,method='linear').T)


def must_reject(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError('required rejection did not occur')


def compare_statistics(y, peer, uniform, names, seed, replicates):
    actual = paired_report_comparison(y,dict(peer_selection=peer,random_selection=uniform),names,
                                      seed=seed,replicates=replicates)
    reference = brute_bootstrap(y,peer,uniform,seed,replicates)
    binary = y > .5
    expected_peer = [auc_pairwise(binary[:,j],peer[:,j]) for j in range(y.shape[1])]
    expected_random = [auc_pairwise(binary[:,j],uniform[:,j]) for j in range(y.shape[1])]
    assert np.allclose(actual['metrics']['peer_selection']['per_target_auc'],expected_peer,atol=1e-15,rtol=0)
    assert np.allclose(actual['metrics']['random_selection']['per_target_auc'],expected_random,atol=1e-15,rtol=0)
    assert math.isclose(actual['peer_minus_random'],float(np.mean(expected_peer)-np.mean(expected_random)),abs_tol=1e-15)
    summary = actual['descriptive_paired_bootstrap']
    assert summary['attempts'] == reference['attempts']
    assert np.array_equal(summary['rejected_unsupported_draws_per_finding'],reference['rejected'])
    assert np.allclose(summary['macro_delta_interval'],reference['macro_interval'],rtol=0,atol=1e-14)
    assert np.allclose(summary['per_finding_delta_intervals'],reference['finding_intervals'],rtol=0,atol=1e-14)
    assert len(summary['per_finding_delta_intervals']) == y.shape[1]
    assert summary['conditional_on_class_support'] and not summary['independent_validation']
    assert not summary['simultaneous_intervals'] and not summary['patient_independence_proven']
    assert all(row['positive']+row['negative'] == len(y) for row in actual['held_class_support'])
    return actual


def main():
    began = time.monotonic()
    numpy_before,python_before = np.random.get_state(),random.getstate()
    streams = make_mask_generators(2026100521)
    fractions = ((0,1),(1,5),(1,2),(4,5),(1,1))
    single_cases,two_finding_cases,mask_calls = 0,0,0
    for batch in range(1,5):
        for findings in (1,2):
            for flat_bits in itertools.product((False,True),repeat=batch*findings):
                signs = np.array(flat_bits,bool).reshape(batch,findings)
                original = signs.copy()
                for numerator,denominator in fractions:
                    fraction = numerator/denominator
                    wanted = rational_counts(signs,numerator,denominator)
                    assert np.array_equal(expected_support(signs,fraction),wanted)
                    for generator in streams:
                        actual = uniform_stratified_mask(signs,fraction,generator)
                        assert np.array_equal(direct_counts(actual,signs),wanted)
                        assert np.array_equal(mask_support(actual,signs),direct_counts(actual,signs))
                        assert np.array_equal(signs,original)
                        mask_calls += 1
                    if findings == 1: single_cases += 1
                    else: two_finding_cases += 1
    assert (single_cases,two_finding_cases,mask_calls) == (150,1700,3700)

    uniform_oracles = []
    oracle_mask_calls = 0
    for n in range(1,5):
        for k in range(1,n+1):
            frequencies = Counter()
            for permutation in itertools.permutations(range(n)):
                generator = EnumeratedPermutation(permutation)
                mask = uniform_stratified_mask(np.zeros((n,1),bool),k/n,generator)
                subset = tuple(i for i in range(n) if mask[i,0])
                frequencies[subset] += 1
                assert generator.calls == (0 if k == n else 1)
                oracle_mask_calls += 1
            assert len(frequencies) == math.comb(n,k)
            assert set(frequencies.values()) == {math.factorial(k)*math.factorial(n-k)}
            uniform_oracles.append(dict(n=n,k=k,subsets=len(frequencies),equal_permutation_preimages=True))

    signs = np.array([[False,True],[False,True],[True,False],[True,False]])
    a,b = make_mask_generators(311),make_mask_generators(311)
    warmup_states = [json.dumps(g.bit_generator.state,sort_keys=True) for g in a]
    for generator in a: assert uniform_stratified_mask(signs,1.,generator).all()
    assert warmup_states == [json.dumps(g.bit_generator.state,sort_keys=True) for g in a]
    for _ in range(20):
        for x,z in zip(a,b):
            assert np.array_equal(uniform_stratified_mask(signs,.8,x),uniform_stratified_mask(signs,.8,z))
    untouched_selector = json.dumps(a[1].bit_generator.state,sort_keys=True)
    unrelated_order_stream = np.random.Generator(np.random.PCG64(661))
    order_state = json.dumps(unrelated_order_stream.bit_generator.state,sort_keys=True)
    for _ in range(20): uniform_stratified_mask(signs,.8,a[0])
    assert untouched_selector == json.dumps(a[1].bit_generator.state,sort_keys=True)
    assert order_state == json.dumps(unrelated_order_stream.bit_generator.state,sort_keys=True)

    y = np.array([[int((i+j)%3 == 0) for j in range(3)] for i in range(12)],float)
    rng = np.random.Generator(np.random.PCG64(281))
    scores = rng.integers(0,5,size=y.shape)/4
    peer = np.clip(scores+.2*(2*y-1),0,1)
    names = ['synthetic_a','synthetic_b','synthetic_c']
    snapshots = [item.copy() for item in (y,scores,peer)]
    identity = compare_statistics(y,scores,scores,names,2026100522,128)
    assert identity['peer_minus_random'] == 0.
    assert identity['descriptive_paired_bootstrap']['macro_delta_interval'] == [0.,0.]
    forward = compare_statistics(y,peer,scores,names,2026100522,128)
    backward = compare_statistics(y,scores,peer,names,2026100522,128)
    assert math.isclose(forward['peer_minus_random'],-backward['peer_minus_random'],abs_tol=1e-15)
    f,b = forward['descriptive_paired_bootstrap'],backward['descriptive_paired_bootstrap']
    assert np.allclose(f['macro_delta_interval'],[-b['macro_delta_interval'][1],-b['macro_delta_interval'][0]],atol=1e-14,rtol=0)
    repeat = paired_report_comparison(y,dict(peer_selection=peer,random_selection=scores),names,
                                     seed=2026100522,replicates=128)
    assert repeat == forward
    for value,original in zip((y,scores,peer),snapshots): assert np.array_equal(value,original)

    rare = np.zeros((8,2));rare[0,0]=1.;rare[1:5,1]=1.
    rare_scores = np.tile(np.linspace(0,1,8)[:,None],(1,2))
    rare_result = compare_statistics(rare,rare_scores,rare_scores,['rare','common'],31,128)
    assert rare_result['descriptive_paired_bootstrap']['rejected_joint_draws'] > 0
    rejection_checks = [
        lambda: uniform_stratified_mask(signs.astype(int),.8,a[0]),
        lambda: uniform_stratified_mask(signs,float('nan'),a[0]),
        lambda: uniform_stratified_mask(signs,-.1,a[0]),
        lambda: uniform_stratified_mask(signs,1.1,a[0]),
        lambda: uniform_stratified_mask(signs,.8,None),
        lambda: make_mask_generators(True),
        lambda: paired_report_comparison(y,dict(peer_selection=peer,random_selection=scores),names,
                                         seed=31,replicates=20,deadline_monotonic=0.),
        lambda: paired_report_comparison(np.zeros_like(y),dict(peer_selection=peer,random_selection=scores),names,
                                         seed=31,replicates=20),
        lambda: paired_report_comparison(y,dict(peer_selection=peer),names,seed=31,replicates=20),
        lambda: paired_report_comparison(np.array([[0.],[1.]]),
                    dict(peer_selection=np.array([[0.],[1.]]),random_selection=np.array([[0.],[1.]])),
                    ['tiny'],seed=0,replicates=2,max_attempt_factor=1),
    ]
    for callback in rejection_checks: must_reject(callback)
    assert numpy_state_equal(numpy_before,np.random.get_state())
    assert python_before == random.getstate()
    for name in ('selection_controls.py','post_seal_statistics.py'):
        source = ast.parse(Path(__file__).with_name(name).read_text())
        imports = [n.module for n in ast.walk(source) if isinstance(n,ast.ImportFrom)]
        imports += [alias.name for n in ast.walk(source) if isinstance(n,ast.Import) for alias in n.names]
        assert all((value or '').split('.')[0] in ('__future__','math','numbers','time','numpy') for value in imports)
    optimized = subprocess.run([sys.executable,'-O','-B',str(Path(__file__).resolve())],capture_output=True,text=True,timeout=10)
    assert optimized.returncode == 2 and optimized.stdout == ''
    assert 'REFUSE_OPTIMIZED_MODE' in optimized.stderr
    print(json.dumps(dict(status='PASS_PORTABLE_NUMPY_ONLY_SYNTHETIC',
        python=platform.python_version(),numpy=np.__version__,
        exhaustive_single_finding_configurations=single_cases,
        exhaustive_two_finding_configurations=two_finding_cases,
        mask_calls_across_two_selectors=mask_calls,small_batch_sizes=[1,2,3,4],
        retained_count_oracle='independent integer-rational cardinalities and row enumeration',
        uniformity_oracle='all permutations fed through actual mask sampler; equal k-subset preimage counts',
        enumerated_permutation_mask_calls=oracle_mask_calls,uniform_cases=uniform_oracles,
        dedicated_stream_replay=True,selector_stream_isolation=True,warmup_consumes_no_rng=True,
        unrelated_order_generator_unchanged=True,global_numpy_rng_unchanged=True,global_python_rng_unchanged=True,
        bootstrap_oracle='independent explicit row-resampling and positive-negative pair enumeration; no module private AUC helpers used',
        bootstrap_independent_reference_cases=4,bootstrap_replicates_per_case=128,
        bootstrap_identity=True,bootstrap_sign_swap=True,bootstrap_reproducible=True,
        rare_class_joint_rejected_draws=rare_result['descriptive_paired_bootstrap']['rejected_joint_draws'],
        no_finding_dropped_on_rejection=True,deadline_rejection=True,max_attempt_rejection=True,
        rejection_contract_cases=len(rejection_checks),optimized_mode_refused=True,inputs_preserved=True,
        executor_imported_or_run=False,torch_required_or_imported=False,models_or_native=False,
        real_data_or_private_assets=False,network_calls=False,synthetic_wall_seconds=time.monotonic()-began),indent=2,sort_keys=True))


if __name__ == '__main__': main()
