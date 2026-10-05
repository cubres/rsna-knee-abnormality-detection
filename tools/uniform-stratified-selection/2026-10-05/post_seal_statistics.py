"""Descriptive paired whole-study bootstrap, with synthetic-only preparation.

No file/model/data loading and no global RNG use. Native caller may invoke
this only after both immutable prediction seals and the approved target-read
boundary. Class-unsupported draws are rejected jointly and disclosed, making
the intervals conditional and descriptive rather than independent validation.
"""
from __future__ import annotations
import math
import numbers
import time
import numpy as np


def _prepared_auc(binary, values):
    prepared = []
    for finding in range(binary.shape[1]):
        order = np.argsort(values[:, finding], kind='stable')
        sorted_values = values[order, finding]
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        prepared.append((order, starts, binary[order, finding].astype(np.float64)))
    return prepared


def _weighted_auc(prepared, weights):
    result = []
    for order, starts, positive in prepared:
        ordered_weights = weights[order]
        p = np.add.reduceat(ordered_weights * positive, starts)
        n = np.add.reduceat(ordered_weights * (1. - positive), starts)
        total_p, total_n = p.sum(), n.sum()
        if not total_p > 0 or not total_n > 0:
            raise ValueError('all findings require both observed classes')
        before_n = np.cumsum(n) - n
        result.append(float(np.dot(p, before_n + .5 * n) / (total_p * total_n)))
    return np.asarray(result, np.float64)


def paired_report_comparison(soft_targets, predictions, finding_names, *,
                             seed, replicates=2000, alpha=.05,
                             max_attempt_factor=20, deadline_monotonic=None):
    """Use identical whole-study resampling weights for both arms/all findings."""
    y = np.asarray(soft_targets)
    if y.ndim != 2 or min(y.shape) < 1 or y.shape[0] < 2:
        raise ValueError('targets need at least two studies and one finding')
    if not np.issubdtype(y.dtype, np.number) or np.iscomplexobj(y):
        raise ValueError('targets must be real numerical values')
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError('targets must be finite report probabilities')
    names = list(finding_names)
    if len(names) != y.shape[1] or len(set(names)) != len(names):
        raise ValueError('finding names must be complete and unique')
    arms = ('peer_selection', 'random_selection')
    if set(predictions) != set(arms):
        raise ValueError('exactly the preregistered two arms are required')
    for value, name in ((seed, 'seed'), (replicates, 'replicates'),
                        (max_attempt_factor, 'max_attempt_factor')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError(name + ' must be an integer')
    if seed < 0 or replicates < 2 or max_attempt_factor < 1:
        raise ValueError('invalid seed or bootstrap counts')
    if not isinstance(alpha, numbers.Real) or not math.isfinite(float(alpha)) or not 0 < alpha < 1:
        raise ValueError('alpha must be in (0,1)')
    binary = y > .5
    positives = binary.sum(0).astype(np.int64)
    negatives = y.shape[0] - positives
    if ((positives == 0) | (negatives == 0)).any():
        raise ValueError('all findings require both observed classes')
    prepared, metrics = {}, {}
    for arm in arms:
        values = np.asarray(predictions[arm])
        if values.shape != y.shape or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError('prediction shape/dtype contract')
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('predictions must be finite probabilities')
        prepared[arm] = _prepared_auc(binary, values)
        auc = _weighted_auc(prepared[arm], np.ones(y.shape[0]))
        metrics[arm] = dict(per_target_auc=auc.tolist(), macro_auc=float(auc.mean()))
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    deltas = []
    rejected_per_finding = np.zeros(y.shape[1], np.int64)
    attempts = 0
    max_attempts = int(replicates) * int(max_attempt_factor)
    while len(deltas) < replicates and attempts < max_attempts:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise ValueError('post-seal descriptive statistics exceeded deadline')
        attempts += 1
        indices = generator.integers(0, y.shape[0], size=y.shape[0])
        weights = np.bincount(indices, minlength=y.shape[0]).astype(np.float64)
        sampled_p = weights @ binary
        unsupported = (sampled_p == 0) | (sampled_p == y.shape[0])
        if unsupported.any():
            rejected_per_finding += unsupported
            continue
        deltas.append(_weighted_auc(prepared[arms[0]], weights)
                      - _weighted_auc(prepared[arms[1]], weights))
    if len(deltas) != replicates:
        raise ValueError('insufficient jointly class-supported bootstrap draws')
    deltas = np.asarray(deltas, np.float64)
    interval = np.quantile(deltas, [alpha / 2, 1 - alpha / 2], axis=0, method='linear')
    macro_interval = np.quantile(deltas.mean(1), [alpha / 2, 1 - alpha / 2], method='linear')
    return dict(
        metrics=metrics,
        peer_minus_random=metrics[arms[0]]['macro_auc'] - metrics[arms[1]]['macro_auc'],
        held_class_support=[dict(finding=name, negative=int(n), positive=int(p))
                            for name, n, p in zip(names, negatives, positives)],
        descriptive_paired_bootstrap=dict(
            sampling_unit='whole study; same weights for both arms and all findings',
            generator='NumPy PCG64 dedicated stream', seed=int(seed),
            requested_valid_replicates=int(replicates), valid_replicates=len(deltas),
            attempts=attempts, rejected_joint_draws=attempts - len(deltas),
            rejected_unsupported_draws_per_finding=rejected_per_finding.tolist(),
            rejection_rule='reject entire draw if any finding loses either class; never change macro denominator',
            interval_method='pointwise percentile, NumPy linear quantiles', alpha=float(alpha),
            macro_delta_interval=macro_interval.tolist(),
            per_finding_delta_intervals=interval.T.tolist(),
            conditional_on_class_support=True, simultaneous_intervals=False,
            independent_validation=False, patient_independence_proven=False,
            previously_exposed_panel=True,
            interpretation='Descriptive conditional resampling of this exposed report-proxy panel only; not confirmatory significance or clinical/official-score evidence'))
