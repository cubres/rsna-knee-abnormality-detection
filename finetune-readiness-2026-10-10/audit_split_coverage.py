#!/usr/bin/env python3
"""Original CPU-only ledger audit. Self-check rows are entirely invented.

This does not run a model or calculate AUC. Private identifiers are consumed only
to count intersections; they are never included in the result.
"""
import argparse
import copy
import json
from pathlib import Path


def audit(ledger):
    rows = ledger['rows']
    findings = ledger['findings']
    if type(findings) is not list or not findings or any(type(f) is not str or not f for f in findings):
        raise ValueError('Findings must be a nonempty list of strings')
    if len(findings) != len(set(findings)):
        raise ValueError('Finding names must be unique')
    if type(rows) is not list or not rows:
        raise ValueError('Rows must be a nonempty list')
    def tokens(value):
        if type(value) is not list or any(type(t) is not str or not t for t in value):
            raise ValueError('Exposure inventories must be explicit lists of nonempty tokens')
        return set(value)
    exposed = tokens(ledger.get('initialization_exposed_study_tokens'))
    exposed_patients = tokens(ledger.get('initialization_exposed_patient_tokens'))
    errors = []
    seen = set()
    split_studies = {'fit': set(), 'held': set()}
    split_patients = {'fit': set(), 'held': set()}
    coverage = {s: [{'negative': 0, 'positive': 0, 'unknown': 0} for _ in findings]
                for s in split_studies}
    patient_map_complete = True
    for row in rows:
        if type(row) is not dict:
            raise ValueError('Each row must be an object')
        split = row['split']
        if split not in split_studies:
            raise ValueError('Only fit and held rows are supported')
        study = row['study_token']
        if type(study) is not str or not study:
            raise ValueError('Study tokens must be nonempty strings')
        if study in seen:
            errors.append('duplicate_study')
        seen.add(study)
        split_studies[split].add(study)
        patient = row.get('patient_token')
        if patient is None:
            patient_map_complete = False
        else:
            if type(patient) is not str or not patient:
                raise ValueError('Patient tokens must be nonempty strings or null')
            split_patients[split].add(patient)
        labels, known = row['truth_labels'], row['known_mask']
        if type(labels) is not list or type(known) is not list or len(labels) != len(findings) or len(known) != len(findings):
            raise ValueError('Label/mask width differs from finding count')
        for j, (label, mask) in enumerate(zip(labels, known)):
            if type(mask) is not bool:
                raise ValueError('Known-label masks must be explicit booleans')
            if not mask:
                coverage[split][j]['unknown'] += 1
            elif type(label) not in (int, float) or label not in (0, 1):
                raise ValueError('Known truth labels must be binary, not soft report targets')
            else:
                coverage[split][j]['positive' if label else 'negative'] += 1
    if split_studies['fit'] & split_studies['held']:
        errors.append('study_overlap')
    if split_patients['fit'] & split_patients['held']:
        errors.append('patient_overlap')
    if not patient_map_complete:
        errors.append('patient_isolation_unproven')
    if any(not v for v in split_studies.values()):
        errors.append('fit_or_held_split_empty')
    if exposed & split_studies['held']:
        errors.append('initialization_exposed_to_held')
    if exposed_patients & split_patients['held']:
        errors.append('initialization_exposed_to_held_patient')
    if ledger.get('initialization_exposure_inventory_complete') is not True:
        errors.append('initialization_exposure_unproven')
    if ledger.get('held_used_for_checkpoint_selection') is not False:
        errors.append('held_checkpoint_selection_unproven_or_present')
    missing_classes = [j for j, c in enumerate(coverage['held'])
                       if not c['negative'] or not c['positive']]
    if missing_classes:
        errors.append('held_known_class_coverage_incomplete')
    eng = ledger['engineering']
    if type(eng) is not dict:
        raise ValueError('Engineering must be an object')
    counters_valid = all(type(eng.get(k)) is int and eng[k] >= 0
                         for k in ('optimizer_updates', 'expected_updates', 'build_failures'))
    engineering_pass = (eng.get('finite_loss') is True
                        and eng.get('finite_gradients') is True
                        and counters_valid
                        and eng.get('optimizer_updates') == eng.get('expected_updates')
                        and type(eng.get('expected_updates')) is int
                        and eng['expected_updates'] > 0
                        and eng.get('build_failures') == 0)
    return {'engineering_pass': engineering_pass,
            'independent_evaluation_ready': not errors,
            'scope': 'ledger invariants only; no AUC, confidence interval or score claim',
            'science_holds': sorted(set(errors)),
            'studies': {s: len(v) for s, v in split_studies.items()},
            'distinct_patients': ({s: len(v) for s, v in split_patients.items()}
                                  if patient_map_complete else None),
            'known_class_coverage': coverage,
            'held_findings_missing_either_class': missing_classes}


def self_check():
    valid = {'findings': ['invented_finding'], 'rows': [
        {'study_token': 'invented_s1', 'patient_token': 'invented_p1', 'split': 'fit',
         'truth_labels': [0], 'known_mask': [True]},
        {'study_token': 'invented_s2', 'patient_token': 'invented_p2', 'split': 'held',
         'truth_labels': [0], 'known_mask': [True]},
        {'study_token': 'invented_s3', 'patient_token': 'invented_p3', 'split': 'held',
         'truth_labels': [1], 'known_mask': [True]}],
        'initialization_exposed_study_tokens': ['invented_s1'],
        'initialization_exposed_patient_tokens': ['invented_p1'],
        'initialization_exposure_inventory_complete': True,
        'held_used_for_checkpoint_selection': False,
        'engineering': {'finite_loss': True, 'finite_gradients': True,
                        'optimizer_updates': 2, 'expected_updates': 2, 'build_failures': 0}}
    cases = [('clean_synthetic_ledger', valid, None)]
    def variant(name, mutate, hold):
        item = copy.deepcopy(valid)
        mutate(item)
        cases.append((name, item, hold))
    variant('study_disjoint_but_patient_leaks',
            lambda x: x['rows'][1].update(patient_token='invented_p1'), 'patient_overlap')
    variant('pretrained_model_saw_held',
            lambda x: x['initialization_exposed_study_tokens'].append('invented_s2'),
            'initialization_exposed_to_held')
    variant('pretrained_model_saw_other_knee_of_held_patient',
            lambda x: x['initialization_exposed_patient_tokens'].append('invented_p2'),
            'initialization_exposed_to_held_patient')
    variant('unknown_positive_does_not_supply_class',
            lambda x: x['rows'][2].update(known_mask=[False]), 'held_known_class_coverage_incomplete')
    variant('patient_mapping_absent',
            lambda x: x['rows'][1].update(patient_token=None), 'patient_isolation_unproven')
    variant('held_chose_pretrained_checkpoint',
            lambda x: x.update(held_used_for_checkpoint_selection=True),
            'held_checkpoint_selection_unproven_or_present')
    variant('nominal_amp_steps_but_no_updates',
            lambda x: x['engineering'].update(optimizer_updates=0), 'engineering_only')
    variant('boolean_update_counter_is_not_integer_count',
            lambda x: x['engineering'].update(optimizer_updates=True, expected_updates=1),
            'engineering_only')
    results = []
    for name, item, hold in cases:
        result = audit(item)
        passed = (result['engineering_pass'] and result['independent_evaluation_ready']
                  if hold is None else not result['engineering_pass'] if hold == 'engineering_only'
                  else hold in result['science_holds'] and result['engineering_pass'])
        if not passed:
            raise RuntimeError('Synthetic control failed: ' + name)
        results.append({'case': name, 'passed': passed, 'engineering_pass': result['engineering_pass'],
                        'independent_evaluation_ready': result['independent_evaluation_ready'],
                        'science_holds': result['science_holds']})
    malformed = [copy.deepcopy(valid) for _ in range(3)]
    malformed[0]['findings'] = []
    malformed[1]['rows'] = []
    malformed[2]['initialization_exposed_study_tokens'] = 'invented_s2'
    for name, item in zip(('empty_findings', 'empty_rows', 'scalar_exposure_inventory'), malformed):
        try:
            audit(item)
        except ValueError:
            results.append({'case': name, 'passed': True, 'malformed_ledger_rejected': True})
        else:
            raise RuntimeError('Malformed ledger was accepted: ' + name)
    return {'synthetic_only': True, 'model_runs': 0, 'cases': results, 'passed': len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    if args.self_check == bool(args.ledger):
        parser.error('Choose exactly one of --self-check or --ledger')
    result = self_check() if args.self_check else audit(json.loads(args.ledger.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
