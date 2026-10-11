"""Original fixed half-and-half rank fusion for two qualified probability tables.

This module does not construct a model, fetch weights, contact Kaggle or fit a
blend weight. The caller supplies independently bound native component files.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

RAPTOR_CHECKPOINT_SHA256 = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
PUBLIC_MAIN_READER_SHA256 = 'e9ec648e1b2f6121f36bbe6ae806a5714571db83804c719ea02d5c4bf7fe47d7'

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def probability_matrix(values):
    rows = [[float(value) for value in row] for row in values]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('A nonempty rectangular probability matrix is required')
    if any(not math.isfinite(value) or not 0 <= value <= 1 for row in rows for value in row):
        raise ValueError('Every component probability must be finite and in [0,1]')
    return rows

def average_tie_percentiles(values):
    """Column ranks divided by N; a tied group receives its average rank."""
    rows = probability_matrix(values)
    n, columns = len(rows), len(rows[0])
    result = [[0.0] * columns for _ in rows]
    for column in range(columns):
        order = sorted(range(n), key=lambda index: rows[index][column])
        start = 0
        while start < n:
            end = start + 1
            while end < n and rows[order[end]][column] == rows[order[start]][column]:
                end += 1
            percentile = ((start + 1 + end) / 2) / n
            for index in order[start:end]:
                result[index][column] = percentile
            start = end
    return result

def fixed_half_rank_blend(a_probabilities, b_probabilities):
    """Exactly 0.5*rank(A) + 0.5*rank(B), with the same policy for every label."""
    a, b = probability_matrix(a_probabilities), probability_matrix(b_probabilities)
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError('The component matrix shapes differ')
    a_rank, b_rank = average_tie_percentiles(a), average_tie_percentiles(b)
    return [[(av + bv) / 2 for av, bv in zip(ar, br)] for ar, br in zip(a_rank, b_rank)]

def align_component(component_uids, component_labels, values, required_uids, required_labels):
    """Keep the requested output order; missing, extra or duplicate rows fail."""
    ids, labels = list(component_uids), list(component_labels)
    required_ids, required_columns = list(required_uids), list(required_labels)
    if len(ids) != len(set(ids)) or len(required_ids) != len(set(required_ids)):
        raise ValueError('Duplicate study IDs are not permitted')
    if set(ids) != set(required_ids) or labels != required_columns:
        raise ValueError('The complete study set or label order differs')
    rows = probability_matrix(values)
    if len(rows) != len(ids) or len(rows[0]) != len(labels):
        raise ValueError('IDs, labels and probability dimensions disagree')
    lookup = dict(zip(ids, rows))
    return [lookup[uid] for uid in required_ids]

def load_raptor_raw(npz_path, receipt_path, required_uids, required_labels,
                    expected_npz_sha256, expected_reader_sha256):
    """Load probabilities, not the Raptor reader's already ranked CSV."""
    import numpy as np
    if file_sha256(npz_path) != expected_npz_sha256:
        raise ValueError('The native Raptor probability artifact hash differs')
    receipt = json.loads(Path(receipt_path).read_text())
    if (receipt.get('source_sha256') != expected_reader_sha256 or
            receipt.get('checkpoint_sha256') != RAPTOR_CHECKPOINT_SHA256 or
            receipt.get('native_inference_completed') is not True or
            receipt.get('full_test_coverage') is not True or
            receipt.get('mode') != 'inference' or
            receipt.get('studies') != len(required_uids)):
        raise ValueError('The complete Raptor source/checkpoint/run receipt is required')
    errors = ('header_failures', 'pixel_failures', 'study_build_failures',
              'studies_without_series_metadata', 'model_fallback_studies', 'consumer_error_count')
    if any(receipt.get(key) != 0 for key in errors):
        raise ValueError('Raptor decoding or prediction failures disqualify this blend')
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {'study_uids', 'labels', 'values', 'plain', 'mirrored'}:
            raise ValueError('The Raptor raw artifact schema differs')
        ids, labels = arrays['study_uids'].tolist(), arrays['labels'].tolist()
        values, plain, mirrored = arrays['values'], arrays['plain'], arrays['mirrored']
        expected_shape = (len(ids), len(labels))
        if any(array.shape != expected_shape or not np.isfinite(array).all() or
               not ((array >= 0).all() and (array <= 1).all())
               for array in (values, plain, mirrored)):
            raise ValueError('Raptor mirror component shapes or values differ')
        if not np.allclose(values, (plain + mirrored) / 2, rtol=0, atol=1e-7):
            raise ValueError('Raptor raw values are not the declared mirror probability mean')
        return align_component(ids, labels, values.tolist(), required_uids, required_labels)

def read_probability_csv(path, required_uids, required_labels, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise ValueError('The native probability CSV hash differs')
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['StudyInstanceUID'] + list(required_labels):
            raise ValueError('The native probability CSV label order differs')
        rows = list(reader)
    return align_component([row['StudyInstanceUID'] for row in rows], required_labels,
                           [[row[label] for label in required_labels] for row in rows],
                           required_uids, required_labels)

def export_fixed_half(a_values, b_values, required_uids, required_labels, directory, native_binding):
    """Write a fresh result after a coordinator has bound both native runs."""
    if (native_binding.get('both_native_components_qualified') is not True or
            native_binding.get('fallback_or_missing_studies') != 0 or
            native_binding.get('official_score') is not None):
        raise ValueError('Both exact native components must qualify before fusion')
    ids, labels = list(required_uids), list(required_labels)
    output = fixed_half_rank_blend(a_values, b_values)
    if len(ids) != len(set(ids)) or len(output) != len(ids) or len(output[0]) != len(labels):
        raise ValueError('The output dimensions or study IDs differ')
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    csv_path = destination / 'fixed_half_rank_blend.csv'
    with csv_path.open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID'] + labels)
        writer.writerows([[uid] + row for uid, row in zip(ids, output)])
    receipt = {'method': '0.5 average-tie percentiles from each component, direct mean',
               'weight': 0.5, 'weight_policy': 'one fixed global value, no fitted per-label weights',
               'studies': len(ids), 'labels': labels, 'native_binding': native_binding,
               'csv_sha256': file_sha256(csv_path), 'competition_submitted': False, 'official_score': None}
    with (destination / 'blend_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
    return csv_path
