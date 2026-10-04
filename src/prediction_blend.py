# MIT License
# Copyright (c) 2026 cubres
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Strict study alignment and probability blending."""
from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, TextIO


def _names(values: Iterable[str], field: str) -> list[str]:
    names = list(values)
    if not names or any(not isinstance(v, str) or not v or v.isspace() for v in names):
        raise ValueError(f"{field} must contain nonempty string names")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicate names")
    return names


def _numeric(value: Any, field: str, *, probability: bool) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    if isinstance(value, (str, Decimal)):
        try:
            exact = Decimal(value)
        except (InvalidOperation, ValueError):
            raise ValueError(f"{field} must be numeric") from None
        if not exact.is_finite():
            raise ValueError(f"{field} must be finite")
        if exact < 0 or (probability and exact > 1):
            raise ValueError(f"{field} is outside the permitted range")
        value = exact
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{field} must be a finite numeric value") from None
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if number < 0 or (probability and number > 1):
        raise ValueError(f"{field} is outside the permitted range")
    return number


def read_prediction_csv(stream: TextIO) -> list[dict[str, str]]:
    """Read a caller-owned CSV stream, rejecting duplicate or empty headers.

    Validate headers before creating dictionaries; duplicate CSV columns cannot
    be recovered from a dictionary after a permissive reader has collapsed them.
    Identity and probability checks are performed by blend_prediction_tables.
    """
    reader = csv.reader(stream, strict=True)
    try:
        headers = _names(next(reader), "CSV headers")
    except StopIteration:
        raise ValueError("prediction CSV is empty") from None
    except csv.Error as error:
        raise ValueError(f"invalid prediction CSV: {error}") from None
    rows = []
    try:
        for index, values in enumerate(reader, start=2):
            if len(values) != len(headers):
                raise ValueError(f"CSV row {index} has {len(values)} values; expected {len(headers)}")
            rows.append(dict(zip(headers, values)))
    except csv.Error as error:
        raise ValueError(f"invalid prediction CSV: {error}") from None
    return rows


def blend_prediction_tables(
    prediction_tables: Iterable[Iterable[Mapping[str, Any]]], *,
    study_ids: Iterable[str], label_names: Iterable[str],
    weights: Iterable[Any], id_column: str,
) -> list[dict[str, str | float]]:
    """Align exact string identities, blend declared probabilities, preserve order.

    Every table must contain exactly the requested studies and each declared
    label. Tables and labels may arrive in any order. Extra metadata columns are
    ignored. All tables are validated, including those assigned zero weight.
    IDs are never stripped or coerced. Numeric CSV strings are accepted; booleans
    are rejected. The result contains id_column followed by the declared labels.
    """
    requested = _names(study_ids, "study_ids")
    labels = _names(label_names, "label_names")
    _names([id_column], "id_column")
    if id_column in labels:
        raise ValueError("id_column cannot also be a probability label")
    tables = list(prediction_tables)
    raw_weights = list(weights)
    if not tables:
        raise ValueError("at least one prediction table is required")
    if len(raw_weights) != len(tables):
        raise ValueError("one explicit weight is required per prediction table")
    parsed_weights = [_numeric(v, f"weight {i}", probability=False) for i, v in enumerate(raw_weights)]
    scale = max(parsed_weights)
    if scale == 0:
        raise ValueError("at least one weight must be positive")
    # Scaling avoids overflow when finite weights are individually very large.
    scaled_weights = [w / scale for w in parsed_weights]
    denominator = math.fsum(scaled_weights)
    requested_set = set(requested)
    aligned = []
    for table_index, rows in enumerate(tables):
        by_study = {}
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"table {table_index} row {row_index} must be a mapping")
            if id_column not in row:
                raise ValueError(f"table {table_index} row {row_index} is missing {id_column}")
            identity = row[id_column]
            if not isinstance(identity, str) or not identity or identity.isspace():
                raise ValueError(f"table {table_index} row {row_index} has an invalid study identity")
            if identity in by_study:
                raise ValueError(f"table {table_index} has duplicate study identity {identity!r}")
            missing_labels = [label for label in labels if label not in row]
            if missing_labels:
                raise ValueError(f"table {table_index} study {identity!r} is missing labels {missing_labels!r}")
            by_study[identity] = [_numeric(row[label], f"table {table_index} study {identity!r} label {label!r}", probability=True) for label in labels]
        actual = set(by_study)
        if actual != requested_set:
            missing, extra = sorted(requested_set - actual), sorted(actual - requested_set)
            raise ValueError(f"table {table_index} study mismatch: missing={missing!r}, extra={extra!r}")
        aligned.append(by_study)
    output = []
    for identity in requested:
        result: dict[str, str | float] = {id_column: identity}
        for label_index, label in enumerate(labels):
            result[label] = math.fsum(table[identity][label_index] * weight for table, weight in zip(aligned, scaled_weights)) / denominator
        output.append(result)
    return output
