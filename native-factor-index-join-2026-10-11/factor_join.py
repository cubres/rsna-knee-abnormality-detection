"""Original generic checks at a saved-training-contract boundary.

These checks bind current arguments and a fresh registration to the conserved
contract. A caller-supplied authority hash is not authorization. This module
does not inspect competition data, select checkpoints or measure model gains.
"""
import hashlib
import json
import os
from pathlib import Path
import re

import torch


class FactorHold(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise FactorHold(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "explicit SHA256 required")
    return value


def tensor_pin(tensor):
    require(isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.is_contiguous(),
            "explicit contiguous CPU tensor required")
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}


def input_bindings(labels, pos_weight, ordered_ids, data_sha256):
    require(isinstance(labels, torch.Tensor) and labels.ndim == 2 and labels.shape[0] > 0 and labels.shape[1] > 0,
            "nonempty matrix of fixed labels required")
    require(isinstance(pos_weight, torch.Tensor) and pos_weight.ndim == 1 and pos_weight.shape[0] == labels.shape[1],
            "weight length differs from labels")
    require(labels.is_floating_point() and pos_weight.is_floating_point()
            and bool(torch.isfinite(labels).all()) and bool(torch.isfinite(pos_weight).all()), "nonfinite labels or weights")
    require(type(ordered_ids) is list and len(ordered_ids) == labels.shape[0]
            and all(type(uid) is str and bool(uid) for uid in ordered_ids) and len(set(ordered_ids)) == len(ordered_ids),
            "full unique ordered identity namespace required")
    return {"labels_pin": tensor_pin(labels), "pos_weight_pin": tensor_pin(pos_weight),
            "ordered_fit_ids_sha256": digest(ordered_ids), "source_data_contract_sha256": sha(data_sha256)}


def validate_registered_inputs(saved_bindings, registered_bindings, labels, pos_weight, ordered_ids, data_sha256):
    """A new registration cannot reinterpret an old contract's factors."""
    current = input_bindings(labels, pos_weight, ordered_ids, data_sha256)
    require(type(saved_bindings) is dict and canonical(saved_bindings) == canonical(current), "current inputs differ from conserved saved bindings")
    require(type(registered_bindings) is dict and canonical(registered_bindings) == canonical(current), "current inputs differ from explicit registration")
    return current


class IndexedStudyDataset(torch.utils.data.Dataset):
    """Call the base reader once and return (x, y, metadata, absolute_index)."""
    def __init__(self, base, expected_ids=None):
        self.base = base
        self.expected_ids = None if expected_ids is None else list(expected_ids)
        if self.expected_ids is not None:
            require(hasattr(base, "uids") and list(base.uids) == self.expected_ids
                    and len(base) == len(self.expected_ids), "base reader namespace differs")
    def __len__(self):
        return len(self.base)
    def __getitem__(self, index):
        require(type(index) is int and 0 <= index < len(self.base), "absolute base index required")
        if self.expected_ids is not None:
            require(self.base.uids[index] == self.expected_ids[index], "base reader identity changed at observed index")
        value = self.base[index]
        require(isinstance(value, tuple) and len(value) == 3, "base reader must return exactly three values")
        return (*value, index)


def create_only_receipt(path, record):
    """Persist one JSON receipt exclusively; existing and partial bytes survive.

    File and containing-directory fsync establish the declared durability
    boundary. Unknown-outcome retries must inspect the exact existing bytes
    separately, never overwrite, repair, remove, or select another artifact.
    """
    p = Path(path)
    require(p.suffix == ".json" and p.name != "kernel-metadata.json" and not p.is_symlink() and not p.exists(),
            "fresh receipt path required; preserve existing evidence")
    require(type(record) is dict, "explicit receipt object required")
    content = canonical(record)
    with p.open("xb") as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    actual = p.read_bytes()
    require(actual == content, "receipt readback differs; retain evidence")
    return {"path": str(p), "bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()}
