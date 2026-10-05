"""Fixed-shape stratified masks for small minibatches; no model integration.

Inputs are finite real loss matrices and explicit Boolean class membership.
For the original Knee rule, a caller forms ``positive = soft_target > .5``.
The random control is a separate opt-in function and is never auto-selected.
Dense pairwise ranking costs O(batch_size**2 * findings) memory and work.
This module does not establish speed, compilation behavior or score gains.
"""
from __future__ import annotations

import math
import numbers
import numpy as np


def _retain_value(retain: float) -> float:
    if isinstance(retain, (bool, np.bool_)) or not isinstance(retain, numbers.Real):
        raise TypeError("retain must be a real scalar, not a Boolean")
    value = float(retain)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("retain must be finite and in [0, 1]")
    return value


def _positive_matrix(positive: np.ndarray) -> np.ndarray:
    value = np.asarray(positive)
    if value.ndim != 2 or min(value.shape) <= 0:
        raise ValueError("positive must have nonempty shape (batch, findings)")
    if value.dtype != np.bool_:
        raise TypeError("positive must have Boolean dtype; derive the class rule explicitly")
    return value


def _loss_matrix(loss: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    value = np.asarray(loss)
    if value.shape != shape:
        raise ValueError("loss and positive must have the same (batch, findings) shape")
    if value.dtype.kind not in "iuf":
        raise TypeError("loss must contain real integer or floating-point values")
    if not np.isfinite(value).all():
        raise ValueError("loss values must all be finite")
    return value


def _dense_mask(loss: np.ndarray, positive: np.ndarray, retain: float) -> np.ndarray:
    # Axes are candidate row i, competing row j, finding f. Original row
    # order breaks equal-loss ties, exactly as stable sorting within a stratum.
    same_class = positive[:, None, :] == positive[None, :, :]
    smaller_loss = loss[None, :, :] < loss[:, None, :]
    equal_loss = loss[None, :, :] == loss[:, None, :]
    row = np.arange(loss.shape[0], dtype=np.int64)
    earlier_row = row[None, :, None] < row[:, None, None]
    rank = (same_class & (smaller_loss | (equal_loss & earlier_row))).sum(axis=1)
    class_size = same_class.sum(axis=1)
    # float64 matches the original Python float * integer rounding rule.
    kept_count = np.maximum(1, np.floor(np.float64(retain) * class_size)).astype(np.int64)
    return rank < kept_count


def dense_peer_mask(loss: np.ndarray, positive: np.ndarray, retain: float) -> np.ndarray:
    """Return the stable low-loss mask within each finding/class stratum.

    Every nonempty stratum keeps max(1, floor(retain * stratum_size)) cells.
    Consequently retain=0 still keeps one cell per nonempty stratum; retain=1
    keeps every cell. The output is Boolean and has the same fixed shape.
    Inputs are never mutated. No variable-length gather or sorted subarray is
    used in the implementation.
    """
    classes = _positive_matrix(positive)
    values = _loss_matrix(loss, classes.shape)
    return _dense_mask(values, classes, _retain_value(retain))


def random_stratified_mask(positive: np.ndarray, retain: float, *, seed: int) -> np.ndarray:
    """Return a seeded random control with the same per-stratum kept counts.

    NumPy PCG64 generates one random priority per row/finding. Dense stable
    ranking resolves any priority ties by original row order. The mask matches
    the low-loss mask's counts for every column/class, not its cell identities.
    Same inputs and seed reproduce the mask within the recorded NumPy version.
    The generator is private, so global NumPy RNG state is unaffected.
    """
    classes = _positive_matrix(positive)
    value = _retain_value(retain)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise TypeError("seed must be a nonnegative integer, not a Boolean")
    if not 0 <= int(seed) <= 2**64 - 1:
        raise ValueError("seed must be in [0, 2**64 - 1]")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    priorities = rng.random(classes.shape)
    return _dense_mask(priorities, classes, value)


def selected_support(mask: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Return shape (findings, 2): retained negative and positive cell counts."""
    classes = _positive_matrix(positive)
    weights = np.asarray(mask)
    if weights.shape != classes.shape or weights.dtype != np.bool_:
        raise ValueError("mask must be Boolean and match positive's shape")
    return np.stack([(weights & ~classes).sum(axis=0),
                     (weights & classes).sum(axis=0)], axis=1)


def dense_peer_mask_torch(loss, positive, retain: float):
    """Optional lazy Torch counterpart for already-installed Torch only.

    This is an isolated tensor method, not a trainer, device adapter or tested
    TPU/CUDA implementation. Contract checks may synchronize an accelerator;
    no performance benefit is claimed. The core uses only fixed-shape tensors.
    """
    import torch
    value = _retain_value(retain)
    if not isinstance(loss, torch.Tensor) or not isinstance(positive, torch.Tensor):
        raise TypeError("loss and positive must be Torch tensors")
    if positive.ndim != 2 or min(positive.shape) <= 0 or loss.shape != positive.shape:
        raise ValueError("loss and positive require identical nonempty (batch, findings) shape")
    if positive.dtype != torch.bool:
        raise TypeError("positive must be Boolean")
    if loss.device != positive.device:
        raise ValueError("loss and positive must be on the same device")
    supported = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                 torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if loss.dtype not in supported:
        raise TypeError("Unsupported Torch loss dtype; supported: uint8, int8/int16/int32/int64, float16/bfloat16/float32/float64")
    if not torch.isfinite(loss).all().item():
        raise ValueError("loss values must all be finite")
    same_class = positive[:, None, :] == positive[None, :, :]
    smaller = loss[None, :, :] < loss[:, None, :]
    equal = loss[None, :, :] == loss[:, None, :]
    row = torch.arange(loss.shape[0], device=loss.device)
    earlier = row[None, :, None] < row[:, None, None]
    rank = (same_class & (smaller | (equal & earlier))).sum(dim=1)
    class_size = same_class.sum(dim=1)
    kept_count = torch.floor(class_size.to(torch.float64) * value).clamp_min(1).to(torch.int64)
    return rank < kept_count
