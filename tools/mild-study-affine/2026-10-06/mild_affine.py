"""Original study-coherent mild affine augmentation, before normalization.

No Torch import occurs until a tensor is applied.  Parameter sampling uses only
an owned PCG64 generator; it does not consume NumPy or Torch global RNG state.
Positive angles rotate clockwise in image coordinates (x right, y down).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import numbers
from typing import Sequence

import numpy as np


EXPECTED_STUDY_SHAPE = (12, 3, 336, 336)
MAX_CENTER_OUTSIDE_FRACTION = 0.10
SEED_DOMAIN = b"knee-mild-affine-pcg64-v1\x00"


class AffineHold(ValueError):
    """A violated treatment or raw-input contract; do not silently recover."""


@dataclass(frozen=True)
class AffineParams:
    angle_deg: float
    tx_frac: float
    ty_frac: float
    scale: float

    def __post_init__(self):
        for name, low, high in (
            ("angle_deg", -5.0, 5.0),
            ("tx_frac", -0.02, 0.02),
            ("ty_frac", -0.02, 0.02),
            ("scale", 0.98, 1.02),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise AffineHold(f"{name} must be a real number")
            value = float(value)
            if not math.isfinite(value) or not low <= value <= high:
                raise AffineHold(f"{name} outside the frozen treatment bounds")
            object.__setattr__(self, name, value)


IDENTITY = AffineParams(0.0, 0.0, 0.0, 1.0)


def _integer(value, name, upper=None):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise AffineHold(f"{name} must be an integer")
    value = int(value)
    if value < 0 or (upper is not None and value >= upper):
        raise AffineHold(f"{name} outside the permitted integer range")
    return value


def study_seed(study_uid: str, epoch: int, run_seed: int) -> int:
    """Derive a stable 128-bit seed from exact UTF-8 UID, epoch, and run seed.

    No worker ID, arm name, network index, Python hash(), time, global RNG,
    normalization, or data-dependent content participates in this derivation.
    JSON fixes boundaries; the domain and little-endian digest conversion fix
    the algorithm.  Unicode UIDs are accepted without silent normalization.
    """
    if not isinstance(study_uid, str) or not study_uid:
        raise AffineHold("study_uid must be a nonempty exact string")
    epoch = _integer(epoch, "epoch")
    run_seed = _integer(run_seed, "run_seed", 2**64)
    encoded = json.dumps(
        [run_seed, epoch, study_uid], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8", errors="strict")
    return int.from_bytes(hashlib.sha256(SEED_DOMAIN + encoded).digest()[:16], "little")


def sample_mild_affine(study_uid: str, epoch: int, run_seed: int) -> AffineParams:
    """Exactly four owned PCG64 uniform draws in the documented order."""
    rng = np.random.Generator(np.random.PCG64(study_seed(study_uid, epoch, run_seed)))
    return AffineParams(
        angle_deg=rng.uniform(-5.0, 5.0),
        tx_frac=rng.uniform(-0.02, 0.02),
        ty_frac=rng.uniform(-0.02, 0.02),
        scale=rng.uniform(0.98, 1.02),
    )


def _dimensions(height, width):
    height, width = _integer(height, "height"), _integer(width, "width")
    if height < 2 or width < 2:
        raise AffineHold("height and width must be at least two")
    return height, width


def inverse_theta(params: AffineParams, height=336, width=336) -> np.ndarray:
    """Float64 normalized output-to-input map for align_corners=False.

    In physical pixel coordinates the FORWARD map is
      output-center = scale * [[cos,-sin],[sin,cos]] * (input-center)
                      + [tx_frac*width, ty_frac*height].
    The center is ((width-1)/2, (height-1)/2).  Aspect-ratio factors below
    convert physical pixels to Torch's normalized pixel-center coordinates.
    """
    if not isinstance(params, AffineParams):
        raise AffineHold("params must be validated AffineParams")
    height, width = _dimensions(height, width)
    angle = math.radians(params.angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    hw, wh, scale = height / width, width / height, params.scale
    return np.array(
        [
            [c / scale, hw * s / scale,
             -2.0 * (c * params.tx_frac + hw * s * params.ty_frac) / scale],
            [-wh * s / scale, c / scale,
             2.0 * (wh * s * params.tx_frac - c * params.ty_frac) / scale],
        ],
        dtype=np.float64,
    )


def padding_metrics(params: AffineParams, height=336, width=336) -> dict:
    """Synthetic geometry only: center outside, extent outside, missing mass.

    Center-outside is the governing conservative metric: inverse-mapped
    coordinates outside [0,width-1] x [0,height-1].  Missing bilinear mass
    measures the mean zero-padding interpolation weight, a different quantity.
    Float64 rounding tolerance is 1e-10 physical pixel, solely for exact edges.
    """
    height, width = _dimensions(height, width)
    y, x = np.meshgrid(
        (2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height - 1.0),
        (2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0),
        indexing="ij",
    )
    theta = inverse_theta(params, height, width)
    sx = ((theta[0, 0] * x + theta[0, 1] * y + theta[0, 2] + 1.0) * width - 1.0) / 2.0
    sy = ((theta[1, 0] * x + theta[1, 1] * y + theta[1, 2] + 1.0) * height - 1.0) / 2.0
    tolerance = 1e-10
    center_outside = (sx < -tolerance) | (sx > width - 1 + tolerance) | (sy < -tolerance) | (sy > height - 1 + tolerance)
    extent_outside = (sx < -0.5 - tolerance) | (sx > width - 0.5 + tolerance) | (sy < -0.5 - tolerance) | (sy > height - 0.5 + tolerance)
    wx = np.clip(sx + 1.0, 0.0, 1.0) * np.clip(width - sx, 0.0, 1.0)
    wy = np.clip(sy + 1.0, 0.0, 1.0) * np.clip(height - sy, 0.0, 1.0)
    return {
        "source_center_outside_fraction": float(np.mean(center_outside)),
        "source_extent_outside_fraction": float(np.mean(extent_outside)),
        "mean_missing_bilinear_mass": float(np.mean(1.0 - wx * wy)),
    }


def apply_parameter_batch(raw_batch, params: Sequence[AffineParams],
                          expected_study_shape=EXPECTED_STUDY_SHAPE):
    """Apply one inverse grid per study, shared by every window and channel.

    Raw floating values must be finite in [0,1].  Float32/Float64 are explicit;
    half/BFloat16 are refused here, before any native autocast/normalization.
    Input validation occurs once for the batch.  An actual-grid center-outside
    guard uses four dtype epsilons at exact normalized boundaries; this is less
    than .0001 physical pixel for Float32 at 336 pixels.  No clipping, flips,
    hidden random draws, padding substitution, or recovery fallback occurs.
    """
    import torch
    from torch.nn import functional as F

    if not isinstance(raw_batch, torch.Tensor):
        raise AffineHold("raw_batch must be a Torch tensor")
    if raw_batch.dtype not in (torch.float32, torch.float64):
        raise AffineHold("raw input must be Float32 or Float64 before autocast")
    expected_study_shape = tuple(expected_study_shape)
    if len(expected_study_shape) != 4 or any(
        isinstance(d, bool) or not isinstance(d, numbers.Integral) or int(d) <= 0
        for d in expected_study_shape
    ):
        raise AffineHold("expected study shape must contain four positive integers")
    if raw_batch.ndim != 5 or tuple(raw_batch.shape[1:]) != expected_study_shape:
        raise AffineHold("raw input shape violates the explicit study contract")
    batch, windows, channels, height, width = raw_batch.shape
    _dimensions(height, width)
    params = tuple(params)
    if batch < 1 or len(params) != batch or any(not isinstance(p, AffineParams) for p in params):
        raise AffineHold("exactly one validated affine parameter set is required per study")
    # One synchronization for finite/bounds, rather than one per image/plane.
    valid_raw = torch.isfinite(raw_batch).all() & (raw_batch.amin() >= 0.0) & (raw_batch.amax() <= 1.0)
    if not bool(valid_raw.item()):
        raise AffineHold("raw input contains nonfinite values or values outside [0,1]")
    with torch.autocast(device_type=raw_batch.device.type, enabled=False):
        theta = torch.as_tensor(
            np.stack([inverse_theta(p, height, width) for p in params]),
            dtype=raw_batch.dtype, device=raw_batch.device,
        )
        study_grid = F.affine_grid(theta, (batch, channels, height, width), align_corners=False)
        edge_tolerance = 4.0 * torch.finfo(raw_batch.dtype).eps
        outside = ((study_grid[..., 0].abs() > 1.0 - 1.0 / width + edge_tolerance)
                   | (study_grid[..., 1].abs() > 1.0 - 1.0 / height + edge_tolerance))
        outside_fraction = outside.to(torch.float64).mean(dim=(1, 2))
        if bool((outside_fraction > MAX_CENTER_OUTSIDE_FRACTION).any().item()):
            raise AffineHold("actual inverse-grid source-center-outside fraction exceeds .10")
        # repeat_interleave applies exactly the same grid bytes to all study windows.
        window_grid = study_grid.repeat_interleave(windows, dim=0)
        result = F.grid_sample(
            raw_batch.reshape(batch * windows, channels, height, width),
            window_grid, mode="bilinear", padding_mode="zeros", align_corners=False,
        )
    result = result.reshape(batch, windows, channels, height, width)
    valid_result = torch.isfinite(result).all() & (result.amin() >= 0.0) & (result.amax() <= 1.0)
    if not bool(valid_result.item()):
        raise AffineHold("sampled output violated finite [0,1] contract")
    return result


def apply_mild_affine(raw_study, params: AffineParams,
                      expected_shape=EXPECTED_STUDY_SHAPE):
    """Single-study convenience API; returns the same shape/dtype/device."""
    import torch
    if not isinstance(raw_study, torch.Tensor):
        raise AffineHold("raw_study must be a Torch tensor")
    return apply_parameter_batch(raw_study.unsqueeze(0), (params,), expected_shape)[0]


def apply_study_batch(raw_batch, study_uids: Sequence[str], epoch: int, run_seed: int,
                      expected_study_shape=EXPECTED_STUDY_SHAPE):
    """Per-study/epoch deterministic treatment for [B,12,3,336,336] raw data."""
    if isinstance(study_uids, (str, bytes)):
        raise AffineHold("study_uids must be a sequence of exact UID strings")
    study_uids = tuple(study_uids)
    params = tuple(sample_mild_affine(uid, epoch, run_seed) for uid in study_uids)
    return apply_parameter_batch(raw_batch, params, expected_study_shape)


def synthetic_runtime_check(device="cpu") -> dict:
    """Small model-free native attestation; uses no real images, IDs, labels.

    This checks actual installed Torch operations, preserving global RNG state.
    It is a runtime correctness check, not an anatomy, speed, or score claim.
    """
    import torch
    cpu_before = torch.random.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None
    numpy_before = np.random.get_state()
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_initialized():
        raise AffineHold("initialize CUDA before the RNG-preservation attestation")
    height, width = 32, 48
    x = torch.arange(width, dtype=torch.float32, device=torch_device).reshape(1, 1, 1, width)
    ramp = (x / (width - 1)).expand(12, 3, height, width).clone()
    shape = (12, 3, height, width)
    identity = apply_mild_affine(ramp, IDENTITY, shape)
    identity_error = float((identity - ramp).abs().max().item())
    if identity_error > 1e-5:
        raise AffineHold("native Float32 identity interpolation error exceeds 1e-5")
    params = sample_mild_affine("synthetic-runtime-study", 1, 2026100623)
    transformed = apply_study_batch(ramp.unsqueeze(0), ("synthetic-runtime-study",), 1, 2026100623, shape)[0]
    plane_alignment_error = float((transformed - transformed[:1, :1]).abs().max().item())
    if plane_alignment_error != 0.0:
        raise AffineHold("native window/channel coherence failed")
    # Native-sized high-contrast input catches subpixel identity roundoff that
    # a smooth or small ramp could hide.  Float32 1e-4 is explicit, not bitwise
    # identity: affine_grid/grid_sample still interpolate for zero treatment.
    row = torch.arange(336, dtype=torch.float32, device=torch_device)
    high_contrast = ((17.0 * row[:, None] + 13.0 * row[None, :]).remainder(31.0) / 30.0).expand(12, 3, 336, 336).clone()
    high_contrast_identity = apply_mild_affine(high_contrast, IDENTITY)
    native_identity_error = float((high_contrast_identity - high_contrast).abs().max().item())
    if native_identity_error > 1e-4:
        raise AffineHold("native-sized Float32 identity interpolation error exceeds 1e-4")
    with torch.autocast(device_type=torch_device.type, enabled=True):
        under_outer_autocast = apply_study_batch(ramp.unsqueeze(0), ("synthetic-runtime-study",), 1, 2026100623, shape)[0]
    outer_autocast_error = float((under_outer_autocast - transformed).abs().max().item())
    if under_outer_autocast.dtype != torch.float32 or outer_autocast_error != 0.0:
        raise AffineHold("outer autocast changed the explicit Float32 augmentation")
    # Saturated pixels can expose one-ULP interpolation overshoot that smooth
    # ramps miss.  Keep the strict output contract: an installed kernel that
    # violates it holds before training, without clipping or a hidden tolerance.
    # Only this model-free fixture overrides windows/channels, while retaining
    # the actual native 336x336 grid and the identical application code.
    saturation_params = tuple(AffineParams(*values) for values in itertools.product(
        (-5.0, 0.0, 5.0), (-0.02, 0.0, 0.02), (-0.02, 0.0, 0.02), (0.98, 1.0, 1.02)))
    saturated_input = torch.ones((81, 1, 1, 336, 336), dtype=torch.float32, device=torch_device)
    saturated_output = apply_parameter_batch(saturated_input, saturation_params, (1, 1, 336, 336))
    saturated_min = float(saturated_output.amin().item())
    saturated_max = float(saturated_output.amax().item())
    mixed_pattern = ((row[:, None] + row[None, :]).remainder(2.0)).reshape(1, 1, 1, 336, 336)
    mixed_output = apply_parameter_batch(mixed_pattern.expand(81, 1, 1, 336, 336), saturation_params, (1, 1, 336, 336))
    mixed_min = float(mixed_output.amin().item())
    mixed_max = float(mixed_output.amax().item())
    # Interior linear ramp is exactly bilinear in source x; independent physical
    # translation gives expected x_in = x_out - tx_frac*width.
    tx = AffineParams(0.0, 0.02, 0.0, 1.0)
    translated = apply_mild_affine(ramp, tx, shape)
    expected = (x[0, 0, 0, 2:-2] - 0.02 * width) / (width - 1)
    translation_error = float((translated[0, 0, height // 2, 2:-2] - expected).abs().max().item())
    if translation_error > 1e-5:
        raise AffineHold("native independent translation interpolation check failed")
    numpy_after = np.random.get_state()
    rng_unchanged = (torch.equal(cpu_before, torch.random.get_rng_state())
                     and numpy_before[0] == numpy_after[0]
                     and np.array_equal(numpy_before[1], numpy_after[1])
                     and numpy_before[2:] == numpy_after[2:])
    if cuda_before is not None:
        cuda_after = torch.cuda.get_rng_state_all()
        rng_unchanged = rng_unchanged and len(cuda_before) == len(cuda_after) and all(torch.equal(a, b) for a, b in zip(cuda_before, cuda_after))
    if not rng_unchanged:
        raise AffineHold("synthetic runtime check changed global RNG state")
    return {
        "status": "PASS_SYNTHETIC_RUNTIME_ONLY",
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device": str(torch_device),
        "dtype": "float32",
        "identity_max_abs_error": identity_error,
        "native_shape_identity_max_abs_error": native_identity_error,
        "native_shape_identity_tolerance": 1e-4,
        "translation_max_abs_error": translation_error,
        "window_channel_alignment_max_abs_error": plane_alignment_error,
        "outer_autocast_max_abs_error": outer_autocast_error,
        "saturated_and_binary_stress_grid_count_each": 81,
        "saturated_output_min": saturated_min,
        "saturated_output_max": saturated_max,
        "binary_output_min": mixed_min,
        "binary_output_max": mixed_max,
        "strict_output_range_without_clipping": True,
        "numpy_cpu_torch_initialized_cuda_rng_unchanged": bool(rng_unchanged),
        "synthetic_parameter_center_outside_fraction": padding_metrics(params, height, width)["source_center_outside_fraction"],
        "real_data_used": False,
        "models_or_targets_used": False,
    }
