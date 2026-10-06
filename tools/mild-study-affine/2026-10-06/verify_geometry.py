"""Meaningful synthetic geometry/RNG checks; no models or competition data."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import mild_affine as affine


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def expect_hold(operation, message):
    try:
        operation()
    except affine.AffineHold:
        return
    raise RuntimeError(message)


def independent_forward(params, height, width):
    """Physical 3x3 map, inverted by a linear solver rather than theta algebra."""
    radians = params.angle_deg * math.pi / 180.0
    c, s = math.cos(radians), math.sin(radians)
    forward = np.eye(3, dtype=np.float64)
    forward[:2, :2] = params.scale * np.array(((c, -s), (s, c)))
    center = np.array(((width - 1) / 2.0, (height - 1) / 2.0))
    translation = np.array((params.tx_frac * width, params.ty_frac * height))
    forward[:2, 2] = center + translation - forward[:2, :2] @ center
    return forward


def independent_source_coordinates(params, height, width):
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    destination = np.stack((x.ravel(), y.ravel(), np.ones(height * width)))
    source = np.linalg.solve(independent_forward(params, height, width), destination)
    return source[0].reshape(height, width), source[1].reshape(height, width)


def theta_source_coordinates(params, height, width):
    y, x = np.meshgrid(2 * (np.arange(height) + .5) / height - 1,
                       2 * (np.arange(width) + .5) / width - 1, indexing="ij")
    source = affine.inverse_theta(params, height, width) @ np.stack((x.ravel(), y.ravel(), np.ones(height * width)))
    return (((source[0] + 1) * width - 1) / 2).reshape(height, width), (((source[1] + 1) * height - 1) / 2).reshape(height, width)


def run_checks():
    started = time.perf_counter()
    checks = []
    combinations = tuple(affine.AffineParams(*values) for values in itertools.product(
        (-5., 0., 5.), (-.02, 0., .02), (-.02, 0., .02), (.98, 1., 1.02)))
    center_values, missing_values, extent_values = [], [], []
    max_inverse_error = 0.0
    for params in combinations:
        # Rectangular, physical-pixel oracle catches omitted aspect-ratio terms.
        sx, sy = independent_source_coordinates(params, 60, 100)
        tx, ty = theta_source_coordinates(params, 60, 100)
        max_inverse_error = max(max_inverse_error, float(np.max(np.abs(sx-tx))), float(np.max(np.abs(sy-ty))))
        require(max_inverse_error < 1e-10, "rectangular inverse disagrees with independent homogeneous solve")
        require(np.linalg.det(affine.inverse_theta(params)[:, :2]) > 0, "orientation reversal/flip")
        px, py = independent_source_coordinates(params, 336, 336)
        center = float(np.mean((px < -1e-10) | (px > 335+1e-10) | (py < -1e-10) | (py > 335+1e-10)))
        wx = np.maximum(0, np.minimum(1, px + 1)) * np.maximum(0, np.minimum(1, 336 - px))
        wy = np.maximum(0, np.minimum(1, py + 1)) * np.maximum(0, np.minimum(1, 336 - py))
        missing = float(np.mean(1 - wx * wy))
        measured = affine.padding_metrics(params)
        require(abs(center - measured["source_center_outside_fraction"]) < 1e-14, "center padding definition mismatch")
        require(abs(missing - measured["mean_missing_bilinear_mass"]) < 1e-12, "bilinear missing-mass oracle mismatch")
        require(center <= .10, "finite 81-point screen exceeds hold threshold")
        center_values.append(center)
        missing_values.append(missing)
        extent_values.append(measured["source_extent_outside_fraction"])
    checks += ["81_point_screen_not_continuous_maximum_proof", "independent_rectangular_pixel_geometry", "orientation_preserved", "distinct_padding_metrics"]

    center = np.array((49.5, 29.5, 1.0))
    right = center + np.array((20.0, 0.0, 0.0))
    turned = independent_forward(affine.AffineParams(5., 0., 0., 1.), 60, 100) @ right
    require(turned[1] > center[1] and turned[0] < right[0], "positive angle does not move right landmark clockwise/down")
    shifted = independent_forward(affine.AffineParams(0., .02, -.02, 1.), 60, 100) @ center
    require(np.allclose(shifted-center, (2., -1.2, 0.), atol=1e-12, rtol=0), "translation is not a fraction of physical dimensions")
    scaled = independent_forward(affine.AffineParams(0., 0., 0., 1.02), 60, 100) @ right
    require(abs(np.linalg.norm(scaled[:2]-center[:2])-20.4) < 1e-12, "forward scale is not center-based magnification")
    checks += ["clockwise_landmark_direction", "physical_dimension_translation", "center_based_scale"]

    numpy_before = np.random.get_state()
    reference = affine.sample_mild_affine("synthetic-study-α", 3, 2026100623)
    samples = [affine.sample_mild_affine(f"synthetic-{i}", i % 5, 2026100623) for i in range(500)]
    numpy_after = np.random.get_state()
    require(numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1]) and numpy_before[2:] == numpy_after[2:], "sampling consumes global NumPy RNG")
    require(reference == affine.sample_mild_affine("synthetic-study-α", 3, 2026100623), "same UID/epoch/seed changed")
    require(len({affine.study_seed(f"synthetic-{i}", i % 5, 2026100623) for i in range(500)}) == 500, "synthetic seed collisions")
    require(reference != affine.sample_mild_affine("synthetic-study-α", 4, 2026100623), "epoch not represented")
    require(reference != affine.sample_mild_affine("synthetic-study-β", 3, 2026100623), "UID not represented")
    require(reference != affine.sample_mild_affine("synthetic-study-α", 3, 2026100624), "run seed not represented")
    reordered = {i: affine.sample_mild_affine(f"synthetic-{i}", i % 5, 2026100623) for i in reversed(range(500))}
    require(all(samples[i] == reordered[i] for i in range(500)), "sampling depends on call order")
    checks += ["owned_pcg64_global_numpy_rng_preserved", "uid_epoch_run_seed_identity", "500_synthetic_distinct_seeds", "sampling_order_independent"]

    child_code = (
        "import sys,json,dataclasses;sys.path.insert(0,sys.argv[1]);import mild_affine as m;"
        "print(json.dumps({'sample':dataclasses.asdict(m.sample_mild_affine('synthetic-study-α',3,2026100623)),"
        "'seed':str(m.study_seed('synthetic-study-α',3,2026100623)),'torch_imported':'torch' in sys.modules},sort_keys=True))"
    )
    cross_process = []
    for hash_seed in ("0", "7", "987654"):
        environment = dict(os.environ)
        environment["PYTHONHASHSEED"] = hash_seed
        child = subprocess.run((sys.executable, "-B", "-c", child_code, str(Path(__file__).resolve().parent)),
                               check=True, capture_output=True, text=True, env=environment, timeout=30)
        cross_process.append(json.loads(child.stdout))
    require(all(item == cross_process[0] for item in cross_process), "cross-process/PYTHONHASHSEED drift")
    require(cross_process[0]["sample"] == dataclasses.asdict(reference), "child differs from parent")
    require(not cross_process[0]["torch_imported"], "pure helper import eagerly loads Torch")
    checks += ["three_cross_process_hashseed_invariance", "torch_optional_at_import"]

    import torch
    runtime = affine.synthetic_runtime_check("cpu")
    checks += ["installed_torch_synthetic_runtime_attestation", "outer_autocast_dtype_precision_preserved", "global_torch_rng_preserved"]
    max_ramp_error = 0.0
    height, width = 100, 150
    y, x = torch.meshgrid(torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64), indexing="ij")
    pattern = .2 + .2*x/(width-1) + .3*y/(height-1)
    plane = pattern.expand(12, 3, height, width).clone()
    shape = (12, 3, height, width)
    for params in (affine.AffineParams(5., .02, -.02, .98), affine.AffineParams(-5., -.02, .02, 1.02), affine.IDENTITY):
        transformed = affine.apply_mild_affine(plane, params, shape)
        sx, sy = independent_source_coordinates(params, height, width)
        interior = (sx >= 1) & (sx <= width-2) & (sy >= 1) & (sy <= height-2)
        expected = .2 + .2*sx/(width-1) + .3*sy/(height-1)
        error = float(np.max(np.abs(transformed[0, 0].numpy()[interior]-expected[interior])))
        max_ramp_error = max(max_ramp_error, error)
        require(error < 1e-12, "Float64 bilinear ramp disagrees with independent physical inverse")
        require(torch.equal(transformed, transformed[:1, :1].expand_as(transformed)), "window/channel grid not coherent")
    checks += ["float64_independent_ramp_interpolation", "all_36_plane_grid_coherence"]

    # Integer physical translation of an impulse must preserve location and sign.
    impulse = torch.zeros((12, 3, 336, 336), dtype=torch.float32)
    impulse[:, :, 127, 201] = 1.
    moved = affine.apply_mild_affine(impulse, affine.AffineParams(0., 4/336, -3/336, 1.))
    peak = np.unravel_index(int(torch.argmax(moved[0, 0]).item()), (336, 336))
    require(peak == (124, 205), "impulse moved in wrong physical direction/location")
    require(abs(float(moved[0, 0, 124, 205])-1.) < 1e-4, "integer impulse interpolation mismatch")
    checks += ["independent_impulse_translation_direction"]

    bshape = (12, 3, 64, 96)
    bplane = torch.linspace(0, 1, 64*96).reshape(1, 1, 64, 96).expand(bshape).clone()
    batch = torch.stack((bplane, bplane*.7))
    normal = affine.apply_study_batch(batch, ("synthetic-first", "synthetic-second"), 2, 2026100623, bshape)
    reverse = affine.apply_study_batch(batch.flip(0), ("synthetic-second", "synthetic-first"), 2, 2026100623, bshape).flip(0)
    require(torch.equal(normal, reverse), "batch order changes UID-associated transforms")
    multipliers = torch.arange(1, 37, dtype=torch.float32).reshape(12, 3, 1, 1)/36
    coherent = affine.apply_mild_affine(bplane*multipliers, affine.AffineParams(3., .01, -.01, 1.01), bshape)
    baseline = affine.apply_mild_affine(bplane, affine.AffineParams(3., .01, -.01, 1.01), bshape)
    require(float((coherent-baseline*multipliers).abs().max()) < 2e-7, "distinct-plane linearity/alignment failed")
    checks += ["batch_uid_order_invariance", "distinct_plane_linear_alignment"]

    full_double = impulse.to(torch.float64)
    double_identity = affine.apply_mild_affine(full_double, affine.IDENTITY)
    double_identity_error = float((double_identity-full_double).abs().max())
    require(double_identity.dtype == torch.float64 and double_identity_error < 1e-10, "Float64 native-shape identity/dtype contract")
    checks += ["float64_dtype_native_shape_identity"]

    invalid_parameters = ((math.nan, 0., 0., 1.), (6., 0., 0., 1.), (0., .021, 0., 1.),
                          (0., 0., -.021, 1.), (0., 0., 0., .97), (0., 0., 0., math.inf),
                          (True, 0., 0., 1.))
    for values in invalid_parameters:
        expect_hold(lambda values=values: affine.AffineParams(*values), "invalid parameter accepted")
    for uid, epoch, seed in (("", 0, 1), (1, 0, 1), ("s", -1, 1), ("s", True, 1), ("s", 0, -1), ("s", 0, 2**64)):
        expect_hold(lambda uid=uid, epoch=epoch, seed=seed: affine.study_seed(uid, epoch, seed), "invalid seed input accepted")
    good = torch.zeros((1,12,3,336,336), dtype=torch.float32)
    expect_hold(lambda: affine.apply_parameter_batch(good.half(), (affine.IDENTITY,)), "Float16 accepted")
    expect_hold(lambda: affine.apply_parameter_batch(good.to(torch.bfloat16), (affine.IDENTITY,)), "BFloat16 accepted")
    expect_hold(lambda: affine.apply_parameter_batch(good[:, :, :1], (affine.IDENTITY,)), "wrong adjacent-channel shape accepted")
    expect_hold(lambda: affine.apply_parameter_batch(good, ()), "wrong parameter count accepted")
    expect_hold(lambda: affine.apply_study_batch(good, "synthetic", 1, 1), "bare UID string accepted as batch sequence")
    for bad_value in (-.001, 1.001, math.nan, math.inf):
        bad = good.clone()
        bad[0,0,0,0,0] = bad_value
        expect_hold(lambda bad=bad: affine.apply_parameter_batch(bad, (affine.IDENTITY,)), "invalid raw input accepted")
    tiny = torch.zeros((12,3,4,4), dtype=torch.float32)
    expect_hold(lambda: affine.apply_mild_affine(tiny, affine.AffineParams(5.,.02,.02,.98),(12,3,4,4)), "actual-grid >.10 outside fraction accepted")
    checks += ["seven_invalid_parameter_refusals", "six_invalid_seed_refusals", "explicit_dtype_shape_uid_count_refusals", "four_nonfinite_raw_bounds_refusals", "actual_grid_padding_hold"]

    return {
        "status": "PASS_SYNTHETIC_GEOMETRY_ONLY",
        "real_ids_images_targets_models_used": False,
        "named_checks": checks,
        "named_check_count": len(checks),
        "parameter_combination_count": len(combinations),
        "source_center_outside_fraction_max_81_points": max(center_values),
        "source_extent_outside_fraction_max_81_points": max(extent_values),
        "mean_missing_bilinear_mass_max_81_points": max(missing_values),
        "continuous_range_maximum_proven": False,
        "every_applied_grid_enforces_center_fraction_limit": .10,
        "independent_rectangular_inverse_max_abs_error_pixels": max_inverse_error,
        "independent_float64_ramp_max_abs_error": max_ramp_error,
        "float64_native_shape_identity_max_abs_error": double_identity_error,
        "runtime": runtime,
        "wall_seconds": time.perf_counter()-started,
        "python_version": sys.version.split()[0],
        "optimized_python": not __debug__,
        "helper_sha256": hashlib.sha256(Path(affine.__file__).read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, sort_keys=True))
