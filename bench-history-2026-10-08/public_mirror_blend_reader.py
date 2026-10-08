# SPDX-License-Identifier: Apache-2.0
"""Bee V15 blend orchestrator: V14 r384 plain/mirror pair plus the r224 single-view arm.

One owned run. The r384 path calls the unchanged V14 reader functions (checkpoint, model,
predict_pair, build_volume), so its probabilities are the V14 arithmetic. The r224 arm reads
the same 96-slice volume and runs its own transform (r224_arm.py). Fault policy as V14: a
study whose volume cannot be built is predicted from zeros, a failed r384 model call gives
0.5 pair probabilities, and a failed r224 call gives 0.5 r224 probabilities for that study.
If the r224 checkpoint is absent, not byte-identical, not strictly loadable or not loadable on
every device, the r224 arm is disabled for the whole run and the submission is the r384
ordinal rank alone (receipted as FALLBACK).

Blend (Codex rank_blend, secondary_weight 0.25): final = ordinal_rank(0.75 * ordinal_rank(r384
pair average) + 0.25 * ordinal_rank(r224)), per finding, exact study order. No per-label weights,
no quality-based selection, no joins or repairs.

Outputs go only to the given output directory (new files; refuses to overwrite). The launcher
exports submission.csv to the Kaggle root after the owned command completes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BLEND_MODE = "R384_0.75_R224_0.25_ORDINAL_RANK_BLEND_V15"
FALLBACK_MODE = "R384_ONLY_ORDINAL_RANK_ARM_FALLBACK"
PUBLIC_REFERENCE_R384 = 0.949
PUBLIC_REFERENCE_R224 = 0.945


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _device_index(device):
    return int(str(device).split(":")[1]) if str(device).startswith("cuda:") else None


def production_hooks(reader, arm, deps):
    """Real CUDA hooks. Every call below is exercised only in the owned Kaggle run."""
    _, np, _, _, _, timm, torch = deps
    require(torch.cuda.is_available(), "This experiment requires CUDA/fp16")
    count = int(torch.cuda.device_count())
    require(count >= 1, "At least one CUDA device is required")

    def predict_r224(model, volume, mask, device):
        windows = arm.prepare_windows(volume, mask, device, np, torch)
        return arm.predict(model, windows, np, torch, str(device).startswith("cuda"))

    def reset_peak(device):
        index = _device_index(device)
        if index is not None:
            torch.cuda.reset_peak_memory_stats(index)

    def peak(device):
        index = _device_index(device)
        if index is None:
            return {}
        return {"max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                "device_name": torch.cuda.get_device_name(index)}

    return dict(
        devices=[f"cuda:{index}" for index in range(count)],
        find_checkpoint=reader.find_attached_checkpoint,
        load_r384=lambda checkpoint, device: reader.make_model(checkpoint, device, timm, torch),
        predict_r384=lambda model, volume, mask, device: reader.predict_pair(model, volume, mask, device, np, torch),
        prepare_arm=lambda input_root, devices: arm.prepare_arm(input_root, devices, timm, torch),
        predict_r224=predict_r224,
        reset_peak=reset_peak,
        peak=peak,
        versions=dict(torch=torch.__version__, timm=timm.__version__, numpy=np.__version__),
        names=[torch.cuda.get_device_name(index) for index in range(count)],
    )


def _write_csv(path, ranks, ids, pd, np, columns):
    frame = pd.DataFrame(np.asarray(ranks, dtype=np.float32), columns=columns[1:])
    frame.insert(0, "StudyInstanceUID", ids)
    frame = frame[columns]
    with path.open("x") as stream:
        frame.to_csv(stream, index=False)
    return _sha_file(path)


def run_blend(args, reader, arm, hooks=None):
    """Inference over the full competition root. Writes receipts and CSVs into args.output_dir."""
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    deps = reader.dependencies()
    _, np, pd, _, _, _, torch_module = deps
    if hooks is None:
        hooks = production_hooks(reader, arm, deps)
    devices = list(hooks["devices"])
    require(bool(devices), "At least one device is required")
    for device in devices:
        hooks["reset_peak"](device)
    checkpoint = hooks["find_checkpoint"](args.input_root)
    root = args.competition_root
    series_root = root / "test_series"
    if not series_root.is_dir():
        series_root = root / "test_images"
    require(series_root.is_dir(), "Test DICOM directory is absent")
    test = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    all_ids = test["StudyInstanceUID"].tolist()
    columns = ["StudyInstanceUID", *reader.LABELS]
    require(len(set(all_ids)) == len(all_ids), "Duplicate test study ID")
    require(sample.columns.tolist() == columns, "Sample submission label/schema drift")
    require(sample["StudyInstanceUID"].tolist() == all_ids, "Test/sample row order differs")
    ids = all_ids if args.mode == "inference" else all_ids[:args.max_studies]
    require(bool(ids), "No studies selected")
    series = pd.read_csv(root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    grouped = {key: frame.to_dict("records") for key, frame in series.groupby("StudyInstanceUID", sort=False)}
    studies_without_series = sum(1 for uid in ids if uid not in grouped)

    r384_models = {device: hooks["load_r384"](checkpoint, device) for device in devices}
    arm_models, arm_status = hooks["prepare_arm"](args.input_root, devices)
    arm_on = arm_models is not None

    jobs = queue.Queue(maxsize=4)
    results, r384_fallbacks, r224_fallbacks, consumer_errors = {}, [], [], []
    lock = threading.Lock()

    def consumer(device):
        while True:
            item = jobs.get()
            try:
                if item is None:
                    return
                study_id, volume, mask, counters = item
                fallback = None
                try:
                    values, plain, mirrored = hooks["predict_r384"](r384_models[device], volume, mask, device)
                except Exception as error:
                    values = plain = mirrored = np.full(12, 0.5, dtype=np.float32)
                    fallback = {"study_id": study_id, "device": device, "error_type": type(error).__name__,
                                "message": str(error)[:300]}
                    try:
                        torch_module.cuda.empty_cache()
                    except Exception:
                        pass
                record = dict(values=values, plain=plain, mirrored=mirrored, counters=counters,
                              r224=None, r224_attempt=None, r224_seconds=None, device=device)
                if arm_on:
                    t0 = time.perf_counter()
                    try:
                        probabilities, attempt = hooks["predict_r224"](arm_models[device], volume, mask, device)
                        record.update(r224=probabilities, r224_attempt=attempt)
                    except Exception as error:
                        record.update(r224=np.full(12, 0.5, dtype=np.float32), r224_attempt="DEFAULT_0.5")
                        with lock:
                            r224_fallbacks.append({"study_id": study_id, "device": device,
                                                   "error_type": type(error).__name__, "message": str(error)[:300]})
                    record["r224_seconds"] = time.perf_counter() - t0
                with lock:
                    results[study_id] = record
                    if fallback is not None:
                        r384_fallbacks.append(fallback)
                    count = len(results)
                    if count % 25 == 0 or count == len(ids):
                        print(json.dumps({"completed": count, "total": len(ids),
                                          "elapsed_s": round(time.monotonic() - started, 2),
                                          "r384_fallbacks": len(r384_fallbacks),
                                          "r224_fallbacks": len(r224_fallbacks)}), flush=True)
            except Exception as error:
                with lock:
                    consumer_errors.append({"study_id": item[0] if item else None,
                                            "error_type": type(error).__name__, "message": str(error)[:300]})
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=consumer, args=(device,)) for device in devices]
    for thread in threads:
        thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.prep_workers) as executor:
            for offset in range(0, len(ids), 8):
                batch = ids[offset:offset + 8]
                futures = [executor.submit(reader.build_volume, uid, grouped.get(uid, []), series_root, deps)
                           for uid in batch]
                for uid, future in zip(batch, futures):
                    try:
                        volume, mask, counters = future.result()
                    except Exception as error:
                        volume, mask, counters = reader._zero_study(np, reader._new_counters())
                        counters.update(study_build_failed=True, study_build_error_type=type(error).__name__,
                                        study_build_error=str(error)[:300])
                    jobs.put((uid, volume, mask, counters))
    finally:
        for _ in threads:
            jobs.put(None)
        jobs.join()
        for thread in threads:
            thread.join()

    for uid in ids:
        if uid not in results:
            default = np.full(12, 0.5, dtype=np.float32)
            results[uid] = dict(values=default, plain=default, mirrored=default, counters=reader._new_counters(),
                                r224=np.full(12, 0.5, dtype=np.float32) if arm_on else None,
                                r224_attempt="MISSING_RESULT_DEFAULT" if arm_on else None,
                                r224_seconds=None, device=None)
            r384_fallbacks.append({"study_id": uid, "device": None, "error_type": "MissingResult",
                                   "message": "no consumer result; reference default applied"})
    require(set(results) == set(ids) and len(results) == len(ids), "Incomplete study coverage")

    pair = np.stack([results[uid]["values"] for uid in ids]).astype(np.float32)
    plain = np.stack([results[uid]["plain"] for uid in ids]).astype(np.float32)
    mirrored = np.stack([results[uid]["mirrored"] for uid in ids]).astype(np.float32)
    require(pair.shape == (len(ids), 12) and bool(np.isfinite(pair).all())
            and bool(((pair >= 0) & (pair <= 1)).all()), "R384 output shape/finite/range gate")
    r384_ranks = arm.ordinal_ranks(pair, np)
    if arm_on:
        r224 = np.stack([results[uid]["r224"] for uid in ids]).astype(np.float32)
        require(r224.shape == (len(ids), 12) and bool(np.isfinite(r224).all())
                and bool(((r224 >= 0) & (r224 <= 1)).all()), "R224 output shape/finite/range gate")
        r224_ranks = arm.ordinal_ranks(r224, np)
        final = arm.ordinal_ranks(arm.blend_ranks(r384_ranks, r224_ranks, np), np)
        blend_mode = BLEND_MODE
    else:
        r224, r224_ranks = None, None
        final = r384_ranks
        blend_mode = FALLBACK_MODE

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_sha = _write_csv(args.output_dir / "submission.csv", final, ids, pd, np, columns)
    r384_sha = _write_csv(args.output_dir / "submission_r384_only.csv", r384_ranks, ids, pd, np, columns)
    r224_sha = None
    if arm_on:
        r224_sha = _write_csv(args.output_dir / "submission_r224_only.csv", r224_ranks, ids, pd, np, columns)
    raw_path = args.output_dir / "probabilities_v15.npz"
    with raw_path.open("xb") as stream:
        arrays = dict(study_uids=np.asarray(ids, dtype=str), labels=np.asarray(reader.LABELS),
                      r384_plain=plain, r384_mirrored=mirrored, r384_pair_average=pair)
        if arm_on:
            arrays["r224_probability"] = r224
        np.savez_compressed(stream, **arrays)

    seconds = [results[uid]["r224_seconds"] for uid in ids if results[uid]["r224_seconds"] is not None]
    attempts = {}
    for uid in ids:
        name = results[uid]["r224_attempt"]
        if name is not None:
            attempts[name] = attempts.get(name, 0) + 1
    peaks = {str(device): hooks["peak"](device) for device in devices}
    receipt = {
        "mode": args.mode, "full_test_coverage": ids == all_ids, "studies": len(ids),
        "native_inference_completed": True, "competition_submitted": False, "official_score": None,
        "public_reference_r384": PUBLIC_REFERENCE_R384, "public_reference_r224": PUBLIC_REFERENCE_R224,
        "public_ref_r384": "nartaa/rsna-knee-0949-anatomical-mirror", "public_ref_r224": arm.PUBLIC_REF,
        "checkpoint_sha256": reader.CHECKPOINT_SHA256, "checkpoint_name": Path(checkpoint).name,
        "dataset_ref": reader.DATASET_REF, "dataset_version": reader.DATASET_VERSION,
        "r224_checkpoint_sha256_expected": arm.CHECKPOINT_SHA256,
        "r224_checkpoint_sha256_verified": bool(arm_status["checkpoint_sha256_verified"]),
        "r224_checkpoint_bytes_observed": arm_status["checkpoint_bytes_observed"],
        "r224_arm_enabled": bool(arm_on), "r224_strict_load_ok": bool(arm_status["strict_load_ok"]),
        "r224_fallback_reason": arm_status["reason"],
        "r224_study_fallbacks": len(r224_fallbacks), "r224_study_fallback_examples": r224_fallbacks[:5],
        "r224_attempt_counts": attempts,
        "r224_elapsed_s_per_study": [results[uid]["r224_seconds"] for uid in ids],
        "r224_elapsed_s": dict(count=len(seconds), total=float(sum(seconds)) if seconds else None,
                               mean=float(sum(seconds) / len(seconds)) if seconds else None,
                               min=float(min(seconds)) if seconds else None,
                               max=float(max(seconds)) if seconds else None),
        "blend_mode": blend_mode, "blend_weight_r384": arm.R384_WEIGHT if arm_on else 1.0,
        "blend_weight_r224": arm.R224_WEIGHT if arm_on else 0.0,
        "reader_source_sha256": _sha_file(reader.__file__), "r224_source_sha256": _sha_file(arm.__file__),
        "source_sha256": _sha_file(__file__),
        "r384_only_csv_sha256": r384_sha, "r224_only_csv_sha256": r224_sha,
        "prediction_file_sha256": final_sha, "probabilities_file_sha256": _sha_file(raw_path),
        "versions": hooks["versions"], "device_count": len(devices), "devices": hooks["names"],
        "peak_cuda_memory": peaks,
        "elapsed_s": time.monotonic() - started, "prep_workers": args.prep_workers,
        "fault_policy": reader.FAULT_POLICY,
        "missing_slots": sum(results[uid]["counters"]["missing_slots"] for uid in ids),
        "selected_slices": sum(results[uid]["counters"]["selected_slices"] for uid in ids),
        "header_failures": sum(results[uid]["counters"]["header_failures"] for uid in ids),
        "pixel_failures": sum(results[uid]["counters"]["pixel_failures"] for uid in ids),
        "study_build_failures": sum(1 for uid in ids if results[uid]["counters"]["study_build_failed"]),
        "studies_without_series_metadata": studies_without_series,
        "model_fallback_studies": len(r384_fallbacks), "model_fallback_examples": r384_fallbacks[:5],
        "consumer_error_count": len(consumer_errors), "consumer_error_examples": consumer_errors[:5],
        "arm_status": arm_status,
    }
    receipt_path = args.output_dir / "native_inference_receipt.json"
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: receipt[k] for k in ("studies", "blend_mode", "r224_arm_enabled", "model_fallback_studies",
                                              "r224_study_fallbacks", "elapsed_s")}, sort_keys=True), flush=True)
    return receipt
