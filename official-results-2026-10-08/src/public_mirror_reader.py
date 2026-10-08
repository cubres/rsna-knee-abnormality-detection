"""Original inference implementation of the qualified public Knee reader.

Architecture and geometry: dreaddevelopment's Raptor CoAtNet/MIL lineage.
Selected learned weights and plain/mirror recipe: nartaa public V2,
https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397
Notebook source is Apache-2.0. Weights have separate research/educational terms,
including this competition. This module grants no rights to OAI source data.

Fault policy (V14): the same tolerant semantics as the public reference's
reader (its header record, per-slice pixel skip and whole-study fallback).
A header failure keeps the file as a (0.0, path, 0.5) record; a pixel failure
leaves that slot zero and advances the slot index; a study whose volume cannot
be built is predicted from a zero volume and zero mask; a model failure gives
that study 0.5 probabilities. No study is dropped, and any CUDA device count
of at least one is accepted.

Importing this file performs no model load, network request or native inference.
Only main() starts an explicitly requested local Kaggle experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import glob
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

PUBLIC_REF = "nartaa/rsna-knee-0949-anatomical-mirror"
PUBLIC_VERSION = 2
PUBLIC_SESSION_ID = 356175397
PUBLIC_NOTEBOOK_SHA256 = "8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100"
PUBLIC_INFERENCE_CELL_SHA256 = "bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5"
DATASET_REF = "nartaa/rsna-knee-publication-swa-weights-20261007"
DATASET_VERSION = 1
CHECKPOINT_NAME = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
CHECKPOINT_SHA256 = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")
SLOTS = (("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18))
CORPUS_RES = 384
INPUT_RES = 320
SLICE_COUNT = 96
WINDOW_COUNT = 94
CROP_MM = 140.0
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
FAULT_POLICY = ("V14 reference-tolerant: header failure keeps (0.0, 0.5) record; pixel failure skips slice "
                "(zero slot, index advances); study build failure gives zero volume and mask; model failure "
                "gives 0.5 probabilities; no study is dropped")
PROBABILITY_DEFAULT = 0.5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def mirror_axis_for_center(center):
    require(type(center) is int and 0 <= center < SLICE_COUNT, "Invalid centre slice")
    return "channels" if center < 48 else "width"


def find_attached_checkpoint(root):
    candidates = (
        root / DATASET_REF.split("/")[1] / CHECKPOINT_NAME,
        root / "datasets" / DATASET_REF / CHECKPOINT_NAME,
    )
    available = [path for path in candidates if path.is_file()]
    require(bool(available), "Attach the qualified public dataset V1; checkpoint is absent")
    path = available[0]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == CHECKPOINT_SHA256, "Checkpoint bytes differ from the qualified public V1")
    return path


def dependencies():
    import cv2
    import numpy as np
    import pandas as pd
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut
    import timm
    import torch
    return cv2, np, pd, pydicom, apply_modality_lut, timm, torch


def make_model(checkpoint, device, timm, torch):
    nn = torch.nn

    class FindingAttentionReader(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.norm = nn.LayerNorm(backbone.num_features)
            self.att = nn.Sequential(nn.Linear(backbone.num_features, 256), nn.Tanh(),
                                     nn.Dropout(0.2), nn.Linear(256, 12))
            self.clsW = nn.Parameter(torch.zeros(12, backbone.num_features))
            self.clsb = nn.Parameter(torch.zeros(12))
            self.n = 12

        def forward(self, windows):
            batch, count = windows.shape[:2]
            flat = windows.flatten(0, 1)
            encoded = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)
            features = self.norm(encoded.view(batch, count, -1))
            attention = torch.softmax(self.att(features), dim=1)
            pooled = torch.einsum("bkn,bkf->bnf", attention, features)
            return (pooled * self.clsW).sum(-1) + self.clsb

    # Tensor-only deserialization is intentionally stricter than the public source.
    # Its compatibility with this checkpoint must pass the native canary first.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(state, dict) and "model" in state, "Unsupported checkpoint container")
    require(state.get("arch", ARCH) == ARCH, "Checkpoint architecture differs")
    require(int(state.get("res", CORPUS_RES)) == CORPUS_RES, "Checkpoint training resolution differs")
    backbone = timm.create_model(ARCH, pretrained=False, num_classes=0, in_chans=3,
                                 img_size=INPUT_RES, global_pool="avg")
    model = FindingAttentionReader(backbone)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.eval().to(device)
    require(all(not module.training for module in model.modules()), "Model has a training module")
    return model


def _new_counters():
    return {"missing_slots": 0, "selected_slices": 0, "header_failures": 0, "pixel_failures": 0,
            "study_build_failed": False, "study_build_error_type": None, "study_build_error": None}


def _zero_study(np, counters):
    return (np.zeros((SLICE_COUNT, CORPUS_RES, CORPUS_RES), dtype=np.uint8),
            np.zeros(SLICE_COUNT, dtype=np.uint8), counters)


def build_volume(study_id, rows, series_root, deps):
    """Return (volume, mask, counters). Never raises (reference fault policy).

    Per-header and per-pixel failures are absorbed exactly where the reference
    absorbs them. Any exception that still escapes marks the whole study as
    failed: the volume and mask are zero, as in the reference's study fallback.
    """
    np = deps[1]
    try:
        return _build_volume_checked(study_id, rows, series_root, deps)
    except Exception as error:
        counters = _new_counters()
        counters.update(study_build_failed=True, study_build_error_type=type(error).__name__,
                        study_build_error=str(error)[:300])
        return _zero_study(np, counters)


def _build_volume_checked(study_id, rows, series_root, deps):
    cv2, np, _, pydicom, modality_lut, _, _ = deps
    volume = np.zeros((SLICE_COUNT, CORPUS_RES, CORPUS_RES), dtype=np.uint8)
    used = set()
    offset = 0
    counters = _new_counters()

    def header(path):
        # Reference _hdr: an unreadable header keeps the file as (0.0, path, 0.5, False).
        try:
            data = pydicom.dcmread(path, stop_before_pixels=True)
            orientation = getattr(data, "ImageOrientationPatient", None)
            position = getattr(data, "ImagePositionPatient", None)
            if orientation is not None and position is not None and len(orientation) == 6:
                normal = np.cross(np.asarray(orientation[:3], dtype=float),
                                  np.asarray(orientation[3:], dtype=float))
                coordinate = float(np.dot(np.asarray(position, dtype=float), normal))
            else:
                coordinate = float(getattr(data, "InstanceNumber", 0) or 0)
            spacing = getattr(data, "PixelSpacing", None)
            spacing = float(spacing[0]) if spacing is not None else 0.5
            return coordinate, path, spacing, True
        except Exception:
            return 0.0, path, 0.5, False

    def pixels(item):
        # Reference _px: an unreadable pixel array gives None, and that slice is skipped.
        _, path, spacing, _ = item
        try:
            data = pydicom.dcmread(path)
            image = modality_lut(data.pixel_array, data).astype(np.float32)
            if str(getattr(data, "PhotometricInterpretation", "")) == "MONOCHROME1":
                image = image.max() - image
            return image, spacing
        except Exception:
            return None, spacing

    for plane, fluid, count in SLOTS:
        candidates = [row for row in rows if row["Anatomical_Plane"] == plane
                      and str(row["SeriesInstanceUID"]) not in used]
        preferred = [row for row in candidates
                     if fluid in (0, 1) and int(row.get("Fluid_Sensitive", 0) or 0) == fluid]
        chosen = (preferred or candidates)[0] if candidates else None
        if chosen is None:
            counters["missing_slots"] += 1
            offset += count
            continue
        series_id = str(chosen["SeriesInstanceUID"])
        used.add(series_id)
        # Reference uses glob.glob on the series directory; the same listing order is kept.
        paths = glob.glob(f"{series_root}/{study_id}/{series_id}/*.dcm")
        if not paths:
            counters["missing_slots"] += 1
            offset += count
            continue
        with ThreadPoolExecutor(max_workers=16) as executor:
            records = list(executor.map(header, paths))
        counters["header_failures"] += sum(1 for record in records if not record[3])
        ok_spacings = [record[2] for record in records if record[3]]
        median_spacing = float(np.median(ok_spacings)) if ok_spacings else 0.5
        records.sort(key=lambda record: record[0])
        total = len(records)
        low = int(total * 0.02)
        high = max(int(total * 0.98) - 1, low)
        indices = np.linspace(low, high, count).round().astype(int) if total > 1 else [0] * count
        selected = [records[min(int(index), total - 1)] for index in indices]
        with ThreadPoolExecutor(max_workers=8) as executor:
            arrays = list(executor.map(pixels, selected))
        counters["pixel_failures"] += sum(1 for image, _ in arrays if image is None)
        valid_images = [image for image, _ in arrays if image is not None]
        if valid_images:
            low_value, high_value = np.percentile(np.concatenate([image.ravel() for image in valid_images]),
                                                [2.0, 98.0])
        else:
            low_value, high_value = 0.0, 1.0
        for image, spacing in arrays:
            if image is None:
                offset += 1
                continue
            normalized = np.clip((image - low_value) / (high_value - low_value + 1e-6), 0, 1)
            # A non-2D array fails the crop in the reference (unpacking h, w), so the whole study falls back.
            require(normalized.ndim == 2, "Non-2D DICOM pixel array")
            spacing = spacing if spacing > 0 else median_spacing
            size = min(int(round(CROP_MM / max(spacing, 1e-3))), *normalized.shape)
            require(size > 0, "Empty physical crop")
            top, left = ((normalized.shape[0] - size) // 2, (normalized.shape[1] - size) // 2)
            cropped = normalized[top:top + size, left:left + size]
            resized = cv2.resize(cropped, (CORPUS_RES, CORPUS_RES), interpolation=cv2.INTER_AREA)
            volume[offset] = (resized * 255).astype(np.uint8)
            offset += 1
            counters["selected_slices"] += 1
    require(offset == SLICE_COUNT, "Study slot count differs")
    mask = (volume.reshape(SLICE_COUNT, -1).sum(1) > 0).astype(np.uint8)
    return volume, mask, counters


def prepare_windows(volume, mask, device, mirrored, np, torch):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, len(volume)))
    low, high = int(valid.min()), int(valid.max())
    centers = list(range(low + 1, high))
    if not centers:
        centers = [max(1, min((low + high) // 2, SLICE_COUNT - 2))]
    indices = np.linspace(0, len(centers) - 1, WINDOW_COUNT).round().astype(int)
    centers = np.asarray([centers[int(index)] for index in indices], dtype=np.int64)
    triplets = torch.from_numpy(np.ascontiguousarray(
        volume[np.stack([centers - 1, centers, centers + 1], axis=1)])).to(device)
    require(tuple(triplets.shape) == (WINDOW_COUNT, 3, CORPUS_RES, CORPUS_RES), "Triplet shape differs")
    if mirrored:
        sagittal = torch.from_numpy(np.asarray([
            mirror_axis_for_center(int(center)) == "channels" for center in centers
        ], dtype=bool)).to(device).view(-1, 1, 1, 1)
        # Reflect uint8 acquisition channels first; ImageNet channel statistics differ.
        triplets = torch.where(sagittal, triplets.flip(1), triplets.flip(-1))
    border = (CORPUS_RES - INPUT_RES) // 2
    triplets = triplets[:, :, border:border + INPUT_RES, border:border + INPUT_RES]
    # CPU-generated LUT reproduces the public float32 normalization order exactly.
    values = torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0)
    table = torch.stack([values, values, values], dim=0).view(3, 256, 1)
    means = torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    scales = torch.tensor(IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    table = ((table - means) / scales).view(3, 256).contiguous().to(device)
    windows = torch.stack([table[channel][triplets[:, channel].long()] for channel in range(3)], dim=1)
    require(tuple(windows.shape) == (WINDOW_COUNT, 3, INPUT_RES, INPUT_RES), "Model input shape differs")
    return windows


def predict_pair(model, volume, mask, device, np, torch):
    require(all(not module.training for module in model.modules()), "TTA changed model mode")
    predictions = []
    with torch.no_grad():
        for mirrored in (False, True):
            windows = prepare_windows(volume, mask, device, mirrored, np, torch)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(windows.unsqueeze(0)).float()
                require(bool(torch.isfinite(logits).all()), "Nonfinite model logits")
                predictions.append(torch.sigmoid(logits)[0].cpu().numpy())
            del windows
    values = 0.5 * (predictions[0] + predictions[1])
    require(values.shape == (12,) and np.isfinite(values).all(), "Invalid study probabilities")
    return values, predictions[0], predictions[1]


def _device_name(torch, index):
    try:
        return torch.cuda.get_device_name(index)
    except Exception:
        return "unavailable"


def run(args):
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    deps = dependencies()
    _, np, pd, _, _, timm, torch = deps
    require(torch.cuda.is_available(), "This experiment requires CUDA/fp16")
    gpu_count = int(torch.cuda.device_count())
    require(gpu_count >= 1, "At least one CUDA device is required")
    device_names = [_device_name(torch, index) for index in range(gpu_count)]
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint = find_attached_checkpoint(args.input_root)
    root = args.competition_root
    series_root = root / "test_series"
    if not series_root.is_dir():
        series_root = root / "test_images"
    require(series_root.is_dir(), "Test DICOM directory is absent")
    test = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})
    sample = pd.read_csv(root / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    all_ids = test["StudyInstanceUID"].tolist()
    expected_columns = ["StudyInstanceUID", *LABELS]
    require(len(set(all_ids)) == len(all_ids), "Duplicate test study ID")
    require(sample.columns.tolist() == expected_columns, "Sample submission label/schema drift")
    require(sample["StudyInstanceUID"].tolist() == all_ids, "Test/sample row order differs")
    ids = all_ids if args.mode == "inference" else all_ids[:args.max_studies]
    require(bool(ids), "No studies selected")
    series = pd.read_csv(root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    grouped = {key: frame.to_dict("records") for key, frame in series.groupby("StudyInstanceUID", sort=False)}
    # A study without series rows is predicted from a zero volume, as in the reference (no drop).
    studies_without_series = sum(1 for uid in ids if uid not in grouped)
    models = [make_model(checkpoint, f"cuda:{index}", timm, torch) for index in range(gpu_count)]
    jobs = queue.Queue(maxsize=4)
    results, model_fallbacks, consumer_errors = {}, [], []
    lock = threading.Lock()

    def consumer(index):
        while True:
            item = jobs.get()
            try:
                if item is None:
                    return
                study_id, volume, mask, counters = item
                try:
                    values, plain, mirrored = predict_pair(models[index], volume, mask,
                                                           f"cuda:{index}", np, torch)
                    fallback = None
                except Exception as error:
                    # Reference arm default: a study whose model call fails keeps 0.5 probabilities.
                    values = plain = mirrored = np.full(12, PROBABILITY_DEFAULT, dtype=np.float32)
                    fallback = {"study_id": study_id, "device": f"cuda:{index}",
                                "error_type": type(error).__name__, "message": str(error)[:300]}
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                with lock:
                    results[study_id] = (values, plain, mirrored, counters)
                    if fallback is not None:
                        model_fallbacks.append(fallback)
                    count = len(results)
                    if count % 25 == 0 or count == len(ids):
                        print(json.dumps({"completed": count, "total": len(ids),
                                          "elapsed_s": round(time.monotonic() - started, 2),
                                          "model_fallbacks": len(model_fallbacks)}), flush=True)
            except Exception as error:
                # Keep the consumer alive so the producer is never stranded on a full queue.
                with lock:
                    consumer_errors.append({"study_id": item[0] if item else None,
                                            "error_type": type(error).__name__, "message": str(error)[:300]})
            finally:
                jobs.task_done()

    threads = [threading.Thread(target=consumer, args=(index,)) for index in range(gpu_count)]
    for thread in threads:
        thread.start()
    try:
        with ThreadPoolExecutor(max_workers=args.prep_workers) as executor:
            # At most eight queued preparation futures; no fork after Torch/CUDA initialization.
            for offset in range(0, len(ids), 8):
                batch = ids[offset:offset + 8]
                futures = [executor.submit(build_volume, uid, grouped.get(uid, []), series_root, deps) for uid in batch]
                for uid, future in zip(batch, futures):
                    try:
                        volume, mask, counters = future.result()
                    except Exception as error:
                        volume, mask, counters = _zero_study(np, _new_counters())
                        counters.update(study_build_failed=True, study_build_error_type=type(error).__name__,
                                        study_build_error=str(error)[:300])
                    jobs.put((uid, volume, mask, counters))
    finally:
        for _ in threads:
            jobs.put(None)
        jobs.join()
        for thread in threads:
            thread.join()
    # Coverage guard: a study without a result gets the reference default instead of being dropped.
    missing_results = [uid for uid in ids if uid not in results]
    for uid in missing_results:
        default = np.full(12, PROBABILITY_DEFAULT, dtype=np.float32)
        results[uid] = (default, default, default, _new_counters())
        model_fallbacks.append({"study_id": uid, "device": None, "error_type": "MissingResult",
                                "message": "no consumer result; reference default applied"})
    require(set(results) == set(ids) and len(results) == len(ids), "Incomplete study coverage")
    probabilities = np.stack([results[uid][0] for uid in ids])
    require(probabilities.shape == (len(ids), 12) and np.isfinite(probabilities).all(), "Output shape/finite gate")
    require(bool(((probabilities >= 0) & (probabilities <= 1)).all()), "Probability range gate")
    # Exact public ordinal rank transform; do not quietly substitute a different tie policy.
    ranks = probabilities.argsort(0).argsort(0).astype(np.float64) / max(1, len(ids) - 1)
    # Reference guard: any non-finite rank becomes the neutral 0.5.
    ranks[~np.isfinite(ranks)] = 0.5
    output = pd.DataFrame(ranks.astype(np.float32), columns=LABELS)
    output.insert(0, "StudyInstanceUID", ids)
    require(output.columns.tolist() == expected_columns and output["StudyInstanceUID"].tolist() == ids,
            "Final row/schema gate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    name = "submission.csv" if args.mode == "inference" else "canary_predictions.csv"
    destination = args.output_dir / name
    require(not destination.exists(), "Output already exists; preserve it and use a fresh directory")
    with destination.open("x") as stream:
        output.to_csv(stream, index=False)
    raw_path = args.output_dir / "probabilities.npz"
    with raw_path.open("xb") as stream:
        np.savez_compressed(stream, study_uids=np.asarray(ids, dtype=str), labels=np.asarray(LABELS),
                            values=probabilities, plain=np.stack([results[uid][1] for uid in ids]),
                            mirrored=np.stack([results[uid][2] for uid in ids]))
    receipt = {"mode": args.mode, "full_test_coverage": ids == all_ids, "studies": len(ids),
               "native_inference_completed": True, "competition_submitted": False,
               "official_score": None, "public_reference_score": 0.949,
               "public_ref": PUBLIC_REF, "public_version": PUBLIC_VERSION,
               "public_numeric_session_id": PUBLIC_SESSION_ID, "checkpoint_sha256": CHECKPOINT_SHA256,
               "dataset_ref": DATASET_REF, "dataset_version": DATASET_VERSION,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "versions": {"torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__},
               "gpu_count": gpu_count, "devices": device_names,
               "elapsed_s": time.monotonic() - started, "prep_workers": args.prep_workers,
               "fault_policy": FAULT_POLICY,
               "missing_slots": sum(results[uid][3]["missing_slots"] for uid in ids),
               "selected_slices": sum(results[uid][3]["selected_slices"] for uid in ids),
               "header_failures": sum(results[uid][3]["header_failures"] for uid in ids),
               "pixel_failures": sum(results[uid][3]["pixel_failures"] for uid in ids),
               "study_build_failures": sum(1 for uid in ids if results[uid][3]["study_build_failed"]),
               "studies_without_series_metadata": studies_without_series,
               "model_fallback_studies": len(model_fallbacks),
               "model_fallback_examples": model_fallbacks[:5],
               "consumer_error_count": len(consumer_errors), "consumer_error_examples": consumer_errors[:5],
               "prediction_file_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    receipt_path = args.output_dir / "native_inference_receipt.json"
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary", "inference"), default="canary")
    parser.add_argument("--max-studies", type=int, default=3)
    parser.add_argument("--prep-workers", type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    require(1 <= args.max_studies <= 16, "Canary study count must be 1..16")
    run(args)


if __name__ == "__main__":
    main()
