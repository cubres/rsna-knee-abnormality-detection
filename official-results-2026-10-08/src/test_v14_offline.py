"""Offline (CPU) proof for the V14 fault-tolerant reader. Synthetic DICOM only; no Kaggle, no network.

Compares, per synthetic study: the V13 reader (strict), the V14 reader (candidate), the reference
build_study_v2 (verbatim in the licensed fast-read module) and the reference study_lut builder used by
the comparator. Then runs the V13 and V14 run() end to end on CPU with patched CUDA and a fake model.
Run with: <venv>/bin/python -I tests/test_v14_offline.py --work <new scratch dir> --results <json path>
"""
import argparse, contextlib, hashlib, importlib.util, json, sys, time, traceback, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
KNEE = Path(__file__).resolve().parent.parent
RESEARCH = KNEE.parent
OLD_READER = KNEE / "work" / "v13_payload_decoded" / "public_mirror_reader.py"
PKG = KNEE / "candidate_v14d" / "sources_for_review"
NEW_READER = PKG / "public_mirror_reader.py"
FASTREAD = PKG / "licensed_public_v2_fastread_reference.py"
CANARY = PKG / "native_mirror_canary_worker.py"
SHA = {
    "old": "4c0f5e4ef362941927e79ca3ab7e98a2c17a46d5a9230928b9057ce3e4a7b387",
    "fastread": "5a59b6f82e846bcdfd452aaee6ff518cdd9d0b04aab03691b45c3f70a5198cb0",
    "canary": "a4670b18c0706bff00b86473f9a26017097154502f33a626be95d8ded238fe97",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------- synthetic DICOM cohort ----------------
SLOT_DEF = [("Sagittal", 1, "s0"), ("Sagittal", 0, "s1"), ("Coronal", 1, "s2"), ("Coronal", 0, "s3"), ("Axial", -1, "s4")]


def write_dcm(path, pixels, *, photometric="MONOCHROME2", geometry=True, position=0.0, instance=1,
              float_pixels=False, frames=1, mono1=False):
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    meta.MediaStorageSOPInstanceUID = f"1.2.3.4.{instance}"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.InstanceNumber = "MR", instance
    ds.PhotometricInterpretation = "MONOCHROME1" if mono1 else photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = int(pixels.shape[-2]), int(pixels.shape[-1])
    if frames > 1:
        ds.NumberOfFrames = frames
    if geometry:
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float(position)]
        ds.PixelSpacing = [0.5, 0.5]
    if float_pixels:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 32, 32, 31, 0
        ds.add_new(0x7FE00008, "OF", pixels.astype("<f4").tobytes())
    else:
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = pixels.astype("<u2").tobytes()
    ds.save_as(str(path))


def healthy_pixels(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4000, size=(100, 120)).astype(np.uint16)


def make_series(series_dir, tag, n, *, geometry=True, mono1=False, faults=None, all_garbage=False, no_dcm=False):
    series_dir.mkdir(parents=True, exist_ok=True)
    if no_dcm:
        (series_dir / "README.txt").write_text("no dicom here")
        return
    for i in range(n):
        path = series_dir / f"{i:03d}.dcm"
        fault = (faults or {}).get(i)
        seed = abs(hash((tag, i))) % (2**32) if False else (sum(map(ord, tag)) * 1000 + i)
        if all_garbage or fault == "garbage":
            path.write_bytes(b"this is not a dicom file " * 20)
            continue
        if fault == "truncate_header":
            write_dcm(path, healthy_pixels(seed), geometry=geometry, position=i + 1.0, instance=i + 1)
            path.write_bytes(path.read_bytes()[:100])
            continue
        if fault == "truncate_pixels":
            write_dcm(path, healthy_pixels(seed), geometry=geometry, position=i + 1.0, instance=i + 1)
            raw = path.read_bytes()
            path.write_bytes(raw[: int(len(raw) * 0.6)])
            continue
        if fault == "multiframe":
            frames = np.stack([healthy_pixels(seed), healthy_pixels(seed + 1)])
            write_dcm(path, frames, geometry=geometry, position=i + 1.0, instance=i + 1, frames=2)
            continue
        if fault == "nan_float":
            arr = healthy_pixels(seed).astype(np.float32)
            arr[10:20, 10:20] = np.nan
            write_dcm(path, arr, geometry=geometry, position=i + 1.0, instance=i + 1, float_pixels=True)
            continue
        write_dcm(path, healthy_pixels(seed), geometry=geometry, position=i + 1.0, instance=i + 1, mono1=mono1)


CASES = [
    # name, study_id, make_study kwargs
    ("healthy", "1.2.10.1", {}),
    ("healthy_no_geometry", "1.2.10.2", {"geometry": False}),
    ("monochrome1", "1.2.10.3", {"mono1": True}),
    ("truncated_pixels", "1.2.10.4", {"file_faults": {("s0", 0): "truncate_pixels"}}),
    ("truncated_header", "1.2.10.5", {"file_faults": {("s0", 0): "truncate_header"}}),
    ("garbage_header_and_pixels", "1.2.10.6", {"file_faults": {("s0", 0): "garbage"}}),
    ("multiframe_non2d", "1.2.10.7", {"file_faults": {("s2", 0): "multiframe"}}),
    ("nan_pixels", "1.2.10.8", {"file_faults": {("s2", 0): "nan_float"}}),
    ("series_without_dcm", "1.2.10.9", {"no_dcm": ("s4",)}),
    ("no_test_series_rows", "1.2.10.10", {"drop_rows": True}),
    ("all_series_garbage", "1.2.10.11", {"all_garbage": True}),
    ("only_two_planes_present", "1.2.10.12", {"keep_slots": ("s0", "s1")}),
]


def make_study(root, sid, *, geometry=True, mono1=False, file_faults=None, no_dcm=(), drop_rows=False,
               all_garbage=False, keep_slots=None, n=30):
    rows = []
    for plane, fluid, tag in SLOT_DEF:
        uid = f"{sid}.{tag}"
        series_dir = root / "test_series" / sid / uid
        faults = {idx: kind for (t, idx), kind in (file_faults or {}).items() if t == tag}
        make_series(series_dir, uid, n, geometry=geometry, mono1=mono1, faults=faults,
                    all_garbage=all_garbage, no_dcm=(tag in no_dcm))
        if keep_slots is not None and tag not in keep_slots:
            continue
        if drop_rows:
            continue
        rows.append(dict(StudyInstanceUID=sid, SeriesInstanceUID=uid, Anatomical_Plane=plane,
                         Fluid_Sensitive=fluid if fluid in (0, 1) else 0))
    return rows


def build_cohort(root, cases):
    """Create test.csv, test_series.csv, sample_submission.csv and the DICOM tree under root."""
    import pandas as pd
    root.mkdir(parents=True, exist_ok=True)
    all_rows, ids = [], []
    for name, sid, kwargs in cases:
        kw = dict(kwargs)
        drop = kw.pop("drop_rows", False)
        no_dcm = kw.pop("no_dcm", ())
        keep = kw.pop("keep_slots", None)
        file_faults = kw.pop("file_faults", None)
        rows = make_study(root, sid, file_faults=file_faults, no_dcm=no_dcm, drop_rows=drop,
                          keep_slots=keep, **kw)
        all_rows.extend(rows)
        ids.append(sid)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(root / "sample_submission.csv", index=False)
    cols = ["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]
    pd.DataFrame(all_rows, columns=cols).to_csv(root / "test_series.csv", index=False)
    return ids


# The sample_submission must carry the 12 label columns for run(); written separately below.
LABELS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture")


def write_sample(root, ids):
    import pandas as pd
    df = pd.DataFrame({"StudyInstanceUID": ids})
    for label in LABELS:
        df[label] = 0.0
    df.to_csv(root / "sample_submission.csv", index=False)


def arrays_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=False)
    results = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}, "cases": {}}
    checks = results["checks"]

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f" :: {detail}"), flush=True)

    checks_ok_hashes = {
        "old_reader_sha_ok": sha256(OLD_READER) == SHA["old"],
        "fastread_sha_ok": sha256(FASTREAD) == SHA["fastread"],
        "canary_sha_ok": sha256(CANARY) == SHA["canary"],
    }
    results["pins"] = dict(checks_ok_hashes, new_reader_sha256=sha256(NEW_READER))
    check("pinned_sources_match", all(checks_ok_hashes.values()), checks_ok_hashes)

    old = load(OLD_READER, "v13_reader_under_test")
    new = load(NEW_READER, "v14_reader_under_test")
    fast = load(FASTREAD, "licensed_fastread_for_tests")
    deps = new.dependencies()
    cv2, np_, pd, pydicom, _, timm, torch = deps

    # Reference builders, exactly as the public notebook / comparator call them.
    ref_reader = fast.make_reader_v2(384, 16)
    ref_lut = fast.make_builder("study_lut", 384, list(new.SLOTS), hdr_threads=32, px_threads=16)

    cohort_root = work / "cohort"
    ids = build_cohort(cohort_root, CASES)
    write_sample(cohort_root, ids)
    import pandas as pd_mod
    series = pd_mod.read_csv(cohort_root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    rows_by_sid = {k: v.to_dict("records") for k, v in series.groupby("StudyInstanceUID", sort=False)}
    series_root = cohort_root / "test_series"

    zero_vol = np_.zeros((96, 384, 384), np.uint8)
    zero_mask = np_.zeros(96, np.uint8)
    summary_all_new_no_raise = True
    summary_shapes_ok = True
    unit_mismatch = []
    for name, sid, _ in CASES:
        rows = rows_by_sid.get(sid, [])
        case = {}
        # V13 strict reader
        try:
            ov, om, oc = old.build_volume(sid, rows, series_root, deps)
            case["v13"] = dict(raised=False, volume_sha=hashlib.sha256(ov.tobytes()).hexdigest()[:16],
                               mask_sum=int(om.sum()), counters=oc)
        except Exception as error:
            case["v13"] = dict(raised=True, error=type(error).__name__, message=str(error)[:160])
        # V14 tolerant reader (must never raise)
        try:
            nv, nm, nc = new.build_volume(sid, rows, series_root, deps)
            shapes_ok = nv.shape == (96, 384, 384) and nv.dtype == np_.uint8 and nm.shape == (96,) and nm.dtype == np_.uint8
            summary_shapes_ok &= shapes_ok
            case["v14"] = dict(raised=False, shapes_ok=shapes_ok, volume_sha=hashlib.sha256(nv.tobytes()).hexdigest()[:16],
                               mask_sum=int(nm.sum()), counters=nc)
        except Exception as error:
            summary_all_new_no_raise = False
            case["v14"] = dict(raised=True, error=type(error).__name__, message=str(error)[:160])
            nv, nm = zero_vol, zero_mask
        # Reference build_study_v2 with the reference's own whole-study fallback (_pre_one): zeros on error.
        try:
            rv, rm = fast.build_study_v2(sid, {sid: rows}, str(series_root), ref_reader, list(new.SLOTS), 384)
            case["ref_v2"] = dict(raised=False)
        except Exception as error:
            rv, rm = zero_vol, zero_mask
            case["ref_v2"] = dict(raised=True, error=type(error).__name__, message=str(error)[:160])
        # Reference study_lut (the public V2 fast path and the comparator's builder), same fallback.
        try:
            lv, lm = ref_lut(sid, {sid: rows}, str(series_root))
            case["ref_lut"] = dict(raised=False)
        except Exception as error:
            lv, lm = zero_vol, zero_mask
            case["ref_lut"] = dict(raised=True, error=type(error).__name__, message=str(error)[:160])
        case["v14_equals_ref_v2"] = arrays_equal(nv, rv) and arrays_equal(nm, rm)
        case["v14_equals_ref_lut"] = arrays_equal(nv, lv) and arrays_equal(nm, lm)
        case["v13_equals_v14"] = (not case["v13"]["raised"]) and arrays_equal(
            nv, old.build_volume(sid, rows, series_root, deps)[0]) if not case["v13"]["raised"] else None
        case["ref_v2_equals_ref_lut"] = arrays_equal(rv, lv) and arrays_equal(rm, lm)
        results["cases"][name] = case
        if not (case["v14_equals_ref_v2"] and case["v14_equals_ref_lut"]):
            unit_mismatch.append(name)
        print(f"  case {name:28s} v13={'RAISE:'+case['v13'].get('error','') if case['v13']['raised'] else 'ok'} "
              f"v14={'RAISE' if case['v14']['raised'] else 'ok'} ref_v2={'raise' if case['ref_v2']['raised'] else 'ok'} "
              f"ref_lut={'raise' if case['ref_lut']['raised'] else 'ok'} "
              f"v14==ref_v2:{case['v14_equals_ref_v2']} v14==ref_lut:{case['v14_equals_ref_lut']}", flush=True)

    check("v14_build_volume_never_raises", summary_all_new_no_raise)
    check("v14_shapes_and_dtypes_fallback", summary_shapes_ok)
    healthy_names = ["healthy", "healthy_no_geometry", "monochrome1"]
    check("healthy_v13_v14_ref_identical",
          all(results["cases"][n]["v13"]["raised"] is False and results["cases"][n]["v13_equals_v14"] and
              results["cases"][n]["v14_equals_ref_v2"] and results["cases"][n]["v14_equals_ref_lut"]
              for n in healthy_names),
          {n: [results["cases"][n]["v13_equals_v14"], results["cases"][n]["v14_equals_ref_v2"],
               results["cases"][n]["v14_equals_ref_lut"]] for n in healthy_names})
    check("strict_v13_raises_on_faults",
          all(results["cases"][n]["v13"]["raised"] for n in ["truncated_header", "garbage_header_and_pixels",
                                                              "multiframe_non2d", "all_series_garbage",
                                                              "no_test_series_rows"]),
          {n: results["cases"][n]["v13"].get("error") for n in ["truncated_header", "garbage_header_and_pixels",
                                                                  "multiframe_non2d", "all_series_garbage",
                                                                  "no_test_series_rows", "truncated_pixels", "nan_pixels"]})
    check("v14_matches_reference_v2_on_all_cases", not [n for n in results["cases"] if not results["cases"][n]["v14_equals_ref_v2"]],
          {"mismatch_vs_ref_v2": [n for n in results["cases"] if not results["cases"][n]["v14_equals_ref_v2"]]})
    check("v14_matches_reference_study_lut_on_all_cases", not [n for n in results["cases"] if not results["cases"][n]["v14_equals_ref_lut"]],
          {"mismatch_vs_ref_lut": [n for n in results["cases"] if not results["cases"][n]["v14_equals_ref_lut"]]})

    # Pure function parity: prepare_windows on the synthetic healthy study, old vs new, CPU.
    rows = rows_by_sid["1.2.10.1"]
    hv, hm, _ = new.build_volume("1.2.10.1", rows, series_root, deps)
    ov, om, _ = old.build_volume("1.2.10.1", rows, series_root, deps)
    pw_equal = []
    for mirrored in (False, True):
        a = old.prepare_windows(ov, om, "cpu", mirrored, np_, torch).numpy()
        b = new.prepare_windows(hv, hm, "cpu", mirrored, np_, torch).numpy()
        pw_equal.append(bool(np_.array_equal(a, b)))
    check("prepare_windows_old_new_identical", all(pw_equal), pw_equal)

    # End-to-end run() on CPU with patched CUDA and a fake model.
    results["run"] = {}
    run_out = run_level_tests(old, new, cohort_root, work, ids, check, results)

    # Comparator (native_mirror_canary_worker.compare_native_volumes) with the V14 reader.
    comp = load(CANARY, "canary_for_tests")
    results["comparator"] = {}
    for label, order in [("healthy_first_three", ["1.2.10.1", "1.2.10.2", "1.2.10.3"]),
                         ("faulty_first_three", ["1.2.10.4", "1.2.10.5", "1.2.10.6"])]:
        croot = work / f"comparator_{label}"
        croot.mkdir(parents=True, exist_ok=True)
        (croot / "test_series").symlink_to(series_root.resolve())
        pd_mod.DataFrame({"StudyInstanceUID": order}).to_csv(croot / "test.csv", index=False)
        pd_mod.read_csv  # keep import used
        series.to_csv(croot / "test_series.csv", index=False)
        outdir = work / f"comparator_out_{label}"
        outdir.mkdir()
        sys.modules.pop("_licensed_public_v2_study_lut_5a59b6f8", None)  # the comparator registers this name once per process
        try:
            comp.compare_native_volumes(new, PKG, outdir, croot, deps)
            results["comparator"][label] = dict(status="PASS")
        except Exception as error:
            results["comparator"][label] = dict(status="RAISED", error=type(error).__name__, message=str(error)[:300])
        rep = outdir / "native_volume_mask_parity.json"
        if rep.exists():
            results["comparator"][label]["report_status"] = json.loads(rep.read_text()).get("status")
    check("comparator_pass_on_healthy_first_three", results["comparator"]["healthy_first_three"]["status"] == "PASS",
          results["comparator"]["healthy_first_three"])
    results["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2, sort_keys=True, default=str) + "\n")
    failed = [k for k, v in checks.items() if not v["ok"]]
    print("RESULT", "ALL_CHECKS_PASS" if not failed else f"FAILED: {failed}", flush=True)


class FakeModel(object):
    pass


def run_level_tests(old, new, cohort_root, work, ids, check, results):
    import torch
    import torch.nn as nn

    class Fake(nn.Module):
        def forward(self, windows):
            x = windows.float().mean()
            return torch.stack([x * (i + 1) for i in range(12)]).view(1, 12)

    def make_fake(*a, **k):
        return Fake().eval()

    def run_one(module, label, gpu_count, root, inject=(), device_name="Tesla T4"):
        out = work / f"run_{label}"
        saved = {
            "is_available": torch.cuda.is_available, "device_count": torch.cuda.device_count,
            "get_device_name": torch.cuda.get_device_name, "empty_cache": torch.cuda.empty_cache,
            "autocast": torch.autocast,
        }
        saved_mod = {k: getattr(module, k) for k in ("find_attached_checkpoint", "make_model", "predict_pair", "INPUT_RES")}
        counter = {"n": 0}
        orig_predict = module.predict_pair

        def predict_wrapper(model, volume, mask, device, np, torch_):
            k = counter["n"]
            counter["n"] += 1
            if k in inject:
                raise RuntimeError(f"injected model failure at call {k}")
            return orig_predict(model, volume, mask, "cpu", np, torch_)

        try:
            torch.cuda.is_available = lambda: True
            torch.cuda.device_count = lambda: gpu_count
            torch.cuda.get_device_name = lambda index=0: device_name
            torch.cuda.empty_cache = lambda: None
            torch.autocast = lambda *a, **k: contextlib.nullcontext()
            module.find_attached_checkpoint = lambda root: Path("/nonexistent/checkpoint.pt")
            module.make_model = make_fake
            module.predict_pair = predict_wrapper
            module.INPUT_RES = 64
            args = argparse.Namespace(input_root=Path("/nonexistent"), competition_root=root,
                                      output_dir=out, mode="inference", max_studies=3, prep_workers=2)
            module.run(args)
            sub = (out / "submission.csv").read_bytes()
            receipt = json.loads((out / "native_inference_receipt.json").read_text())
            return dict(ok=True, submission_sha=hashlib.sha256(sub).hexdigest(), receipt=receipt,
                        submission_bytes=sub)
        except Exception as error:
            return dict(ok=False, error=type(error).__name__, message=str(error)[:300],
                        traceback_tail=traceback.format_exc()[-600:])
        finally:
            torch.cuda.is_available = saved["is_available"]
            torch.cuda.device_count = saved["device_count"]
            torch.cuda.get_device_name = saved["get_device_name"]
            torch.cuda.empty_cache = saved["empty_cache"]
            torch.autocast = saved["autocast"]
            for k, v in saved_mod.items():
                setattr(module, k, v)

    # 1) Old V13 run and new V14 run on the cohort with only fault-free-or-geometry-fallback studies (happy path).
    healthy_root = work / "cohort_happy"
    happy_ids = ids[:3]
    build_root = healthy_root
    build_root.mkdir(parents=True, exist_ok=True)
    (build_root / "test_series").symlink_to((cohort_root / "test_series").resolve())
    import pandas as pd
    sub = pd.read_csv(cohort_root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    sub[sub["StudyInstanceUID"].isin(happy_ids)].to_csv(build_root / "test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": happy_ids}).to_csv(build_root / "test.csv", index=False)
    write_sample_for(build_root, happy_ids)
    old_happy = run_one(old, "v13_happy_gpu2", 2, healthy_root)
    new_happy = run_one(new, "v14_happy_gpu2", 2, healthy_root)
    results["run"]["v13_happy_gpu2"] = {k: v for k, v in old_happy.items() if k != "submission_bytes"}
    results["run"]["v14_happy_gpu2"] = {k: v for k, v in new_happy.items() if k != "submission_bytes"}
    same = old_happy.get("ok") and new_happy.get("ok") and old_happy["submission_bytes"] == new_happy["submission_bytes"]
    check("happy_path_run_submission_byte_identical_v13_v14", bool(same),
          {"v13": old_happy.get("submission_sha") or old_happy.get("error"), "v14": new_happy.get("submission_sha") or new_happy.get("error")})

    # 2) V13 on the full faulty cohort: expected to die (it first meets the study with no series rows).
    old_faulty = run_one(old, "v13_faulty_gpu2", 2, cohort_root)
    results["run"]["v13_faulty_gpu2"] = {k: v for k, v in old_faulty.items() if k not in ("submission_bytes", "traceback_tail")}
    check("v13_run_dies_on_faulty_cohort", not old_faulty.get("ok"), results["run"]["v13_faulty_gpu2"].get("error"))

    # 2b) V13 on the same cohort WITHOUT the empty-series study: the build-level exception itself kills it.
    noempty = work / "cohort_noempty"
    noempty.mkdir(parents=True, exist_ok=True)
    (noempty / "test_series").symlink_to((cohort_root / "test_series").resolve())
    keep_ids = [i for i in ids if i != "1.2.10.10"]
    pd.DataFrame({"StudyInstanceUID": keep_ids}).to_csv(noempty / "test.csv", index=False)
    sub[sub["StudyInstanceUID"].isin(keep_ids)].to_csv(noempty / "test_series.csv", index=False)
    write_sample_for(noempty, keep_ids)
    old_noempty = run_one(old, "v13_faulty_without_empty_study_gpu2", 2, noempty)
    results["run"]["v13_faulty_without_empty_study_gpu2"] = {k: v for k, v in old_noempty.items() if k not in ("submission_bytes", "traceback_tail")}
    check("v13_run_dies_on_build_exception_without_empty_study", not old_noempty.get("ok"),
          {"error": old_noempty.get("error"), "message": old_noempty.get("message")})
    # 3) V14 on the full faulty cohort, 1 GPU, one injected model failure on the 6th study (F3 garbage, index 5).
    inject_index = ids.index("1.2.10.6")
    new_faulty_1 = run_one(new, "v14_faulty_gpu1_inject", 1, cohort_root, inject=(inject_index,))
    results["run"]["v14_faulty_gpu1_inject"] = {k: v for k, v in new_faulty_1.items() if k not in ("submission_bytes", "traceback_tail")}
    ok_1 = bool(new_faulty_1.get("ok"))
    check("v14_run_completes_on_faulty_cohort_with_injection", ok_1, new_faulty_1.get("error"))
    if ok_1:
        rc = new_faulty_1["receipt"]
        check("v14_receipt_coverage_and_fallbacks",
              rc["studies"] == len(ids) and rc["full_test_coverage"] is True and rc["model_fallback_studies"] == 1
              and rc["studies_without_series_metadata"] == 1 and rc["study_build_failures"] >= 1,
              {k: rc.get(k) for k in ("studies", "full_test_coverage", "model_fallback_studies", "study_build_failures",
                                      "header_failures", "pixel_failures", "missing_slots", "gpu_count", "devices")})
        sub_df = pd.read_csv(work / "run_v14_faulty_gpu1_inject" / "submission.csv", dtype={"StudyInstanceUID": str})
        vals = sub_df[list(LABELS)].to_numpy(dtype=float)
        check("v14_faulty_submission_rows_and_range",
              list(sub_df["StudyInstanceUID"]) == ids and np_finite_in_range(vals) and len(sub_df) == len(ids),
              {"rows": len(sub_df)})
    # 4) Determinism across device counts on the faulty cohort without injection.
    new_faulty_1b = run_one(new, "v14_faulty_gpu1", 1, cohort_root)
    new_faulty_2 = run_one(new, "v14_faulty_gpu2", 2, cohort_root)
    check("v14_faulty_run_ok_gpu1_and_gpu2", bool(new_faulty_1b.get("ok") and new_faulty_2.get("ok")),
          {"gpu1": new_faulty_1b.get("error"), "gpu2": new_faulty_2.get("error")})
    if new_faulty_1b.get("ok") and new_faulty_2.get("ok"):
        check("v14_faulty_submission_identical_gpu1_vs_gpu2",
              new_faulty_1b["submission_bytes"] == new_faulty_2["submission_bytes"],
              {"gpu1": new_faulty_1b["submission_sha"], "gpu2": new_faulty_2["submission_sha"]})
        results["run"]["v14_faulty_gpu1"] = {k: v for k, v in new_faulty_1b.items() if k not in ("submission_bytes", "traceback_tail")}
        results["run"]["v14_faulty_gpu2"] = {k: v for k, v in new_faulty_2.items() if k not in ("submission_bytes", "traceback_tail")}
    return dict(happy=(old_happy, new_happy))


def write_sample_for(root, ids):
    import pandas as pd
    df = pd.DataFrame({"StudyInstanceUID": ids})
    for label in LABELS:
        df[label] = 0.0
    df.to_csv(root / "sample_submission.csv", index=False)


def np_finite_in_range(vals):
    return bool(np.isfinite(vals).all() and ((vals >= 0) & (vals <= 1)).all())


if __name__ == "__main__":
    main()
