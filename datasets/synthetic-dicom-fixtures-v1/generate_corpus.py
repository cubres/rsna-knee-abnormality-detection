#!/usr/bin/env python3
"""Original synthetic encoding fixtures. No input files and no patient data. MIT."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, EnhancedMRImageStorage

VERSION = "1.0.0"
DEFAULT_SEED = 20261013
AXIAL = ["1", "0", "0", "0", "1", "0"]


def cases():
    """Each golden is a closed-form phantom value, never decoded from a DICOM."""
    result = []

    def add(case_id, raw, golden, **kw):
        raw = np.asarray(raw)
        case = dict(case_id=case_id, raw=raw, golden=golden,
                    mono="MONOCHROME2", bits_stored=16, slope=None,
                    intercept=None, center=None, width=None,
                    orientation=AXIAL, frames=1, error="")
        case.update(kw)
        result.append(case)

    # Row-major ramps expose transpose, sign, polarity and non-square errors.
    for r, c in [(1, 7), (3, 11), (8, 5), (16, 16)]:
        n = r * c
        index = np.arange(n).reshape(r, c)
        truth = index.astype(np.float64) / (n - 1)
        for signed in [False, True]:
            raw = (index - n // 2 if signed else index).astype("<i2" if signed else "<u2")
            for invert in [False, True]:
                add(f"ramp_{r}x{c}_{'i16' if signed else 'u16'}_m{1 if invert else 2}",
                    raw, 1 - truth if invert else truth,
                    **dict(mono="MONOCHROME1" if invert else "MONOCHROME2",
                           oracle="ramp_down" if invert else "ramp_up"))

    for signed in [False, True]:
        index = np.arange(45).reshape(5, 9)
        raw = (index - 22 if signed else index).astype("<i2" if signed else "<u2")
        settings = [(0.5, -20, -29 if signed else -18, "half", 1, 8),
                    (2, 5, -30 if signed else 14, "double", -13, 8),
                    (-1, 7, 25 if signed else 3, "negative", 13, 8),
                    (-0.25, 100, 103 if signed else 97.5, "negative_quarter", 20, 16)]
        for k, (slope, intercept, center, family, offset, denominator) in enumerate(settings):
            numerator = (index + offset if family == "half" else
                         4 * index + offset if family == "double" else
                         offset - 2 * index if family == "negative" else offset - index)
            truth = np.clip(numerator / denominator, 0, 1)
            add(f"rescale_{'i16' if signed else 'u16'}_{k}", raw, truth,
                **dict(slope=slope, intercept=intercept, center=center, width=5,
                       oracle="rescale_window_" + family))

    for invert in [False, True]:
        truth5 = np.array([[0, 1/8, 3/8, 5/8, 7/8, 1, 1]])
        add(f"window_width5_m{1 if invert else 2}", np.arange(7, dtype="<u2")[None],
            1 - truth5 if invert else truth5,
            **dict(mono="MONOCHROME1" if invert else "MONOCHROME2", center=3, width=5,
                   oracle="window5_inverse" if invert else "window5"))
        truth1 = np.array([[0, 0, 1, 1, 1]])
        add(f"window_width1_m{1 if invert else 2}", np.arange(-2, 3, dtype="<i2")[None],
            1 - truth1 if invert else truth1,
            **dict(mono="MONOCHROME1" if invert else "MONOCHROME2", center=0, width=1,
                   oracle="window1_inverse" if invert else "window1"))

    orientations = {"axial": AXIAL, "sagittal": ["0", "1", "0", "0", "0", "1"],
                    "coronal": ["1", "0", "0", "0", "0", "-1"],
                    "oblique": ["0.707106781187", "0.707106781187", "0", "0", "0", "1"]}
    raw = np.arange(45, dtype="<u2").reshape(5, 9)
    for name, direction in orientations.items():
        add("orientation_" + name, raw, np.arange(45).reshape(5, 9) / 44,
            **dict(orientation=direction, oracle="ramp_up"))

    for signed in [False, True]:
        for invert in [False, True]:
            raw = np.full((3, 5), -7 if signed else 7, dtype="<i2" if signed else "<u2")
            add(f"constant_{'i16' if signed else 'u16'}_m{1 if invert else 2}", raw,
                np.full((3, 5), 1 if invert else 0),
                **dict(mono="MONOCHROME1" if invert else "MONOCHROME2",
                       oracle="constant_one" if invert else "constant_zero"))
            raw12 = (np.arange(32).reshape(4, 8) - (16 if signed else 0)).astype(
                "<i2" if signed else "<u2")
            truth = np.arange(32).reshape(4, 8) / 31
            add(f"stored12_{'i16' if signed else 'u16'}_m{1 if invert else 2}", raw12,
                1 - truth if invert else truth,
                **dict(bits_stored=12, mono="MONOCHROME1" if invert else "MONOCHROME2",
                       oracle="ramp_down" if invert else "ramp_up"))

    signed_edge = np.array([[-32768, -16384, -1, 0, 1, 16384, 32767]], dtype="<i2")
    add("signed_endpoints", signed_edge,
        np.array([[0, 16384, 32767, 32768, 32769, 49152, 65535]]) / 65535,
        oracle="signed_endpoints")
    unsigned_edge = np.array([[0, 1, 32767, 32768, 40000, 65534, 65535]], dtype="<u2")
    add("unsigned_endpoints", unsigned_edge, unsigned_edge.astype(np.float64) / 65535,
        oracle="unsigned_endpoints")
    add("zero_slope", np.arange(-4, 5, dtype="<i2").reshape(3, 3), np.zeros((3, 3)),
        **dict(slope=0, intercept=123, oracle="constant_zero"))

    raw = np.arange(15, dtype="<u2").reshape(3, 5)
    for name, error in [("missing_pixels", "MISSING_PIXEL_DATA"),
                        ("empty_pixels", "EMPTY_PIXEL_DATA"),
                        ("truncated_pixels", "TRUNCATED_PIXEL_DATA")]:
        add(name, raw, None, **dict(error=error, oracle="refuse"))
    add("multiframe_two", np.stack([raw, raw + 20]), None,
        **dict(frames=2, error="MULTIFRAME_UNSUPPORTED", oracle="refuse"))
    add("rgb_unsupported", np.repeat(raw[..., None], 3, axis=2), None,
        **dict(mono="RGB", error="PHOTOMETRIC_UNSUPPORTED", oracle="refuse"))
    assert len(result) == 48 and sum(c["golden"] is not None for c in result) == 43
    return result


def uid(seed, name):
    digest = hashlib.sha256(f"synthetic-mr-edge-cases/{VERSION}/{seed}/{name}".encode()).digest()
    return "2.25." + str(int.from_bytes(digest[:16], "big"))


def encode(case, seed):
    meta = FileMetaDataset()
    sop_class = EnhancedMRImageStorage if case["frames"] > 1 else MRImageStorage
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = uid(seed, case["case_id"])
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = uid(seed, "implementation")
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = sop_class, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = uid(seed, "study"), uid(seed, case["case_id"] + "/series")
    ds.PatientName, ds.PatientID = "SYNTHETIC^PHANTOM", "SYNTHETIC"
    ds.PatientBirthDate, ds.PatientSex = "", ""
    ds.Modality, ds.Manufacturer = "MR", "SYNTHETIC FIXTURE GENERATOR"
    ds.StudyDate, ds.StudyTime = "20000101", "000000"
    ds.SeriesDescription = "SYNTHETIC " + case["case_id"]
    ds.ImageType, ds.InstanceNumber = ["DERIVED", "SECONDARY"], 1
    ds.ImageOrientationPatient = case["orientation"]
    ds.ImagePositionPatient, ds.PixelSpacing = [0, 0, 0], [1, 1]
    ds.SliceThickness = 1
    raw = case["raw"]
    rgb = case["mono"] == "RGB"
    ds.Rows, ds.Columns = (raw.shape[:2] if rgb else raw.shape[-2:])
    ds.SamplesPerPixel, ds.PhotometricInterpretation = (3 if rgb else 1), case["mono"]
    if rgb:
        ds.PlanarConfiguration = 0
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, case["bits_stored"], case["bits_stored"] - 1
    ds.PixelRepresentation = int(raw.dtype.kind == "i")
    if case["frames"] > 1:
        ds.NumberOfFrames = case["frames"]
    if case["slope"] is not None:
        ds.RescaleSlope, ds.RescaleIntercept = str(case["slope"]), str(case["intercept"])
    if case["width"] is not None:
        ds.WindowCenter, ds.WindowWidth, ds.VOILUTFunction = str(case["center"]), str(case["width"]), "LINEAR"
    payload = raw.astype("<i2" if ds.PixelRepresentation else "<u2").tobytes()
    if case["case_id"] != "missing_pixels":
        ds.PixelData = (b"" if case["case_id"] == "empty_pixels" else
                        payload[:-2] if case["case_id"] == "truncated_pixels" else payload)
    stream = io.BytesIO()
    ds.save_as(stream, enforce_file_format=True)
    return stream.getvalue(), ds


def write_new(path, payload):
    with path.open("xb") as handle:
        handle.write(payload)


def generate(destination, seed):
    destination = Path(destination)
    destination.mkdir(parents=False, exist_ok=False)
    (destination / "dicom").mkdir()
    (destination / "golden").mkdir()
    rows = []
    for case in cases():
        blob, ds = encode(case, seed)
        dcm_name = "dicom/" + case["case_id"] + ".dcm"
        write_new(destination / dcm_name, blob)
        golden_name, golden_hash = "", ""
        if case["golden"] is not None:
            stream = io.BytesIO()
            np.save(stream, np.asarray(case["golden"], dtype="<f4"), allow_pickle=False)
            golden_blob = stream.getvalue()
            golden_name = "golden/" + case["case_id"] + ".npy"
            write_new(destination / golden_name, golden_blob)
            golden_hash = hashlib.sha256(golden_blob).hexdigest()
        rows.append(dict(case_id=case["case_id"], dicom_path=dcm_name,
            dicom_sha256=hashlib.sha256(blob).hexdigest(), dicom_bytes=len(blob),
            rows=ds.Rows, columns=ds.Columns, frames=case["frames"],
            samples_per_pixel=ds.SamplesPerPixel, bits_allocated=16,
            bits_stored=ds.BitsStored, pixel_representation=ds.PixelRepresentation,
            photometric_interpretation=ds.PhotometricInterpretation,
            rescale_slope=case["slope"] if case["slope"] is not None else 1,
            rescale_intercept=case["intercept"] if case["intercept"] is not None else 0,
            window_center=case["center"] if case["center"] is not None else "",
            window_width=case["width"] if case["width"] is not None else "",
            orientation=";".join(case["orientation"]),
            expected_contract="refuse" if case["error"] else "float32_2d_0_to_1_no_reorientation",
            expected_error=case["error"], oracle=case["oracle"],
            golden_path=golden_name, golden_sha256=golden_hash))
    with (destination / "manifest.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    note = dict(version=VERSION, seed=seed, dicom_count=48, golden_count=43,
                generator="generate_corpus.py", numpy=np.__version__, pydicom=pydicom.__version__,
                clinical_data=False, complete_clinical_iod_conformance_claim=False)
    write_new(destination / "generation.json", (json.dumps(note, indent=2) + "\n").encode())
    print(json.dumps(note, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="A new directory; its parent must already exist")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    generate(args.out, args.seed)
