# Original CPU companion: verify the attached synthetic DICOM contract.
from pathlib import Path as _FixturePath
import hashlib as _fixture_hashlib
import importlib.util as _fixture_importutil
import runpy as _fixture_runpy
import sys as _fixture_sys

if not __debug__:
    raise RuntimeError("Golden verification requires assertions; do not use -O")
_fixture_override = globals().get("_KNEE_FIXTURE_DATASET_ROOT")
if _fixture_override is not None:
    _fixture_root = _FixturePath(_fixture_override).resolve()
else:
    _fixture_mounts = [p for p in (
        _FixturePath("/kaggle/input/rsna-knee-synthetic-dicom-fixtures"),
        _FixturePath("/kaggle/input/datasets/prvsiyan/rsna-knee-synthetic-dicom-fixtures"),
    ) if p.is_dir()]
    if len(_fixture_mounts) != 1:
        raise RuntimeError("Attach prvsiyan/rsna-knee-synthetic-dicom-fixtures; expected exactly one mount")
    _fixture_root = _fixture_mounts[0].resolve()
for _fixture_package in ("numpy", "pydicom"):
    if _fixture_importutil.find_spec(_fixture_package) is None:
        _fixture_input = _FixturePath("/kaggle/input")
        _fixture_wheels = sorted(str(p) for p in _fixture_input.rglob(_fixture_package + "-*.whl"))[:5] if _fixture_input.is_dir() else []
        raise RuntimeError(f"{_fixture_package} is unavailable before the canary; offline wheel candidates: {_fixture_wheels}. Install an attached wheel explicitly; this demo never downloads packages.")
_fixture_pins = {
    "fixture_reader.py": "db8e2a303b0fed5783d69825051d5dae94af5dadfc55958f44e57ae8931f5581",
    "verify_corpus.py": "5bce1aedc52c2f6fed2879f38a5f572db77a4ccdec457d16268af8f970db0eb6",
    "corpus/manifest.csv": "d8a9a9424667947d06b2b93ced7b1cfeafd24644444dad2542a56a7e994cced3",
}
for _fixture_name, _fixture_expected in _fixture_pins.items():
    _fixture_path = _fixture_root / _fixture_name
    if _fixture_hashlib.sha256(_fixture_path.read_bytes()).hexdigest() != _fixture_expected:
        raise RuntimeError("Fixture source/manifest identity differs: " + _fixture_name)
_fixture_cached = _fixture_sys.modules.get("fixture_reader")
if _fixture_cached is not None and _FixturePath(_fixture_cached.__file__).resolve() != (_fixture_root / "fixture_reader.py").resolve():
    raise RuntimeError("A different fixture_reader is already imported")
_fixture_previous_path = _fixture_sys.path[:]
_fixture_previous_bytecode = _fixture_sys.dont_write_bytecode
try:
    _fixture_sys.dont_write_bytecode = True
    _fixture_sys.path.insert(0, str(_fixture_root))
    _fixture_namespace = _fixture_runpy.run_path(str(_fixture_root / "verify_corpus.py"), run_name="_knee_fixture_demo")
    _fixture_namespace["verify"](_fixture_root / "corpus")
    _fixture_image, _fixture_geometry = _fixture_namespace["read_fixture"](_fixture_root / "corpus/dicom/ramp_3x11_i16_m2.dcm")
    print("Original non-square phantom: shape", _fixture_image.shape,
          "first/last intensities", float(_fixture_image[0, 0]), float(_fixture_image[-1, -1]))
    print("Orientation is returned as metadata; no transpose, reorientation or MRI prediction is performed.")
finally:
    _fixture_sys.path[:] = _fixture_previous_path
    _fixture_sys.dont_write_bytecode = _fixture_previous_bytecode
