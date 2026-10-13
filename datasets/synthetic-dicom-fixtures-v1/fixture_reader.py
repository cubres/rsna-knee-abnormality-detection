"""Small explicit pixel contract for the synthetic corpus; original MIT code."""
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian


class FixtureRefusal(ValueError):
    """A stable refusal code, independent of third-party exception wording."""


def read_fixture(path):
    ds = pydicom.dcmread(path)
    if ds.file_meta.TransferSyntaxUID != ExplicitVRLittleEndian:
        raise FixtureRefusal("TRANSFER_SYNTAX_UNSUPPORTED")
    if str(ds.PhotometricInterpretation) not in ("MONOCHROME1", "MONOCHROME2"):
        raise FixtureRefusal("PHOTOMETRIC_UNSUPPORTED")
    if int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise FixtureRefusal("MULTIFRAME_UNSUPPORTED")
    r, c = int(ds.Rows), int(ds.Columns)
    if (r < 1 or c < 1 or ds.SamplesPerPixel != 1 or ds.BitsAllocated != 16 or
            ds.PixelRepresentation not in (0, 1) or not 1 <= ds.BitsStored <= 16 or
            ds.HighBit != ds.BitsStored - 1):
        raise FixtureRefusal("PIXEL_ENCODING_UNSUPPORTED")
    if "PixelData" not in ds:
        raise FixtureRefusal("MISSING_PIXEL_DATA")
    if not ds.PixelData:
        raise FixtureRefusal("EMPTY_PIXEL_DATA")
    if len(ds.PixelData) < r * c * 2:
        raise FixtureRefusal("TRUNCATED_PIXEL_DATA")
    if len(ds.PixelData) != r * c * 2:
        raise FixtureRefusal("PIXEL_LENGTH_MISMATCH")
    direction = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    if (direction.shape != (6,) or not np.isfinite(direction).all() or
            not np.allclose([np.linalg.norm(direction[:3]), np.linalg.norm(direction[3:]),
                             np.dot(direction[:3], direction[3:])], [1, 1, 0], atol=1e-6)):
        raise FixtureRefusal("ORIENTATION_INVALID")
    stored = ds.pixel_array
    if stored.shape != (r, c):
        raise FixtureRefusal("SHAPE_MISMATCH")
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    image = stored.astype(np.float64) * slope + intercept
    if not np.isfinite(image).all():
        raise FixtureRefusal("NONFINITE_INTENSITY")
    if "WindowWidth" in ds:
        center, width = float(ds.WindowCenter), float(ds.WindowWidth)
        if not np.isfinite([center, width]).all() or width < 1:
            raise FixtureRefusal("WINDOW_INVALID")
        image = (image > center - 0.5).astype(np.float64) if width == 1 else np.clip(
            (image - (center - 0.5)) / (width - 1) + 0.5, 0, 1)
    else:
        low, high = float(image.min()), float(image.max())
        image = (image - low) / (high - low) if high > low else np.zeros_like(image)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        image = 1 - image
    # Direction is returned as metadata. Decoded row and column axes stay intact.
    return image.astype(np.float32), dict(rows=r, columns=c, orientation=direction.tolist())
