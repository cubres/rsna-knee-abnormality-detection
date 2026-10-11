"""Verify exact source bytes without importing code or changing any file."""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '6ba31b6bd4fc397abb63f96b82e9251a91540f79cde0dc33c4ce46a20e9a09c4'

def verify(directory):
    root = Path(directory).resolve()
    manifest = root / "SOURCE_MANIFEST.json"
    if manifest.is_symlink() or hashlib.sha256(manifest.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError("Source manifest differs from verified public version 1")
    data = json.loads(manifest.read_text())
    for name, expected in data["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid file name")
        path = root / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError("File size or type differs: " + name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(65536), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError("File hash differs: " + name)
    print("Verified 10 payload files and the pinned manifest; no source executed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    verify(parser.parse_args().directory)
