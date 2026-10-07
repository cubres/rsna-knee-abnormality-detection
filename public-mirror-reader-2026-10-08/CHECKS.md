# Verification actually completed

The supplied original reader hash is `4c0f5e4ef362941927e79ca3ab7e98a2c17a46d5a9230928b9057ce3e4a7b387`. It was independently source reviewed against the exact upstream V2. Healthy-input geometry, intensity math, mirroring, finding-attention pooling and probability/rank aggregation agreed in source review. Intentional differences and native holds are listed in [the scientific contract](SCIENTIFIC_CONTRACT.md).

The copied source verifier passed **24 checks in normal Python** and **24 checks with `-O`**. Receipts are in [evidence/source_checks.json](evidence/source_checks.json) and [evidence/source_checks_optimized.json](evidence/source_checks_optimized.json). These checks parse/compile the module, inspect constants and AST, import the module without native libraries, and call only the pure mirror-axis helper. They check source/configuration behavior, not model accuracy or native pixel equivalence.

The figures are original schematics generated from the documented constants, not experimental plots. Both revised diagram exports were visually inspected for legibility. The opening banner is disclosed AI-generated art; its prompt and file hash are in [ARTWORK_PROVENANCE.json](ARTWORK_PROVENANCE.json).

No model was loaded, checkpoint fetched, real DICOM decoded, GPU run launched, competition entry submitted or official score reproduced by these checks or packet preparation. All native mirror canary, parity and full-inference claims remain **UNRUN**.
