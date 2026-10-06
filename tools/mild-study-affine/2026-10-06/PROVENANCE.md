# Provenance and evidence

The helper, verifier, and pipeline renderer are original code written for this
reusable tool. No public competition implementation was copied. The diagram
uses synthetic grid marks and invented study inputs, not medical images.

Reviewed source bytes are preserved:

| File | SHA256 |
| --- | --- |
| mild_affine.py | a47e8dfa175706b61946b3c21c819711c7047e682cd9e6c97e022d63e8a64dbc |
| verify_geometry.py | c8cdd272492482e9ef7290fc97cde85c424865a85aa9748b9ba34f3283bcd221 |

The inverse affine matrix follows an independent derivation in physical
pixel coordinates, then conversion into Torch's normalized coordinates.
The verifier compares it with a homogeneous forward matrix inverted by a
linear solver, including rectangular images to expose aspect-ratio errors.
Independent source review checked geometry, RNG isolation, alignment,
padding, autocast, and stress gates.

Primary API references:

- [Torch 2.10 affine_grid](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.affine_grid.html): theta shape and matched align_corners setting.
- [Torch 2.10 grid_sample](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.grid_sample.html): normalized coordinates, bilinear mode, and zero padding.
- [Torch 2.10 AffineGridGenerator.cpp](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/AffineGridGenerator.cpp): grid generation uses bmm, motivating the locally disabled autocast scope.
- [NumPy PCG64](https://numpy.org/doc/2.2/reference/random/bit_generators/pcg64.html): the explicitly owned bit generator.

No source text or diagrams from those references are reproduced. The MIT
license applies to the original files in this directory, not to dependencies
or linked materials. NumPy and Torch keep their own licensing.

`verification_cpu_normal.json` and `verification_cpu_optimized.json` come
from executing the copied verifier with existing local CPU libraries. They
record exact code hashes, Python/NumPy/Torch versions, synthetic checks,
numerical errors, and padding measurements. They use no real study UID,
image, label, model, checkpoint, prediction, or notebook. The execution
evidence is CPU only; CUDA/native ordering is not attested by these receipts.

The 81-point screen is finite. The every-applied-grid center-outside check
enforces the 0.10 boundary for actual sampled transforms. Strict saturated
output checks may hold on a kernel with rounding overshoot, without clipping.
No anatomical, speed, training, generalization, or competition-score benefit
is established by this package.
