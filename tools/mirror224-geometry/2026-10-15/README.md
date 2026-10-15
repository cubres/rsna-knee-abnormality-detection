# A small anatomical-mirror test set

This original Python demo generates ten invented uint8 triplets at the boundaries of the five slots in a 96-slice layout. It checks which axis to reverse before channel-dependent normalization: slice channels for sagittal slots 0–1, image width for slots 2–4. The expected pixels come from an independent coordinate formula, so a function that merely reverses twice does not pass by itself.

```sh
python3 synthetic_geometry.py --self-test
python3 synthetic_geometry.py > invented_geometry.json
```

Python 3.9+ and its standard library are sufficient. There are no downloads, random draws, patient identifiers, images, weights or model calls. The generated JSON is an MIT-licensed, deterministic synthetic dataset; its `input_uint8` and `expected_uint8` arrays can be used as fixtures for a NumPy or GPU implementation. Six refusal examples cover invalid centres/depth/shape/value types. The toy LUT example demonstrates that sagittal channel reversal and channel-dependent normalization do not commute.

The geometry follows the publicly described Raptor mirror convention in [varshithakolupuri V1](https://www.kaggle.com/code/varshithakolupuri/rsna-0-956-one-free-flip-68-ranks-up?scriptVersionId=358309699), credited there to nartaa. This package contains an original stdlib implementation and invented fixtures, not copied upstream notebook code. Credit also goes to nartaa, dreaddevelopment and Helios for the model and ensemble context. Upstream Apache-2.0 notebook terms and separate model/data terms remain separate from this package's MIT license. No clinical or accuracy claim follows from these fixtures; they do not validate DICOM orientation, preprocessing, GPU numerical parity, source data rights or leaderboard performance.

The companion `PROPOSAL.md` fixes one future comparison: mirror averaging for Raptor224 after the faithful 0.957 recipe has completed. No notebook or model is changed by this package. Existing public context: [The Bee's Knees](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push), [competition repository](https://github.com/cubres/rsna-knee-abnormality-detection).
