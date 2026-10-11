## What it is

An archive of two compact fitted linear heads (ridge probe v1 and routed ridge probe v3) with the PCA and scaling arrays needed to apply them to deterministic OrthoDiffusion mid_2 descriptors of knee MRI volumes.

It contains no DICOM images, report text, patient identifiers or competition labels (README.md).

## Contents

| File | Size (bytes) | What it holds |
|---|---:|---|
| `README.md` | 1,407 | Bundle README: descriptor contract, archive hash, licence and receipt caveat |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/pca_axial_components_f32.npy` | 917,632 | PCA components for axial descriptors, ridge probe v1 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/pca_axial_mean_f32.npy` | 7,296 | PCA mean for axial descriptors, ridge probe v1 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/pca_coronal_components_f32.npy` | 917,632 | PCA components for coronal descriptors, ridge probe v1 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/pca_coronal_mean_f32.npy` | 7,296 | PCA mean for coronal descriptors, ridge probe v1 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/pca_sagittal_components_f32.npy` | 917,632 | PCA components for sagittal descriptors, ridge probe v1 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/pca_sagittal_mean_f32.npy` | 7,296 | PCA mean for sagittal descriptors, ridge probe v1 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/receipt.json` | 8,532 | Ridge probe v1 receipt; not opened for this card |
| `rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1/teacher_ridge_head_bundle.npz` | 39,298 | Fitted ridge head arrays for probe v1 (NumPy archive; array names not inspected) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/pca_axial_components_f32.npy` | 1,835,136 | PCA components for axial descriptors, routed probe v3 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/pca_axial_mean_f32.npy` | 7,296 | PCA mean for axial descriptors, routed probe v3 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/pca_coronal_components_f32.npy` | 1,835,136 | PCA components for coronal descriptors, routed probe v3 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/pca_coronal_mean_f32.npy` | 7,296 | PCA mean for coronal descriptors, routed probe v3 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/pca_sagittal_components_f32.npy` | 1,835,136 | PCA components for sagittal descriptors, routed probe v3 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/pca_sagittal_mean_f32.npy` | 7,296 | PCA mean for sagittal descriptors, routed probe v3 (float32 array) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/receipt.json` | 24,612 | Routed ridge probe v3 receipt; not opened for this card |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/teacher_routed_head_bundle.npz` | 34,550 | Fitted routed head arrays for probe v3 (NumPy archive; array names not inspected) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/z_axial_mean_f32.npy` | 1,152 | Mean for axial inputs, routed probe v3 (meaning from file name) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/z_axial_scale_f32.npy` | 1,152 | Scale for axial inputs, routed probe v3 (meaning from file name) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/z_coronal_mean_f32.npy` | 1,152 | Mean for coronal inputs, routed probe v3 (meaning from file name) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/z_coronal_scale_f32.npy` | 1,152 | Scale for coronal inputs, routed probe v3 (meaning from file name) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/z_sagittal_mean_f32.npy` | 1,152 | Mean for sagittal inputs, routed probe v3 (meaning from file name) |
| `rsna_two_teacher_bundle_v1/rsna_teacher_routed_ridge_probe_v3/z_sagittal_scale_f32.npy` | 1,152 | Scale for sagittal inputs, routed probe v3 (meaning from file name) |

The versioned API lists 23 files including README.md, all listed above. The two receipts and the NumPy arrays were not opened for this card; their descriptions come from the file names and sizes.

## How it was produced

README.md: the image encoder is the official MIT-licensed OrthoDiffusion implementation with orientation-specific checkpoints. The descriptor contract is: plane order AXIAL, CORONAL, SAGITTAL; one selected 16-slice, 256 by 256 uint8 volume per plane; normalization x / 127.5 minus 1; feature block mid_2 at timestep 100; four deterministic noise draws averaged; forward and reversed-depth orientations averaged; and 1,792 values per plane (mean, maximum, standard deviation and adaptive 1 by 2 by 2 pooling).

The earlier owner card attributes to README.md that the fitted heads used report-derived pseudo-labels and excluded all 58 fully annotated studies, and every normalized-report duplicate of those studies, from fitting; those exclusions were not independently verified in this refresh. The README says the local validation receipts are diagnostic only and not an official competition score; only a completed Kaggle submission with a public score establishes leaderboard performance.

## How to load it

Read the included `README.md` before loading fitted arrays. The live Kaggle license label is `unknown`; the card does not grant permission for medical-report-derived weights. Confirm the relevant source/competition terms before use. The following numeric inspection recipe neither runs the encoder nor applies a head to a study.

```python
from pathlib import Path
import numpy as np

root = Path("/kaggle/input/rsna-knee-two-teacher-bundle-v1")
base = root / "rsna_two_teacher_bundle_v1/rsna_teacher_ridge_probe_v1"
pca = np.load(base / "pca_axial_components_f32.npy",
              mmap_mode="r", allow_pickle=False)
if pca.dtype != np.dtype("float32") or pca.ndim != 2:
    raise ValueError("Unexpected PCA header contract")
with np.load(base / "teacher_ridge_head_bundle.npz", allow_pickle=False) as head:
    print("PCA shape:", pca.shape, "head array count:", len(head.files))
```

The file-name contract is version-bound below; array shapes and head-key names were not verified in this refresh. Preserve plane order and every descriptor detail in README.md before applying PCA or a head. A completed local array read is not an inference run, a report-source license review, proof of held-study exclusion or an official score. The current anatomical-mirror notebook is a separate campaign path, not evidence that these older two-teacher heads produced its score.

## Licence and provenance

Live Kaggle licence metadata: `unknown` (unchanged). The earlier card called this "other"; that was not the current metadata label. This card supplies no new licence grant. The encoder is the official MIT-licensed OrthoDiffusion implementation, as the README states; the MIT licence text is in the separate OrthoDiffusion bundle. The heads were fitted on report-derived pseudo-labels, and the README does not name the report sources or their licences, so those terms are not established here.

The earlier owner card/README states that the 58 fully annotated studies were excluded from fitting and that RSNA Knee competition terms govern use. This refresh does not independently establish the exclusions or report-source permissions.

## Attribution and citation

The README names the official OrthoDiffusion implementation. No paper citation is given in this bundle.

## Changelog

- Version 1 (2026-08-12): Initial release

## Exact data version and verification scope

This card is bound to [Kaggle data version 1](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-two-teacher-bundle-v1/versions/1), numeric dataset id `11626403`, captured on 2026-10-11T19:35:16.790105+00:00. Dataset title, slug, licence metadata and public visibility are unchanged. The versioned API listed 23 files totalling 8,417,391 uncompressed bytes. No `MANIFEST.json` is listed; no current payload SHA256 was obtained. Existing October 7 header/hash statements above are historical observations, not this refresh's byte verification. A manifest-based flatpack verifier therefore does not apply to this release as-is.

[Existing campaign notebook](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push) · [Public campaign experiments and reproduction notes](https://github.com/cubres/rsna-knee-abnormality-detection). The notebook's complete recipe and official result are separate from a standalone dataset's usefulness; this refresh supplies no new score or compatibility claim. Version-bound file inventories are available with the reviewed card package on GitHub after publication.
