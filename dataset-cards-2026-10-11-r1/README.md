# RSNA Knee: useful dataset-loading recipes, 2026-10-11

These description proposals correct the load instructions for existing public datasets. No asset was repackaged, no payload was downloaded, and no dataset was updated while preparing this packet. The source descriptions are the owner's existing cards; the revised software examples are original.

Each `cards/<slug>/` directory contains the full proposed description, a filename/size inventory for one exact Kaggle data version, and before/after description digests. `SOURCE_BINDINGS.json` records the exact identity/version/licence and limits. Filename and size agreement does not authenticate a payload. None of these selected releases lists `MANIFEST.json`, so a manifest-based flatpack verifier cannot be applied to them as-is. Previously stated array headers and October 7 hashes remain historical observations.

The proposals preserve dataset identity, title, slug, licence metadata, public visibility, files and versions. They add no rights, medal, performance, compatibility or independent-validation claim. This source package does not attest a live card update or Kaggle Markdown rendering. Description deployment requires an independent review and a fresh identity/version/description binding.

[Campaign experiments and reproduction notes](https://github.com/cubres/rsna-knee-abnormality-detection)

The Knee card corrects a concrete discrepancy: live metadata says `unknown`, whereas the earlier text said `other`. The proposal preserves `unknown` and grants no rights. Its NumPy example inspects numeric fitted arrays without running an encoder, inference or a head. The current anatomical-mirror result is a separate campaign path and is not attributed to this older teacher bundle.
