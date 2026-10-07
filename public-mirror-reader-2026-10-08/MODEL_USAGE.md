# Selected checkpoint: attach the publisher's version

The original reader selects only `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` from [nartaa's publication dataset, version 1](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/versions/1). Its publisher-provided SHA-256 is:

```text
7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef
```

The module binds that digest before deserialization. This packet has not fetched or independently hashed the actual checkpoint payload, and tensor-only deserialization has not yet been tested natively. Treat the digest here as a publisher identity pin, not a completed native verification receipt. No weights are distributed in this repository.

The publisher describes the selected accuracy checkpoint as a seed-42 model trained at 384 pixels with SWA over epochs 14, 11 and 12, then scored using native 320-pixel inputs, 94 windows and anatomical mirroring. The other 224-pixel efficiency checkpoint in that dataset is a distinct model and is not loaded by this reader. These aggregate details do not supply a complete independently reproducible training recipe.

The publisher says its learned weights used masked external supervision from 2,399 baseline OAI knees. Its Gold58 set was reused for selection and is not independent cross-validation. The public score establishes neither clinical validity nor statistical significance. This repository neither includes nor grants access to that supervision, source MRI, reports or participant records.

## Publisher's model-use terms

The following model-use text is reproduced from the original versioned dataset description, separately from every code license:

> These learned weights are provided for research and educational use, including the RSNA Knee competition. This model-use permission grants no rights to obtain, publish, redistribute or re-identify OAI participant data, source MRI images, reports or other controlled-access records. Access to OAI data remains governed by NDA's terms. Third-party source code and pretrained components retain their respective licences. No CC0 or other open-data licence is applied to OAI data or raw corpora by this model release.

The dataset's category is **Other**. These terms do not explicitly grant separate checkpoint redistribution or sublicensing permission. Attach the original V1 under its own terms, keep [the required attribution and full NDA acknowledgement](NOTICE.md), and do not apply MIT, Apache-2.0 or CC0 to checkpoint bytes.
