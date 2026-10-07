# Credits and license scope

This results note and its aggregate-analysis/figure tools are original work by **cubres**, licensed under the accompanying [MIT license](LICENSE). They do not contain third-party learned weights or MRI/participant records.

The linked original mirror reader retains its [Apache-2.0 source scope](https://github.com/cubres/rsna-knee-abnormality-detection/blob/454a541e769edec700ea25d2e649c1e9c43d4e71/public-mirror-reader-2026-10-08/LICENSE_SCOPE.md). Its code license does not change the checkpoint publisher's custom Other permission for research and educational model use. The checkpoint is not distributed or relicensed here; attach the [original weights V1](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/versions/1) under the publisher's terms. Separate checkpoint redistribution or sublicensing rights are not inferred.

The full upstream credits, implementation-change notice and publisher-required OAI/NDA acknowledgement remain in the immutable [complete NOTICE](https://github.com/cubres/rsna-knee-abnormality-detection/blob/454a541e769edec700ea25d2e649c1e9c43d4e71/public-mirror-reader-2026-10-08/NOTICE.md). In particular:

- **nartaa / Danial Zakaria** supplied the selected learned checkpoint and public plain/anatomical-mirror recipe in [RSNA Knee 0949 Anatomical Mirror V2](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397). Its displayed 0.949 score belongs to that upstream notebook.
- **dreaddevelopment** supplied the Raptor CoAtNet/MIL recipe and public baseline lineage. The selected weights are the publisher's retrained checkpoints.
- **timm / Ross Wightman and contributors** supplied the CoAtNet implementation and component lineage.
- The checkpoint publisher credits weak-label contributions from **stevenleehans**, **pilkwang** and **riadmohamed42's JEV labels and verification**, alongside the publisher's own report-derived labels. Their tables and reports are not included.
- **RSNA**, the competition contributors, **OAI/NDA**, **NIMH** and **NIAMS** retain their respective data-use and acknowledgement requirements. The complete NOTICE preserves the publisher's acknowledgement and links [NDA Study 3407](https://nda.nih.gov/study.html?id=3407) and [DOI 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676).

The competition's [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) and applicable controlled-access terms govern data use. This repository's software licenses grant no rights to obtain, redistribute or re-identify controlled-access records. Existing [mild-affine code provenance](../tools/mild-study-affine/2026-10-06/PROVENANCE.md) is retained separately from this measured result.
