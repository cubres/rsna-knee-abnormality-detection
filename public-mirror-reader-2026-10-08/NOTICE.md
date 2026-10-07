# Attribution, changes and required acknowledgement

## Upstream lineage

- **nartaa / Danial Zakaria**: selected learned checkpoint and plain/anatomical-mirror inference recipe in [RSNA Knee 0949 Anatomical Mirror, V2](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397), Apache-2.0. The displayed 0.949 score belongs to that upstream notebook.
- **dreaddevelopment**: [Raptor CoAtNet/MIL recipe and public baseline weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense). The selected nartaa weights are the publisher's retrained checkpoints, not these original baseline weights.
- **timm / Ross Wightman and contributors**: [CoAtNet implementation and pretrained component lineage](https://github.com/huggingface/pytorch-image-models). Third-party components retain their own licenses.
- The checkpoint publisher credits weak-label contributions from [stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), [pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), and [riadmohamed42's JEV labels and verification](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification), together with the publisher's report-derived labels. These tables and source reports are not included.
- **RSNA and competition data contributors**: data use remains subject to the [RSNA Knee competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

## Original implementation and changes

`src/public_mirror_reader.py` is an original readable implementation of the visible upstream recipe, not a copy of private campaign code or an independently trained model. It adds checkpoint/source identity constants, strict sample-schema and study coverage gates, tensor-only deserialization, failure on decode/model errors, bounded thread queues without multiprocessing, separate canary output names and create-only output files. It does not execute the upstream fast-reader payload. The selected checkpoint bytes, geometry and aggregation formulas remain explicitly attributed to the public references. These differences require native validation; this packet makes no reproduced-score or bit-equivalence claim.

## Publisher's required OAI / NDA acknowledgement

The following paragraph is preserved verbatim from the upstream publisher's credits and dataset description. It acknowledges the upstream research lineage; distributing this reader does not assert that this repository obtained or redistributed controlled-access OAI records.

Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.

[Shared NDA Study 3407](https://nda.nih.gov/study.html?id=3407) · [DOI: 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676). Official acknowledgement source: [NDA manuscript preparation](https://nda.nih.gov/nda/manuscript-preparation).

The upstream weights package grants only its stated model-use permission. OAI access and redistribution remain governed by NDA. This packet contains no OAI or competition MRI, reports, source label tables, participant identifiers or learned checkpoint payloads.
