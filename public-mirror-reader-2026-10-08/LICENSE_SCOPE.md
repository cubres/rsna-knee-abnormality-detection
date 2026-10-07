# License scope and redistribution

| Material | Scope in this packet | Terms |
|---|---|---|
| `src/public_mirror_reader.py` | Original expression of the upstream Raptor/nartaa inference recipe; retains attribution and implementation-change notice | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| Upstream nartaa notebook source | Referenced, not bundled | Apache-2.0 as shown on the exact public V2 card |
| Original README, scientific exposition, diagrams, supplied AI-generated banner, diagram generator and source-check tool | Original materials in this packet | [MIT](LICENSES/MIT.txt), copyright 2026 cubres |
| Selected nartaa checkpoint | **Not included**; attach original dataset V1 | Publisher's custom Other / research and educational model-use permission |
| timm and other installed libraries | **Not bundled** | Their respective upstream licenses; timm is Apache-2.0 |
| Competition MRI/CSV inputs and OAI records | **Not included** | Competition rules and, where applicable, controlled-access NDA terms |

The Apache license and this MIT grant cover their stated code/documentation scopes. They do not change the checkpoint's terms or grant access to controlled-access data. Apache source-derived material retains its upstream attribution and notices; the readable implementation and its intentional engineering changes are described in [NOTICE.md](NOTICE.md).

The publisher states that its learned weights may be used for research and educational purposes, including the RSNA Knee competition. That model-use permission explicitly grants no rights to obtain, publish, redistribute or re-identify OAI participant data, source MRI images, reports or controlled-access records. It does not explicitly grant separate checkpoint redistribution or sublicensing rights. This packet therefore links the original versioned dataset rather than redistributing or relicensing its payload. Do not put those weights under MIT, Apache-2.0 or CC0 merely because the surrounding code uses an open-source license.

Attach the [original weights V1](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/versions/1) under its published terms, retain [the complete acknowledgement](NOTICE.md), and follow the [competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) and [NDA access terms](https://nda.nih.gov/). No raw data, model weights, patient identifiers, reports, study-level labels, prediction tables or private notebook sources are distributed here.
