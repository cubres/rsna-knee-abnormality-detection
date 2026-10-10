# RSNA Knee 2026: One Team's Submissions, Scored

An annotated ledger of one team's **90 official submissions** to the RSNA Knee Abnormality Detection competition on Kaggle (6 August to 10 October 2026). Each row gives the public score, method, components, blend weights, outcome and notes, including the **negative results** that did not help. It also includes the Efficiency Prize formula with illustrative worked examples.

This is one team's campaign record (account prvsiyan). It is not a leaderboard, not an official publication, and it contains no competition images, labels or hidden-set content. It is not produced or endorsed by Kaggle, RSNA or the competition hosts.

## What

### Headline findings

- **This team's best public score is 0.949**, reached by four rows: 56948803 (bench notebook), 56949940 (public notebook, version 34), 56952077 (mirror plus 224 px single view, validation) and 57024205 (four-view mirror TTA, clean replicate).
- **The 0.949 reader is the anatomical mirror reader** (nartaa recipe): plain and mirror probabilities averaged on a 384 px SWA checkpoint, 320 px crop, per-finding percentile rank. It is +0.008 over the 0.941 plateau of the team's earlier public notebook (rows 56759401, 56779708, 56779710).
- **Several plausible additions tied or lost:** a ConvNeXt 2.5D rank blend (0.948, rows 56955785 and 56957785), a 0.55 / 0.45 BK-A plus mirror blend (0.947, row 57024089) and a MaVIT standalone reader (0.935, row 57016763). None beat mirror-only.
- **Hidden reruns can fail.** Of the six NO_SCORE rows, three raised unhandled errors, two wrote no submission file and one exceeded the runtime limit. One scored row (57017206, 0.505) is annotated as degraded.
- **Efficiency:** under illustrative constants, one AUC tick (0.001) is worth about 70 s of runtime. The dataset publishes no runtime for any row. Row 57024289 (plain-view-only reader, 0.948) is the team's candidate efficiency final; the decision is pending.

### Method families (experiments.csv)

| Family | Scored / rows | Best public | Best row | Outcome |
|---|---|---|---|---|
| Report-text reader lineage | 5 / 5 | 0.891 | 55354294 | superseded |
| Legacy target-specific DINOv2 blends | 9 / 9 | 0.894 | 55361195 | plateau |
| RadImageNet rank blends | 4 / 4 | 0.906 | 55399732 | reference |
| DINOv3 consensus and early H&S versions | 6 / 7 | 0.922 | 55646554 | plateau |
| CPU students and depth-index fix | 5 / 5 | 0.883 | 55470873 | superseded |
| Forks of public frontier notebooks | 4 / 4 | 0.920 | 55606155 | plateau |
| DINOsaur and Raptor fusion (H&S lineage) | 15 / 15 | 0.936 | 55765530 | plateau |
| Philosophical DINOsaur and SlotHead residual | 2 / 2 | 0.936 | 55909953 | rejected |
| Public notebook versions and 2026-09-09 batch arms | 11 / 11 | 0.940 | 56107593 | plateau |
| probe22 per-finding outer weights | 3 / 4 | 0.941 | 56354597 | plateau |
| Public notebook versions 31 and 32 | 3 / 3 | 0.941 | 56759401 | superseded |
| BK-A stack | 6 / 7 | 0.943 | 56795351 | plateau |
| Anatomical mirror reader (nartaa recipe) | 2 / 3 | 0.949 | 56948803 | reference |
| Mirror plus nartaa 224 px blend | 1 / 1 | 0.949 | 56952077 | tie |
| Mirror plus goodpjw ConvNeXt blend | 2 / 2 | 0.948 | 56955785 | rejected |
| BK-A without RadImageNet (scored alone) | 1 / 1 | 0.938 | 57016729 | reference |
| Plain-view-only anatomical reader (candidate) | 1 / 2 | 0.948 | 57024289 | candidate_efficiency |
| Four-view anatomical-mirror TTA | 2 / 3 | 0.949 | 57024205 | tie |
| MaVIT 288 px standalone probe | 1 / 1 | 0.935 | 57016763 | rejected |
| Mirror plus BK-A 0.55 / 0.45 blend | 1 / 1 | 0.947 | 57024089 | rejected |

### Negative results: what did not help

Scores are public AUC. Terms are defined in the Glossary.

- **Legacy DINOv2 blends (F02):** seven rows sit at 0.894 (rows 55361195 to 55363947). The row with an empty Kaggle description scored 0.893 (row 55360051), and the Lateral Meniscus blend fell to 0.891 (row 55364278). Target-specific weights gave no measurable public gain.
- **ConvNeXt 2.5D on the mirror reader, 0.70 / 0.30:** 0.948 on validation (row 56955785) and 0.948 public (row 56957785), 0.001 below mirror-only (0.949).
- **Mirror plus 224 px single view, 0.75 / 0.25:** tie at 0.949 (row 56952077).
- **BK-A plus mirror, 0.55 / 0.45:** 0.947 (row 57024089), 0.002 below mirror-only.
- **Four-view mirror TTA:** the clean replicate tied at 0.949 (row 57024205). Row 57016750 has no score, and row 57017206 scored 0.505 and is annotated as degraded.
- **Plain-view-only reader:** 0.948 (row 57024289), 0.001 below mirror-only. An earlier run (row 57016738) has no score.
- **BK-A without the RadImageNet member:** 0.938 (row 57016729), scored alone; the team annotates it as not a blend measurement. Full BK-A is 0.943 (row 56795351).
- **Student fleet:** 0.918 alone (row 56800704); a 15 percent student blend on the BK native stack gave 0.943 (row 56818632), the same as the stack alone (row 56795351).
- **MaVIT 288 px standalone:** 0.935 (row 57016763), below the 0.943 gate in its description, so no blend was run.
- **DINOv3 consensus, two-view depth TTA and anatomy-query specialist:** 0.920 to 0.922 (rows 55676080, 55693183, 55693329, 55701391), no gain over the base reproduction (row 55646554, 0.922). The frozen D224 branch fell to 0.917 (row 55726212).
- **DINOsaur overlays:** OrthoDiffusion 0.935 (rows 55805823, 55808850); OrthoFoundation Lateral OA 0.935 (row 55800346); a SlotHead residual dropped the 0.936 control to 0.934 (row 55943980).
- **RadImageNet overlays:** an EfficientNet-B3 10 percent blend scored 0.898 (row 55396029), below 0.899 (row 55391852); a BiomedCLIP and ViTDet overlay fell to 0.887 (row 55424726).
- **Batch arm G:** 0.922 alone (row 56122418) and 0.922 in A+B+G (row 56122723), against 0.940 for arms A, B and A+B.
- **No score:** rows 56355206 (runtime limit), 56795572 and 55458627 (unhandled errors).

### Positive results that stuck

- **Mirror reader, 0.941 to 0.949 (+0.008):** rows 56948803 and 56949940.
- **Depth-index alignment, 0.851 to 0.859 (+0.008):** row 55563720 against 55563560.
- **20 percent RadImageNet rank blend, 0.899 to 0.906 (+0.007):** row 55399732.
- **CC0 MaxSpan Raptor leg, 0.933 to 0.935 (+0.002):** row 55727399.

### Efficiency Prize formula (illustrative constants)

    Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400      (lower is better)

The team used Benchmark 0.5 and maxAUC 0.96, approximations not confirmed against the competition page in this release. Under them, one AUC tick is about 70 s of runtime. This dataset publishes no runtime for any row. `efficiency_formula.md` gives the worked examples and says what is measured.

## Why

Kaggle's leaderboard shows one score per submission, not what was tried or what failed. This ledger records the method, components, blend weights, outcome and notes of each official submission, including the negative results.

## Files

| File | Contents |
|---|---|
| `submissions.csv` | One row per official submission (90 rows, 12 columns). |
| `experiments.csv` | One row per method family (20 rows, 14 columns). |
| `efficiency_formula.md` | Efficiency Prize formula and illustrative examples. |
| `schema.json` | Column dictionary for both CSV files. |
| `README.md` | This card. |
| `LICENSE`, `NOTICE.md` | CC BY 4.0 licence and per-file terms. |
| `LICENSE-APACHE-2.0` | Apache License 2.0 text for the two code snippets. |
| `MANIFEST.json` | SHA-256 and byte size of each file above. |
| `dataset-metadata.json` | Kaggle metadata. |

## Columns

### submissions.csv (90 rows)

Full definitions, types and allowed values are in `schema.json`.

| Column | Meaning |
|---|---|
| `row_id` | Kaggle submission reference. Unique key. |
| `utc` | Submission time from Kaggle, UTC, to the second. |
| `notebook_kind` | `public`, `bench` or `unrecorded`. `public` and `bench` are the team's notebook categories. `unrecorded` (51 rows): the sources do not establish the category. |
| `family_id` | Joins to `experiments.csv`. |
| `method`, `components`, `weights` | Short label; models used; blend weights as predeclared, or `not recorded in the row description`. |
| `public_score` | Kaggle public AUC, 3 decimals. Empty for NO_SCORE rows. |
| `status`, `no_score_reason` | `SCORED` or `NO_SCORE`. For NO_SCORE rows, `no_score_reason` is Kaggle's errorDescription, quoted verbatim. |
| `outcome` | Relative to the family; allowed values and meanings are in `schema.json`. |
| `notes` | Verbatim Kaggle description quotes (`[...]` marks an omission), score comparisons and row ids. Judgements not in the description are labelled `Team annotation`. |

The dataset publishes no runtimes. Kaggle does not return hidden-run runtimes, and the team's runtime receipts are not reproduced here.

### experiments.csv (20 families)

`family_id`, `family_name`, `method_summary`: identifiers and text. `n_rows`, `n_scored`, `n_no_score`: counts. `best_public_score`, `min_public_score` (3 decimals), `mean_public_score` (4 decimals): computed over scored rows. `best_row_id`: earliest row attaining the best score. `first_utc`, `last_utc`: UTC range. `outcome`: family-level status. `verdict`: paragraph with row ids.

### Glossary

- **Public score:** Kaggle's public-leaderboard AUC (area under the ROC curve), 3 decimals. One tick is 0.001.
- **Predeclared:** a weight, gate or rule the row description states before the run.
- **Team annotation:** a judgement the team recorded for a row that the Kaggle description does not contain. Labelled wherever it is used.
- **Public notebook version N:** the Kaggle version number of a public notebook.
- **Bench notebook:** the team's private validation notebook category (`notebook_kind` = `bench`).
- **Hidden rerun:** the competition's private re-execution of a notebook. Kaggle does not return its runtime.
- **Fail-closed:** a notebook chain writes no submission, or falls back to a reference path, instead of writing degraded output.
- **Degraded table:** a near-constant output file.
- **OOF:** out-of-fold predictions, used to choose weights without the public split.
- **Rank blend:** a weighted average of ranks.
- **Gold58:** the team's name for its 58-annotation gold set, as the row descriptions use it.
- **TTA:** test-time augmentation, averaging predictions over transformed inputs.
- **SWA checkpoint:** a model checkpoint from stochastic weight averaging.
- **Plain view / mirror view:** the study as given, and a mirrored view whose probabilities are averaged with it.
- **nartaa:** author of the public notebooks and SWA checkpoint dataset behind the mirror reader. See Credits.
- **BK-A:** the team's own orchestration of a public stack. "Rad off" means without its RadImageNet member.
- **RadImageNet:** a radiology-pretrained backbone used as one ensemble member.
- **H&S:** the notebook title "Head and shoulders, knees and toes".
- **DINOsaur, Raptor:** public model lines and checkpoints that the team forked or reproduced (see Credits).
- **probe22:** the team's probe family with per-finding outer weights and a 55/10/15/20 view split.
- **Target abbreviations** (ACL, MCL, Effusion, Synovitis, Medial OA, Lateral OA, PF OA, LM, LOA): knee finding names as the team's notes use them.

## How to load

The loader below is Apache-2.0 code (see Licence).

```python
import pandas as pd

subs = pd.read_csv("submissions.csv", parse_dates=["utc"])
scored = subs[subs["status"] == "SCORED"].astype({"public_score": float})
print(scored.nlargest(5, "public_score")[["row_id", "method", "public_score"]])
```

## Provenance

### Where each field comes from

| Fields | Source |
|---|---|
| `row_id`, `utc`, `public_score`, `status`, `no_score_reason` | Kaggle submissions list, snapshot 10 October 2026, 11:47:18 UTC. |
| Quoted text in `notes` | Kaggle submission description, same snapshot (verbatim). |
| Other fields | The team's annotation of each submission. |
| `experiments.csv` counts and aggregates | Computed from `submissions.csv`. |
| Efficiency formula | Competition overview text as recorded by the team on 8 October 2026. Constants are approximations. |

### How it was produced

1. The team's submissions were read from the Kaggle submissions list: 90 rows, 84 with a public score and 6 with an error description.
2. Each row was annotated from its Kaggle description and the team's records. Quoted text in `notes` was copied from the Kaggle description.
3. Public scores, status and error descriptions were checked against the 11:47:18 UTC snapshot, with no mismatches. Counts and aggregates were computed by script. Runtime receipts, local paths, worker identifiers, credentials and internal review text were removed.

### Caveats

- **One team, public split only.** The private score is not in the data. The 0.949 rows may not rank the same on the private split. Differences of 0.001 are one leaderboard tick and are within noise.
- **Notebook category is often unrecorded:** 51 of 90 rows are `unrecorded` in `notebook_kind`.
- **Competition rules are not settled here.** Whether public weights trained on competition data may be used in prize-eligible submissions was an open question at the snapshot date. The components column describes what was used, not eligibility.
- **Efficiency inputs are not published.** No row has a runtime here, and the constants are approximate. The runtime definition is unresolved.
- **Snapshot date.** Scores are as of 10 October 2026. Later submissions are not included.

## Credits

Upstream authors and components named in the rows. "Verified" means checked on the Kaggle page on 2026-10-10 (nartaa notebooks only); "as stated" means the Kaggle row description says so; "as recorded" means the team's records of that date, not re-verified; "not recorded" means no licence was found. No third-party file is redistributed.

| Upstream | Used for | Licence |
|---|---|---|
| nartaa (Danial Zakaria), notebook [rsna-knee-0949-anatomical-mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) | Mirror reader recipe (rows 56948803 to 57024289) | Apache-2.0, verified 2026-10-10 |
| nartaa, notebook [rsna-knee-0945-efficient-224crop](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop) | 224 px single-view arm (row 56952077) | Apache-2.0, verified 2026-10-10 |
| nartaa, SWA checkpoint dataset `nartaa/rsna-knee-publication-swa-weights-20261007` | 384 px and 224 px checkpoints | "Other (specified in description)", as recorded (not re-verified). Not redistributed |
| goodpjw2008, three-fold ConvNeXt 2.5D reader | Mirror blend (rows 56955785, 56957785) | Apache-2.0, as stated in the row descriptions |
| mattiaangeli, public notebook | Exact public 0.922 reproduction (row 55646554) | Apache-2.0, as stated in the row description |
| sofiaanjenje, saidmohamedomary, amanatar (public notebook forks) | Forks in rows 55573246, 55586830, 55645841 | Apache-2.0, as stated in the row descriptions |
| Roman, DINOsaur public line | Reproduction (row 55768059) and port (row 55728923) | Apache-2.0, as stated in the row descriptions |
| Protocol Fusion public recipe | Rows 55728658 and 55728823 | Author not named in the row descriptions. The port is Apache-2.0 as stated in row 55728823. The original recipe's licence is not recorded |
| dreaddevelopment: raptor-knee-maxspan, raptor-knee-native384 | Raptor CoAtNet checkpoints (rows 55727399 onward) | CC0-1.0, as recorded on 2026-09-13 (not independently re-verified). Not redistributed |
| BMEII-AI, RadImageNet (github.com/BMEII-AI/RadImageNet) | RadImageNet member (for example rows 55399732, 55470873) | CC-BY-NC-SA-4.0 on Kaggle copy `marwanmath/resnet-50-radimagenet-marwan`, as recorded 2026-08-09; another copy records "other". Not reconciled; check upstream before reuse |
| Meta AI, DINOv2 and DINOv3 backbones | DINOv2 ensemble (F02); DINOv3 components (F06, F07) | Community copies as recorded: `ericwang03/rsna-knee-dinov2-mil-bundle` "other" (2026-08-10); `dragonseal/dinov2-fold2-train-weights` "unknown" (2026-08-10); `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` "other" (2026-09-14). Meta's terms not recorded |
| pilkwang/rsna-knee-weights | Attached as an input in the H&S notebook metadata; contents not inspected | CC0-1.0, as recorded on 2026-09-14 (not independently re-verified). Not redistributed |
| OrthoDiffusion, Hugging Face model `lanstat0123/orthodiffusion` | OrthoDiffusion specialists (rows 55805823, 55808850) | MIT, as observed on the model page in the team's admission audit of 2026-09-14 (not independently re-verified) |
| OrthoFoundation | OrthoFoundation Lateral OA specialist (row 55800346) | Not recorded in the team's sources |
| BiomedCLIP and ViTDet | Four-target overlay (row 55424726) | Not recorded in the team's sources |
| EfficientNet-B3 | Five-fold branch (row 55396029) | Not recorded in the team's sources |
| ConvNeXtV2-B and EfficientNetV2-L | probe22 arms (row 56355204) | Not recorded in the team's sources |
| Public MRI teacher families | Student distillation (row 55468012) | Not recorded in the team's sources |
| Mattia (public reference notebook) and Renta recipe | Reproductions (rows 56107593, 56016348) | Not recorded in the team's sources. The Renta row depends on a pinned private checkpoint, not redistributed |
| hengck23, MaVIT dataset `hengck23/hengck23-rnsa-knee` | Standalone probe (row 57016763) | "MIT dataset", as stated in the row description. Not redistributed |

Dependencies of the shipped code:

| Package | Used by | Licence |
|---|---|---|
| pandas | README loader ("How to load") | BSD-3-Clause |

`efficiency()` uses only the Python standard library. No other third-party package is imported by shipped code, and the readers' runtime libraries are not recorded in the sources.

### How to cite

prvsiyan (2026). *RSNA Knee 2026: One Team's Submissions, Scored* [Dataset]. Kaggle. https://www.kaggle.com/datasets/prvsiyan/rsna-knee-2026-experiment-ledger

## Licence

- **Dataset licence: CC-BY-4.0.** This is the single Kaggle licence field. The full legal code is at https://creativecommons.org/licenses/by/4.0/legalcode, and `LICENSE` gives the scope.
- **Why this licence.** The primary content is the team's own submission rows and their public scores as displayed on Kaggle, with the team's annotations. That is a compilation for which attribution is the condition. It contains no competition images, labels, predictions or hidden-set content.
- **Code under Apache-2.0.** Two code snippets are Apache-2.0: the pandas loader under How to load, and the `efficiency()` function in `efficiency_formula.md`, section 6. The full text is in `LICENSE-APACHE-2.0`. `NOTICE.md` says how the two licences coexist.
- **Third-party material.** No checkpoint, weight or competition file is redistributed. Upstream licences are in Credits.
- **Not official.** An independent record, not produced or endorsed by Kaggle, RSNA or the competition hosts.

## Changelog

- **1.3.0 (2026-10-10):** Licence changed to CC-BY-4.0, because the primary content is the team's submission rows and displayed public scores. Internal build labels removed from all text. Unsupported figures removed, including the runtime figures; the `runtime_note` column is removed. `notes` quote Kaggle descriptions verbatim. Credits added for the nartaa notebooks and the named components, with licences as recorded, and a dependency table added. An unsupported frontier-score claim is removed.
- **1.2.0 (2026-10-10):** Licence set to CC-BY-NC-SA-4.0 (superseded by 1.3.0). `LICENSE-APACHE-2.0` and `NOTICE.md` added. `no_score_reason` quotes Kaggle's errorDescription verbatim. Credits for Raptor, RadImageNet, DINOv2, DINOv3 and rsna-knee-weights.
- **1.1.0 (2026-10-10):** One-team title and subtitle; glossary and provenance added.
- **1.0.0 (2026-10-10):** First release.
