# RSNA Knee 2026: public notebook scores and assets

Dataset id: `prvsiyan/rsna-knee-2026-public-notebook-scores-and-assets`. Publisher: prvsiyan. Snapshot: 8 to 10 October 2026. Licence: CC-BY-4.0 for the dataset (see Licence). The example code in this README is Apache-2.0 (see `NOTICE.md`).

## What

A snapshot of the public Kaggle notebooks written for the RSNA Knee Abnormality Detection competition (`rsna-knee-abnormality-detection`), taken 8 to 10 October 2026. It contains:

- 24 public notebooks: the score each code page displays, the notebook version it refers to, votes, a short method description, blend weights and the licence the page shows (`public_notebooks.csv`).
- 38 links between those notebooks and the datasets or models they attach (`notebook_assets.csv`), and the 26 datasets and models involved, with licence, size and platform counts (`assets.csv`).
- The Efficiency Prize board as saved on 8, 9 and 10 October 2026 (`efficiency_board_20261008.csv`, `efficiency_board_20261009.csv`, `efficiency_board_20261010.csv`).
- The first 3,000 rows of the main leaderboard, captured on 8 October 2026 (`leaderboard_top3000_20261008.csv`).
- The publisher's own submission rows and their public scores, with row ids (see "Team submissions").

Team names are replaced by pseudonymous keys (see "Team names and keys"). The dataset contains no images, labels, study data, model weights or predictions.

## Why

It answers three questions. Which public notebooks show which score. Which datasets each notebook attaches, and under which licence. How the Efficiency Prize board moved over three days.

Every score, vote and count is dated. Platform numbers change daily, so check the capture time in each file before comparing values across files.

## Files

| File | Rows | Columns | What it is |
|---|---|---|---|
| `public_notebooks.csv` | 24 | 19 | One row per notebook: displayed score and its label, version, votes, method family, ConvNeXt blend weight, licence, what it attaches. |
| `notebook_assets.csv` | 38 | 5 | Join table: which notebook attaches which dataset or model, how that link was established, and whether the asset has a row in `assets.csv`. |
| `assets.csv` | 26 | 14 | One row per dataset or model the notebooks attach or the publisher's listing review found: owner, licence as listed and normalised, size, update time, platform counts, status. |
| `efficiency_board_20261008.csv` | 5,290 | 5 | Efficiency Prize board as saved on 8 October 2026. |
| `efficiency_board_20261009.csv` | 5,374 | 5 | Efficiency Prize board as saved on 9 October 2026. |
| `efficiency_board_20261010.csv` | 5,477 | 5 | Efficiency Prize board as saved on 10 October 2026. |
| `leaderboard_top3000_20261008.csv` | 3,000 | 4 | The first 3,000 rows of the main leaderboard, captured on 8 October 2026. |
| `schema.json` | | | Column dictionary for every CSV: name, type, unit, allowed values, meaning, row counts. |
| `README.md` | | | This card. |
| `LICENSE` | | | Licence terms: CC-BY-4.0 for the dataset, with per-file notes for the efficiency board files and the source notebook. |
| `NOTICE.md` | | | Which licence covers which kind of file (CC-BY-4.0 tables, Apache-2.0 example code) and attribution for the upstream notebooks. |
| `MANIFEST.json` | | | SHA-256 and byte size of every other file in this dataset, so copies can be verified. |
| `dataset-metadata.json` | | | Kaggle dataset metadata used to publish this dataset. |

## Columns

Every column of every CSV is listed below, from `schema.json`. Each file's header and row count were checked against that schema before publication. The three efficiency board files have the same five columns.

### `public_notebooks.csv` (24 rows, 19 columns)

| Column | Type | Unit | Allowed values | Meaning |
|---|---|---|---|---|
| `notebook_ref` | text | owner/slug on Kaggle |  | Notebook identifier, owner/slug. |
| `owner` | text | Kaggle username |  | Account that owns the notebook. The team notebook is owned by prvsiyan. |
| `title_as_displayed` | text |  |  | Title as shown on the code page or in the kernel list at capture time. |
| `version` | text | Kaggle version |  | Kaggle notebook version the score refers to (vN). Blank when not recorded. |
| `votes` | integer | votes |  | Votes at capture time (see votes_snapshot). Blank when not recorded. |
| `votes_snapshot` | text |  |  | Which capture the votes came from: scorecard 15:03 UTC or kernel list 14:42 UTC, both on 8 October 2026. |
| `displayed_score` | decimal (3 dp) | score |  | Score on the code page, rounded to three decimals by the platform. Blank when not recorded. For score_label P it is a claim, not a render. |
| `score_label` | text |  | V, P, - | V: rendered on the code page on the capture date. P: author claim, title or listing, not on the render. -: no score recorded. |
| `scorecard_rank` | integer | rank | 1 to 20 | Rank on the 8 October 15:03 UTC scorecard. Blank if the notebook is not on it. |
| `method_family` | text |  |  | Method as read from the source code, or the claim where marked P. "not opened" where the notebook was not read. |
| `convnext_mean_weight` | decimal fraction | fraction of blend | 0 to 1 | Mean weight of the goodpjw2008 ConvNeXt reader in the blend of the displayed version. Blank when that version has no ConvNeXt reader or the weight is not recorded. |
| `weight_origin` | text |  |  | Where the blend weights come from: copied between notebooks, gated on gold58, hard-coded, or set by the author. |
| `uses_oai_derived_checkpoint` | text |  | yes, no, not established | Whether the notebook uses the nartaa checkpoint, which the dataset description says was trained with masked supervision on OAI knees. |
| `uses_nc_radimagenet_component` | text |  | yes, likely, no, not established | Whether a CC BY-NC-SA 4.0 RadImageNet-derived file is attached or used. "likely" when the stack the notebook builds on is documented to include one. |
| `notebook_licence` | text |  |  | Licence shown on the rendered page, or "not read". |
| `licence_label` | text |  | V, - | V: licence read on the rendered page. -: not read. |
| `assets_itemised` | text |  | yes, partial, no | yes: every attached dataset is listed in notebook_assets.csv. partial: some are listed. no: none are listed. |
| `assets_note` | text |  |  | What is missing from notebook_assets.csv for this notebook, or how the link was established. |
| `notes` | text |  |  | Verified facts, claims and caveats for the row. Evidence labels V, P and I. |

### `notebook_assets.csv` (38 rows, 5 columns)

| Column | Type | Unit | Allowed values | Meaning |
|---|---|---|---|---|
| `notebook_ref` | text | owner/slug on Kaggle |  | Notebook from public_notebooks.csv. |
| `asset_ref` | text | owner/slug on Kaggle |  | Dataset or model the notebook attaches. |
| `asset_type` | text |  | dataset, model | Kaggle dataset, or Kaggle Model (not in assets.csv). |
| `basis` | text |  | attachment list read, source text, inferred from survey text | How the link was established: read from the notebook's attachment list; stated in the source text; or inferred from the survey text. |
| `in_assets_csv` | text |  | yes, no | Whether asset_ref has a row in assets.csv. |

### `assets.csv` (26 rows, 14 columns)

| Column | Type | Unit | Allowed values | Meaning |
|---|---|---|---|---|
| `asset_ref` | text | owner/slug on Kaggle |  | Dataset identifier. |
| `owner` | text | Kaggle username |  | Dataset owner. |
| `title` | text |  |  | Dataset title as the platform reports it. |
| `licence_as_listed` | text |  |  | Licence as the platform reports it. |
| `licence_normalised` | text |  | CC0-1.0, Apache-2.0, MIT, CC-BY-4.0, CC-BY-NC-SA-4.0, other, unknown | The same licence in one vocabulary. "other" means the licence is given in the description, and "unknown" means the platform gives none. |
| `size_bytes` | integer | bytes |  | Total size of the dataset as the platform reports it (decimal bytes). |
| `size_mb` | decimal | MB (10^6 bytes) |  | size_bytes divided by 10^6, rounded to 0.1. |
| `current_version` | integer | version |  | Current version number as the platform reports it. |
| `last_updated_utc` | text | UTC, ISO 8601 |  | Time of the last update, as the platform reports it. |
| `api_kernel_count` | integer | count |  | Number of public notebooks attaching the dataset, as the platform reports it, in the capture named in metadata_captured_utc. Blank if not reported. |
| `api_download_count` | integer | count |  | Download count as the platform reports it, in the same capture. Blank if not reported. |
| `mounted_by_notebooks` | integer | count |  | Number of rows in notebook_assets.csv that link this dataset to a notebook in public_notebooks.csv. |
| `metadata_captured_utc` | text | UTC, ISO 8601 |  | Time of the saved metadata capture that supplies this row. |
| `status_for_frontier` | text |  |  | Publisher assessment: in use, candidate, training data, or not for use, with the reason. Source-cited. |

### `efficiency_board_20261008.csv`, `efficiency_board_20261009.csv`, `efficiency_board_20261010.csv` (5,290 / 5,374 / 5,477 rows, 5 columns each)

| Column | Type | Unit | Allowed values | Meaning |
|---|---|---|---|---|
| `EfficiencyRank` | integer | rank | 1 onwards | Rank on the Efficiency Prize board as the board prints it. Ties leave gaps. Rows are in rank order, not score order. |
| `team_key` | text |  |  | Pseudonymous team key: "t" plus the first 10 hex digits of the SHA-256 of the trimmed team name. Blank where the board prints no name. Names are not published. |
| `PublicScore` | decimal (3 dp) | score | 0.447 to 0.964 observed | Public leaderboard score of the team entry, as the board prints it, shown to three decimals. Not the efficiency metric. The board's metric is not verified [P]. |
| `DateSubmitted` | text | local date-time as printed |  | Submission timestamp as the board prints it, kept as printed. The board states no time zone. |
| `DateSubmittedISO` | text | ISO 8601, no time zone |  | DateSubmitted in ISO form (YYYY-MM-DDTHH:MM:SS). No time zone is implied. |

### `leaderboard_top3000_20261008.csv` (3,000 rows, 4 columns)

| Column | Type | Unit | Allowed values | Meaning |
|---|---|---|---|---|
| `LeaderboardPosition` | integer | position | 1 to 3000 | Position in the captured order of the main leaderboard (API order, highest score first). |
| `team_key` | text |  |  | Pseudonymous team key, computed as in the efficiency board files. Two entries share a key where the trimmed names are equal. |
| `PublicScore` | decimal (3 dp) | score |  | Public leaderboard score, three decimals. |
| `SubmissionDate` | text | ISO 8601 UTC |  | The submission date the API returns for the team's entry. Its exact meaning is not documented. |

Notes on two columns:

- `EfficiencyRank` ties: rank values are shared on each board file (see Limits). The rank after a tie skips, as printed.
- `PublicScore` on the efficiency board is the public leaderboard score of the team entry. It is not the board's efficiency metric, which could not be verified (section "What the figures show and do not show", item 6).

## How to load

```python
import pandas as pd

nb = pd.read_csv("public_notebooks.csv")             # 24 public notebooks, one row each
links = pd.read_csv("notebook_assets.csv")           # which notebook attaches which dataset
assets = pd.read_csv("assets.csv")                   # 26 datasets: licence, size, counts
board = pd.read_csv("efficiency_board_20261010.csv") # Efficiency Prize board, 10 Oct 2026

print(nb.groupby("displayed_score")["notebook_ref"].count())
```

Efficiency statistics (the cut-offs are as stated):

```python
import pandas as pd

eff = pd.read_csv("efficiency_board_20261010.csv")
top100 = eff[eff["EfficiencyRank"] <= 100]               # rank cut-off: 1 to 100
print("top-100 median PublicScore:", top100["PublicScore"].median())
print("entries at or above 0.950:", (eff["PublicScore"] >= 0.950).sum())
```

Which notebooks attach a given dataset, with its licence:

```python
import pandas as pd

links = pd.read_csv("notebook_assets.csv")
assets = pd.read_csv("assets.csv")
merged = links.merge(assets[["asset_ref", "licence_normalised"]], on="asset_ref", how="left")
print(merged.groupby("licence_normalised", dropna=False)["notebook_ref"].nunique())
```

## Curated frontier table: the 24 public notebooks

| Row | Notebook | Author | Scorecard rank | Displayed score | Method family | Assets mounted | Asset licences | Notebook licence | 10 Oct list rank |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `evgendvorkin/rsna-versia-5` | evgendvorkin | 1 | 0.950 (V, v19) | Shared blend: nartaa SWA mirror reader plus goodpjw ConvNeXt, per-label rank blend (weights credited to another notebook) | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | Apache-2.0 | not on list |
| 2 | `matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed` | matterhorn3838 | 2 | 0.950 (V, v7) | Shared blend: nartaa SWA mirror reader plus goodpjw ConvNeXt, per-label rank blend | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | Apache-2.0 | 28 |
| 3 | `matterhorn3838/rsna-knee-d4` | matterhorn3838 | 3 | 0.950 (V, v5) | Same code as row 2 | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | not read | 9 |
| 4 | `sujanmajhisuzan/rsna-knee-apex-grandmaster-stack` | sujanmajhisuzan | 4 | 0.950 (V, v1) | Same fusion cell as row 2 | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | not read | 19 |
| 5 | `haideptry/rsna-knee-v2-vs-apex-comparative-study` | haideptry | 5 | 0.950 (V, v5) | Three tiers: hard-coded weights, or a per-label grid on the gold58 studies when the package is mounted | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007`, `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` | Apache-2.0, CC0-1.0, other | not read | 12 |
| 6 | `kozykappa/rsna-knee-gold-gated-triple-reader` | kozykappa | 6 | 0.950 (V, v2) | Own CoAtNet-384 engine on the nartaa checkpoint, mirror TTA, ConvNeXt at one global weight chosen through a gold58 gate | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | not read | 13 |
| 7 | `hitarthjain0/rsna-knee-apex-grandmaster-stack-v5` | hitarthjain0 | 7 | 0.950 (V, v2) | Shared blend with higher per-label weights; ConvNeXt at 256 and 320 px with flip | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | not read | 14 |
| 8 | `rabari9999/rsna-knee-apex-grandmaster-stack-v5-lb-0-95` | rabari9999 | 8 | 0.950 (V, v1) | Copy of row 5 (gold58 grid) | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007`, `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` (partial list) | Apache-2.0, CC0-1.0, other | not read | not on list |
| 9 | `ranjeet258/rsna-knee-apex-b-under-150` | ranjeet258 | 9 | 0.950 (V, v1) | nartaa 384 plus nartaa 224 crop arm (weight 0.3), then per-label ConvNeXt weights | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | not read | 16 |
| 10 | `prvsiyan/the-bee-s-knees-final-rsna-push` | prvsiyan | 10 | 0.949 (V, v34) | nartaa 384 SWA mirror reader (version v34; no ConvNeXt arm) | not itemised | - | not read | 33 |
| 11 | `nartaa/rsna-knee-0949-anatomical-mirror` | nartaa | 11 | 0.949 (V, v2) | Single 384 SWA checkpoint with anatomical-mirror TTA, no blend | `nartaa/rsna-knee-publication-swa-weights-20261007` | other | Apache-2.0 | 36 |
| 12 | `xianhan/rsna-knee-0949-anatomical-mirror-candidate` | xianhan | 12 | 0.949 (V, v1) | Same code as row 11 | `nartaa/rsna-knee-publication-swa-weights-20261007` | other | not read | 33 |
| 13 | `goodpjw2008/rsna-knee-0-949-coatnet-stack-blend-lb-0-949` | goodpjw2008 | 13 | 0.949 (V, v4) | Stack (weight 0.40) and nartaa reader (weight 0.60), rank blend | `nartaa/rsna-knee-publication-swa-weights-20261007` (partial list) | other | Apache-2.0 | 34 |
| 14 | `sujanmajhisuzan/rsna-knee-apex-grandmaster-v5` | sujanmajhisuzan | 14 | 0.949 (V, v1) | Higher per-label weights (mean 0.20); fallback tier at 0.949 | `goodpjw2008/rsna-knee-2-5d-convnext-reader`, `nartaa/rsna-knee-publication-swa-weights-20261007` | Apache-2.0, other | not read | 35 |
| 15 | `pjmathematician/rsna-knee-d4-blend` | pjmathematician | 15 | 0.946 (V, v3) | Public core (four CoAtNets, DINO-20, A5, RadImageNet heads, Raptor) plus private student models at weight 0.45 | `marwanmath/resnet-50-radimagenet-marwan`, `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812` (partial list) | CC-BY-NC-SA-4.0, other | not read | 45 |
| 16 | `pjmathematician/rsna-knee-d4-lite` | pjmathematician | 16 | 0.945 (V) | Not opened | not opened | - | not read | 47 |
| 17 | `aastikrajan15/knee-s75-w50` | aastikrajan15 | 17 | 0.945 (V) | Not opened | not opened | - | not read | 48 |
| 18 | `aastikrajan15/knee-s75-w60` | aastikrajan15 | 18 | 0.945 (V) | Not opened | not opened | - | not read | 49 |
| 19 | `sujanmajhisuzan/rsna-knee-tri-specialist-superstack` | sujanmajhisuzan | 19 | 0.945 (V) | Not opened | not opened | - | not read | 50 |
| 20 | `nartaa/rsna-knee-0945-efficient-224crop` | nartaa | 20 | 0.945 (V) | 224-pixel crop SWA checkpoint, K94, one view, no blend | `nartaa/rsna-knee-publication-swa-weights-20261007` (partial list) | other | Apache-2.0 | 51 |
| 21 | `bliverrigpdoge/rsna-knee-raptor-student-blend-lb-0-944` | bliverrigpdoge | not on scorecard | 0.944 (P) | community stack plus a Raptor student; rank blend 0.55 / 0.45 (claim) | `bliverrigpdoge/rsna-knee-leg-r9all` (partial list) | CC0-1.0 | not read | 52 |
| 22 | `goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944` | goodpjw2008 | not on scorecard | 0.944 (P, v5) | community stack at 0.70 plus the goodpjw 2.5D ConvNeXt reader at 0.30 (claim on title) | `goodpjw2008/rsna-knee-2-5d-convnext-reader` (partial list) | Apache-2.0 | Apache-2.0 | not on list |
| 23 | `medvax/rsna-knee-independent-2-5d-reader-trial` | medvax | not on scorecard | not recorded | Reproduction of the goodpjw 2.5D ConvNeXt reader; no score recorded | `goodpjw2008/rsna-knee-2-5d-convnext-reader` (partial list) | Apache-2.0 | not read | 53 |
| 24 | `xianhan/rsna-knee-fast-parent-0-957-candidate` | xianhan | not on scorecard | 0.957 (P) | DINO-20 ensemble, A5, RadImageNet stage, CoAtNet and Raptor views; outer CoAtNet weight 0.60 | `pilkwang/rsna-knee-weights`, `metaresearch/dinov2` (model), `marwanmath/resnet-50-radimagenet-marwan`, `prvsiyan/rsna-knee-v52-radimagenet-heads-20260812`, `dreaddevelopment/raptor-knee-maxspan`, `dreaddevelopment/raptor-knee-native384`, `dreaddevelopment/raptor-knee-native384dense` | CC-BY-NC-SA-4.0, CC0-1.0, not in assets.csv, other | not read | not on list |

How to read the table:

- Order: scorecard rank for rows 1 to 20, then the rest in the order of `public_notebooks.csv`.
- Displayed score: the score on the notebook's code page, with its label. `V` means rendered signed out on 8 October 2026. `P` means another author's claim (a title, the survey listing, or the notebook's own text), not the render. Rows 21 to 24 are not on the scorecard.
- Method family: a short label written by the publisher from the source as read, or the claim where marked. "Not opened" means the notebook was listed but not read.
- Assets mounted: the datasets and models in `notebook_assets.csv`. "(partial list)" means the notebook attaches more assets than the table lists; "(not itemised)" means the same with no list of its own. "Not itemised" alone means no asset of the notebook is listed. "Not opened" means the notebook was not read.
- Asset licences: the normalised licences of those assets from `assets.csv`. The asset row marked "not in assets.csv" is a model the table does not describe.
- Notebook licence: the licence the notebook's page shows, where read. "not read" means it was not read; it is not a claim that the licence is absent.
- 10 Oct list rank: the rank on the 10 October 2026 kernel list (captured 11:35 UTC). "not on list" does not mean the notebook is absent from Kaggle.

## Headline facts

- Displayed scores on the scorecard (top 20 by score, 8 October 2026): nine notebooks show 0.950 (rows 1 to 9), five show 0.949 (rows 10 to 14, including the publisher's notebook), one shows 0.946 (row 15) and five show 0.945 (rows 16 to 20) [V, scorecard]. Rows 21 and 22 show 0.944 from their titles or the kernel-list listing [P]. Row 24's title reads 0.957 [P]. Row 23 shows no score.
- All nine 0.950 notebooks attach the nartaa checkpoint dataset and goodpjw2008's ConvNeXt reader (`notebook_assets.csv`). Where a mean ConvNeXt weight is recorded for them, it is 0.16 to 0.22 [V, source reads]. Row 5 attaches the gold58 package as a dataset and row 8 does so per the survey text [V and I].
- On the leaderboard capture (first 3,000 rows, 8 October 2026): the top score is 0.964; 151 entries are above 0.950; 285 sit at exactly 0.950; 97 at exactly 0.949. The 10th is at 0.961, the 50th at 0.956 and the 100th at 0.953 [V, recomputed from `leaderboard_top3000_20261008.csv`].
- The publisher's team entry is at position 443 with public score 0.949. Its SubmissionDate, 2026-10-08T13:26:42.273Z, is the submission time of row 56955785, whose snapshot public score is 0.948. The 0.949 score belongs to rows 56948803, 56949940 and 56952077 (each 0.949 in the snapshot), and none of those has that time [V]. The 10 October efficiency board gives the team's 0.949 entry the time 2026-10-08T10:20:42 (rank 176), which is the time of row 56948803 [V]. So the leaderboard capture pairs a date from one row with a score from others; this card treats row 56948803 as the 0.949 entry the 10 October board shows, and does not resolve the leaderboard capture further.
- Efficiency board, top 13 by rank: PublicScore 0.952 to 0.962 on 8 October, 0.945 to 0.962 on 9 October, 0.947 to 0.960 on 10 October [V].
- Efficiency board, median PublicScore of the top 100 by rank: 0.9415 on 8 October, 0.948 on 9 October, 0.949 on 10 October [V].
- Entries at or above 0.950 on the board: 111 on the 8 October file, 116 on the 9 October file, 433 on the 10 October file. Of the 433, 339 are dated 8 October [V]. See "What the figures show and do not show", item 7.

## Snapshot times (UTC)

- Notebook scorecard: rendered signed out on 8 October 2026 at about 15:03. It shows the top 20 by score.
- Kernel list: 8 October 2026 at 14:42. The source of the votes for rows 21 to 23. The 10 October kernel list used in the frontier table was captured at 11:35 UTC on 10 October 2026.
- Leaderboard: the first 3,000 rows the API returns, saved 8 October 2026 at 14:45.
- Dataset metadata: saved between 15:07 and 18:22 UTC on 8 October, the range of `metadata_captured_utc` in `assets.csv`, which gives the time for each asset.
- Team submission rows: the submission snapshot of 10 October 2026, 11:47 UTC.
- Efficiency board, one line per file:
  - 8 October file: newest DateSubmitted 6 October 2026 21:48:51. File saved 15:44 UTC.
  - 9 October file: newest DateSubmitted 7 October 2026 21:35:51. File saved 18:00 UTC.
  - 10 October file: newest DateSubmitted 8 October 2026 20:59:13. File saved 10:23 UTC on 10 October.
  - The notebook's own run time is not recorded in the saved files.

## What the figures show and do not show

1. **The public frontier is one rounded tick above the publisher's displayed score, and its rows share parts.** The nine 0.950 notebooks share the nartaa checkpoint and goodpjw2008's ConvNeXt reader, and where a mean ConvNeXt weight is recorded it is 0.16 to 0.22 [V]. The publisher's version v35 blends the mirror reader and the ConvNeXt reader at 0.70 and 0.30 and scored 0.948 (official row 56957785) [V]. Its displayed version, v34, scored 0.949 (official row 56949940) [V]. A one-tick difference does not show that the blend helps or hurts.
2. **The weights are not validated optima.** In the source texts the weights are per-label values copied between notebooks, a single value chosen by a gate on the gold58 package, or hard-coded tiers [V]. Forum participants say the gold58 package cannot resolve small score differences [P]. A value read off the public board and then described as chosen in advance is not a validation.
3. **A weaker second reader: one author probe, not reproduced.** The notebook in row 13 reports that a stack at 40 percent over the nartaa reader leaves the displayed score unchanged [P]. The author's claim is not reproduced here.
4. **The top band depends on an eligibility question that is still open.** Every notebook in rows 1 to 14 uses the nartaa checkpoint. Its dataset description says it was trained with masked supervision on knees from the Osteoarthritis Initiative (OAI), that OAI access is governed by NDA terms, and that the release grants no rights to that data [V, dataset text]. On 21 September 2026 the host replied in forum thread 741819 that using OAI "does not violate the RSNA Knee Abnormality Detection challenge rules" if the data is free and generally accessible without institutional review. The host added that it cannot comment on OAI's own data-use terms [V]. Participants read that reply differently [P]. Two follow-up threads were unanswered on 8 October: 743416 (OAI-trained checkpoints) and 744056 (prize eligibility when using public weights another team trained) [V]. No ruling has been made about any named notebook. Treat rows 1 to 14 as eligibility-dependent until the host answers.
5. **The methods of the entries above 0.950 are not visible here.** Forum posts describe label work: LLM readings of the reports checked against the host's image-based definitions, explicit handling of cells the report does not address, soft labels, and pseudo-labels from out-of-fold predictions. Some posts also use external OAI data [P]. Forum authors describe private single models at 224 to 384 pixels [P]. Nothing in this dataset tests those claims.
6. **Rank on the efficiency board is not score order.** On each file, some adjacent ranks are out of score order: the top 21 contain 8, 8 and 9 such adjacent pairs on the three files [V]. An inversion means the entry at rank r has a lower PublicScore than the entry at rank r+1. The board's efficiency metric is not documented in the saved files and was not reproduced, so treat it as unverified [P].
7. **The board changed sharply on 10 October.** Entries at or above 0.950: 111 on the 8 October file, 116 on the 9 October file and 433 on the 10 October file. Of the 433, 339 are dated 8 October. Of the other 94, 70 match an entry on the 9 October file with the same team key and submission time (50 of them are also on the 8 October file). The other 24 match no entry on the 8 or 9 October file, but each of their team keys does appear on one of those files, so they are new submissions by teams already on the board. Of the 116 entries at or above 0.950 on the 9 October file, 70 are still on the 10 October file at the same score; the other 46 are not on it under the same team key and submission time. The 10 October file also has 297 entries at exactly 0.950 and 149 at exactly 0.949. Whether the new 8 October entries copy public notebooks was not checked [I].
8. **The team's own board entry moved.** On the 8 and 9 October files the team is at rank 3,666 and 3,678 with 0.943, from its 5 October submission (row 56859861). On the 10 October file it is at rank 176 with 0.949, printed as "Thu Oct  8 10:20:42 2026", which is the submission time of row 56948803. The printed time matches the submission time, which suggests the board prints UTC. The board states no zone, so this match is an inference [I]. The reason the board switched rows is not established [I].
9. **A displayed score can belong to an earlier version.** Row 10 shows 0.949 from version v34. Version v35 later scored 0.948 (official row 56957785) [V]. Check the `version` column before reading a displayed score as the notebook's current head.
10. **A title claim is not a score.** Row 24 is titled "RSNA Knee Fast Parent 0.957 Candidate" [V]. Its source sets a diagnostic value and routes each finding with weights tuned on the public split [V, source read]. Its displayed score was not checked on a signed-out render.

## Limits

- Scores are rounded to three decimals. A difference of 0.001 is not evidence of anything here.
- This is a snapshot. Versions, votes and scores change daily.
- The leaderboard capture stops at 3,000 rows, the API limit. The true number of teams is higher.
- Licences were read for six notebooks: rows 1, 2, 11, 13, 20 and 22. Rows 11 and 20 were checked on their Kaggle pages on 10 October 2026. The others are as shown on the Kaggle page on 8 October 2026 (not independently re-verified). Every other notebook shows "not read" in `notebook_licence`. Dataset licences are as the platform reports them.
- Method fields describe the source as read. No notebook was run.
- Rows 16 to 19 were listed from the scorecard only and not opened. Rows 21 to 24 rely on titles, the listing review or source text, as their `score_label` and notes say.
- Platform counts in `assets.csv` (`api_kernel_count`, `api_download_count`) change between captures. Two assets are known to differ between captures: the nartaa checkpoint dataset and pilkwang's LLM label dataset. The table uses the capture named in `metadata_captured_utc`.
- EfficiencyRank ties: each board file has rank values that two or more rows share. The 8 October file has 36 such values covering 126 rows, the 9 October file 37 values covering 128 rows, and the 10 October file 32 values covering 114 rows. After a tie the next rank skips (on the 10 October file, two rows at 351 are followed by 353). Rows are in board order and the order within a tie is kept as printed. Sort by PublicScore if you need a score order.
- The competition rules page could not be read signed out on 10 October 2026. The licence basis is therefore stated (see Licence) and was not checked against the rules text.

## Evidence labels and terms

- **V**: checked against the source text, the platform's API metadata, or a rendered public page on the capture date.
- **P**: another author's claim (a title, a model card, a forum post, a notebook's own table). Not reproduced here.
- **I**: the publisher's inference.

Terms used in the table and the text:

- **survey**: the publisher's review of the public notebook and dataset listings and their text, during the capture window.
- **scorecard**: Kaggle's public notebook list sorted by score, top 20, as rendered signed out on 8 October 2026.
- **gold58**: the name of a hand-labelled study package that several notebooks use to tune blend weights. The name is the package's own; this card does not count its studies.
- **OAI**: the Osteoarthritis Initiative, a research cohort whose data are governed by data-use agreements.
- **Anatomical mirror**: test-time augmentation that averages a prediction on a study with the prediction on its anatomically mirrored version [V code].
- **SWA**: stochastic weight averaging, a training method that averages checkpoints saved late in training.
- **K94**: the name the source gives an inference setting; the meaning is as the author states it [P].
- **Rank blend**: combining two models' outputs by averaging their ranks rather than their probabilities.
- **Mirror-only**: a reader with no blend, only the anatomical mirror test-time augmentation.
- **RadImageNet**: a radiology pretraining dataset. The encoder derived from it is under CC BY-NC-SA 4.0 in the assets table.
- **LB**: leaderboard.
- **vN**: Kaggle notebook version N.
- **Architecture and model names** (CoAtNet, ConvNeXt, DINOv2, DINO-20, A5, Raptor, ResNet-50): names the sources use. They are not defined here.

## Team names and keys

The efficiency board and the leaderboard print team names. This dataset does not publish them. `team_key` is "t" followed by the first ten hexadecimal digits of the SHA-256 hash of the team name, with surrounding spaces removed.

- The key is stable across files, so a team can be followed from day to day.
- It is pseudonymous, not anonymous. The names are public on the Kaggle board, so anyone can recompute a key from a name.
- Names are trimmed of surrounding spaces before hashing. Each trimmed name appears once per efficiency board file.
- Blank team names on the board (rank 3,709 on 8 October, 3,724 on 9 October, 591 on 10 October) have no key.
- In the leaderboard capture, one name appears twice; both entries share one key.
- The efficiency board files are no longer byte-identical to the notebook output. The name column was replaced, scores were printed to three decimals, and the ISO time column was added. The SHA-256 values of the unmodified notebook outputs are: 8 October `165323d792c2767de26cd03e581a7282e3aecbe8da5afdc648aafe3e53b65330`, 9 October `fa5417e979b0f5ccc7b628b78ca7ac99e93676285691c657d4e490ab2945fa87`, 10 October `4bfd388bc6b8fdd0924300d6e6bf29e0a7338b4f2ff3ed0cf167a5ef5ad0552d`.

## Team submissions

The publisher's notebook is `prvsiyan/the-bee-s-knees-final-rsna-push` (row 10). Public notebook scores have no submission id: they are the values the code page shows. The rows below are the publisher's Kaggle submissions that statements in this card rest on, and are not a full list of its submissions. Row ids, submission times (UTC) and public scores come from the submission snapshot of 10 October 2026. The description column is an excerpt of the platform's own description, with build labels and private notebook references removed.

| Row id | Submitted (UTC) | Public score | Snapshot description (excerpt; build labels removed) |
|---|---|---|---|
| 56859861 | 5 Oct 2026, 18:57:33 | 0.943 | "unchanged previously scored .943 source; no new method claim" |
| 56948803 | 8 Oct 2026, 10:20:42 | 0.949 | "fault-tolerant anatomical mirror" |
| 56949940 | 8 Oct 2026, 10:48:43 | 0.949 | "fault-tolerant anatomical-mirror reader (nartaa recipe, Apache code + weights under their terms)" |
| 56952077 | 8 Oct 2026, 11:43:41 | 0.949 | "mirror r384 pair + permitted r224 single-view arm, predeclared ordinal rank blend 0.75/0.25" |
| 56955785 | 8 Oct 2026, 13:26:42 | 0.948 | "mirror r384 pair + goodpjw2008 three-fold ConvNeXt 2.5D reader (Apache-2.0), predeclared global ordinal rank blend 0.70/0.30" |
| 56957785 | 8 Oct 2026, 14:19:54 | 0.948 | "fault-tolerant anatomical-mirror reader (nartaa recipe) + goodpjw2008 three-fold ConvNeXt 2.5D reader (Apache-2.0), predeclared global ordinal rank blend 0.70/0.30" |
| 56923965 | 7 Oct 2026, 23:22:21 | no public score recorded | Platform error text: "Your notebook hit an unhandled error while rerunning your code. Note that the hidden dataset can be larger/smaller/different than the public dataset" |

The descriptions of rows 56949940 and 56957785 begin with a build label, which is removed above. The two labels name versions v34 and v35 of the publisher's public notebook. Version v34 is the displayed version (row 10 of `public_notebooks.csv`). The 10 October efficiency board's entry for the team (rank 176, 0.949, 2026-10-08T10:20:42) matches row 56948803.

## Licence

- **Dataset licence (the Kaggle licence field): CC-BY-4.0**, Creative Commons Attribution 4.0 International. Legal code: https://creativecommons.org/licenses/by/4.0/legalcode. See `LICENSE` for the full terms and `NOTICE.md` for per-file terms.
- **Two attribution licences, two kinds of file.** The compiled tables are offered under CC-BY-4.0. The example code blocks in the "How to load" section of this README are the publisher's own code and are offered under the Apache License, Version 2.0. `NOTICE.md` states this per file. The two licences may coexist in this dataset.
- **Basis.** The tables contain public listing facts only: notebook and dataset titles, displayed scores, votes, leaderboard positions, dataset licence fields as the platform shows them, and the publisher's own reading of public notebook code and text. They also contain the publisher's own submission rows and public scores. They contain no competition data: no images, labels, studies, model weights or predictions. CC-BY-4.0 is the licence for that compilation. It is not a claim about the competition's own data, which this dataset does not contain. The competition rules page could not be read signed out on 10 October 2026, so this basis is stated rather than checked against the rules.
- **Per-file terms.**
  - The compiled tables (`public_notebooks.csv`, `notebook_assets.csv`, `assets.csv`, `leaderboard_top3000_20261008.csv`), `schema.json`, this card, `dataset-metadata.json`, `LICENSE`, `NOTICE.md` and `MANIFEST.json`: CC-BY-4.0.
  - `efficiency_board_20261008.csv`, `efficiency_board_20261009.csv`, `efficiency_board_20261010.csv`: CC-BY-4.0 for the compilation. The team names were replaced by keys, scores were printed to three decimals, and an ISO time column was added. The source notebook `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` is licensed under Apache License 2.0 as shown on the Kaggle page on 8 October 2026 (not independently re-verified). That licence covers the notebook's code, which is not in this dataset.
  - Notebook licences (`notebook_licence` in `public_notebooks.csv`): as shown on each notebook's Kaggle page. Rows 11 and 20 (nartaa) are Apache-2.0 as checked on their Kaggle pages on 10 October 2026. The other values are as shown on the Kaggle page on 8 October 2026 (not independently re-verified). Notebook code is not in this dataset.
  - The publisher's own submission rows and public scores: CC-BY-4.0, with the rest of the compilation.
  - Asset licences (`licence_as_listed`, `licence_normalised`) are reported as the platform lists them. No asset is redistributed.
- Kaggle records one licence field for this dataset, CC-BY-4.0. The per-file terms above, in `LICENSE` and in `NOTICE.md` state the rest.
- The competition rules page was not read signed out. If the rules text bars publishing leaderboard-derived tables, the licence would have to be revisited.

## Credits

- Compiled and published by prvsiyan.
- Notebooks and datasets are credited to their owners in the `owner` columns (and `notebook_ref` and `asset_ref`). Owners are Kaggle usernames.
- Notebook rows 11 and 20 describe two notebooks by nartaa (Apache-2.0); attribution to nartaa (Danial Zakaria) and the notebook links is given in `NOTICE.md`.
- Competition: RSNA Knee Abnormality Detection, `rsna-knee-abnormality-detection` (https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

### Dependency and component credits

No file in this dataset imports a third-party package except the example code in this README, which imports pandas. The table below lists that package and the notebooks, datasets and models the measurements in this card depend on, with the licence each is recorded under. Where a licence is shown on a Kaggle page or as the platform lists it, the table says so. Other attached assets are listed with their licences as the platform shows them in `assets.csv`; the table also includes the one attached model that is not in that file.

| Component | Used for | Licence |
|---|---|---|
| pandas (Python package) | Example code in the "How to load" section | BSD-3-Clause |
| `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` (Kaggle notebook) | Source of the three efficiency board files | Apache-2.0, as shown on the Kaggle page on 8 October 2026 (not independently re-verified) |
| `nartaa/rsna-knee-publication-swa-weights-20261007` (Kaggle dataset) | Mounted by the 0.950 and 0.949 notebooks and by row 20 | Other (specified in description), as listed by the platform |
| `goodpjw2008/rsna-knee-2-5d-convnext-reader` (Kaggle dataset) | ConvNeXt reader in the rank blends of the 0.950 family | Apache-2.0, as listed by the platform |
| `marwanmath/resnet-50-radimagenet-marwan` (Kaggle dataset) | RadImageNet-derived encoder attached by rows 15 and 24 | CC-BY-NC-SA-4.0, as listed by the platform |
| `metaresearch/dinov2` (Kaggle model) | Attached by row 24 | Not recorded in this dataset |

## How to cite

prvsiyan, "RSNA Knee 2026: public notebook scores and assets", Kaggle dataset `prvsiyan/rsna-knee-2026-public-notebook-scores-and-assets`, version 1, snapshot 8 to 10 October 2026. When you use a row, cite the notebook or dataset slug from `notebook_ref` or `asset_ref`, together with its owner.

## Changelog

- Version 1, October 2026: first release. Notebook and dataset captures from 8 October 2026; efficiency boards saved on 8, 9 and 10 October 2026; leaderboard capture from 8 October 2026; team submission rows from the submission snapshot of 10 October 2026.
