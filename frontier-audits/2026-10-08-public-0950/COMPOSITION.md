# Knee frontier 0.950: composition of the public notebooks

Author: Claude frontier analyst. Date: 2026-10-08 (UTC), read-only.
Scope: nine public notebooks pulled with kernels_pull (metadata=True) into pulls/20261008T102627Z/. Dataset and kernel metadata read with GetDataset / GetKernel; two small text files downloaded (nartaa README and manifest). No model payload was downloaded. Forum topics read with ShowTopic/ListTopics (untrusted data, summarised).
Evidence labels: [V] verified from source, metadata or a quoted public document; [I] inference; [P] the author's own claim, not checked.

## 1. Summary table

Displayed score is the public scorecard value at 10:30 UTC. Versions are the server currentVersionNumber. Votes are from the kernel metadata.

| # | Notebook (owner/slug) | Ver | Score | Votes | Base code | Added arm(s) | Blend (as in source) | Mirror TTA | Attached sources (public unless noted) | Licence declared in source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed | 7 | 0.950 | 66 | nartaa 0.949 cell, plus a backup CSV write | goodpjw 3-fold ConvNeXt 2.5D | per-label rank blend, 12 hand-set weights (0.03 to 0.35, default 0.15), final pct rank | yes (inherited) | goodpjw2008/rsna-knee-2-5d-convnext-reader; nartaa/rsna-knee-publication-swa-weights-20261007 | none |
| 2 | matterhorn3838/rsna-knee-d4 | 5 | 0.950 | 6 | code identical to #1 (flattened diff empty) | same as #1 | same as #1 | yes | same as #1 | none |
| 3 | sujanmajhisuzan/rsna-knee-apex-grandmaster-stack | 1 | 0.950 | 114 | nartaa cell plus backup write | same as #1 | fusion cell identical to #1 (same 12 weights) | yes | same as #1 | none |
| 4 | haideptry/rsna-knee-v2-vs-apex-comparative-study | 5 | 0.950 | 37 | nartaa | same as #1 | three tiers: tier 2 = #1 weights; tier 3 = hard-coded weights (0.06 to 0.48), replaced by an in-kernel grid search on 58 gold studies if the gold package is found; anti-tie jitter 1e-4 | yes | same as #1, plus mattiaangeli/rsna-knee-coat-resgated-ep10-top3 (CC0; gold58 package) | none |
| 5 | kozykappa/rsna-knee-gold-gated-triple-reader | 2 | 0.950 | 5 | own CoAtNet-384 engine on the nartaa checkpoint (sha verified), K94, mirror | goodpjw ConvNeXt at one global weight 0.22, plus optional crop224 arm (nartaa SWA, sha 3394fd...) | global rank (or Gaussianised rank) blend; weights chosen by a gate on 58 gold studies (bootstrap, thresholds stated in source) | yes | goodpjw2008/rsna-knee-2-5d-convnext-reader; nartaa/rsna-knee-publication-swa-weights-20261007 | none |
| 6 | hitarthjain0/rsna-knee-apex-grandmaster-stack | 1 | 0.950 | null | nartaa cell plus backup write | same as #1 | logit/rank hybrid: 0.70 * expit((1-w) logit(p_sota) + w logit(p_own)) + 0.30 * ((1-w) rank_sota + w rank_own), weights about +0.03 above #1; min-max rescale per column | yes | same as #1 | none (header claims "0.953+") |
| 7 | nartaa/rsna-knee-0949-anatomical-mirror | 2 | 0.949 | 54 | the base recipe | none (one arm, w = 1.0) | none | yes: plain + anatomical-mirror probability average, 320 crop, K94 | nartaa/rsna-knee-publication-swa-weights-20261007 | none |
| 8 | xianhan/rsna-knee-0949-anatomical-mirror-candidate | 1 | 0.949 | 3 | byte-identical notebook source to #7 (flattened diff 0 lines) | none | none | yes | same as #7 | none |
| 9 | pjmathematician/rsna-knee-d4-blend (comparator, 0.946) | 3 | 0.946 | 261 | public 0.943-family core: DINO-20, A5, Rad, Raptor, four CoAt | "OUR LEG": private student fleet, two asset sets (eff6, 8 arms; ens14, 43 arms) | global 0.45 rank weight on the fleet (default); second fleet weight 0.0 by default | no (mirror appears once, not used) | many CC0 and other datasets (see section 4); kernel outputs sofiaanjenje/rsna-knee-e11-train, -e13-train; model metaresearch/dinov2 (Apache-2.0 as recorded by others [P]); our own prvsiyan/rsna-knee-v52-radimagenet-heads-20260812 | none; Rad encoder is CC BY-NC-SA 4.0 (marwanmath) [V from prior brief] |

Not pulled by me: nartaa/rsna-knee-0945-efficient-224crop (0.945). The e3 builder pulled it into knee/e3/r224_public_pull_20261008T1025Z/ (cell_04 sha e85d994f matches the publisher; crop 307 to 224, one view, FP16).

Notebook licences: none of the nine sources contains a licence statement; the kernel API in this SDK version returns no licence field. Licence is therefore unverified for every notebook [V]. The Codex audit records Apache-2.0 for the nartaa source; I did not re-read that. Credits are present in all of them.

Coverage of attachments. For the eight required notebooks (six at 0.950, two at 0.949) the complete attachment list is: the competition; goodpjw2008/rsna-knee-2-5d-convnext-reader (Apache-2.0); nartaa/rsna-knee-publication-swa-weights-20261007 (Other); and, in haideptry only, mattiaangeli/rsna-knee-coat-resgated-ep10-top3 (CC0). All of these were checked. The comparator pjmathematician D4 has 23 attachments; only the key ones (section 3 and 4) were checked. All nine kernels: GPU on, internet off, NvidiaTeslaT4 [V].

## 2. What the +0.001 over 0.949 comes from

Evidence chain.

1. The base is identical. In all six 0.950 notebooks, the main nartaa inference cell differs from the 0.949 source only by a two-line backup write (flattened diff). The preamble checksum cell is rewritten for older hashlib and prints a message; that changes no arithmetic. The mirror is therefore already inside the 0.949 base [V].
2. The added learned arm is goodpjw's 2.5D ConvNeXt reader, a three-fold model from the public dataset goodpjw2008/rsna-knee-2-5d-convnext-reader (Apache-2.0, version 1, created 2026-10-05 00:09 UTC) [V]. The goodpjw card states it scores 0.929 on its own and gives 0.944 when rank-blended at 30% into the 0.943 stack [V, quoted from the notebook text].
3. The blend differs across the six notebooks (per-label, hard-coded, gold-tuned, global 0.22 with gates), yet all six show 0.950 on the scorecard. The only code change that all six share is the ConvNeXt blend [V].
4. The displayed score is rounded to three decimals. 0.949 to 0.950 means a true difference of roughly 0 to 0.002. goodpjw calls +0.001 "one leaderboard tick" [V, quoted].
5. nartaa's ladder shows that the mirror step is also +0.001 on identical weights (0.948 to 0.949) [V, author table]. The mirror is not the source of the 0.950.
6. The D4 0.946 notebook has no mirror TTA. Its gain over the 0.943 family comes from the private fleet at 45% [V].

Conclusion [I]: the +0.001 is most plausibly the ConvNeXt rank blend. It appears again at +0.001 on a different base (0.943 to 0.944, goodpjw). Two bases, one tick each, is consistent with a small real effect or with public noise. A single rounded score cannot separate the two.

Our own analogue, for calibration [V]: BK-A plus our ConvNeXt student fleet at 15% (V8) scored 0.943, the same as BK-A alone. Our student fleet alone scored 0.918 (row 56800704, orchestrator brief), weaker than goodpjw's 0.929. A weak reader at 15% gave no tick on our stack.

## 3. Reproducibility from assets we hold or public permissive terms

| Ingredient | Status | Basis |
|---|---|---|
| nartaa 384 SWA checkpoint, sha 7e5315... | Public, version 1 only, created 2026-10-07 19:26:38 UTC; licence "Other (specified in description)"; research and educational use including the RSNA competition is allowed; no rights to OAI images, reports or identifiers | [V] README and manifest downloaded (2 small text files) |
| nartaa 224 SWA checkpoint, sha 3394fd... | Same dataset, public, 0.945 | [V] manifest |
| goodpjw ConvNeXt 3-fold checkpoints + code | Public, Apache-2.0, version 1, created 2026-10-05 | [V] dataset metadata and file listing |
| mattiaangeli gold58 package, Repair-v1 and others | Public, CC0, version 3 | [V] dataset metadata |
| pjmathematician eff6-assets / ens14-assets (D4 private leg) | GetDataset returns 403 Forbidden for both | [V] |
| prvsiyan/rsna-knee-v52-radimagenet-heads-20260812 (ours) | Public, "Other", version 1, attached by 251 kernels (per GetDataset kernel count) and by the D4 notebook | [V] |

Verdict [I]. The nartaa base and the goodpjw reader are both reproducible from public assets with a public-data licence on each. The per-label weights are not a reproducible optimum: the goodpjw card says the stack's blend constants were tuned on the public board and on the 58 gold studies, and haideptry's in-kernel search uses only 58 studies. The D4 0.946 leg is not reproducible from our account.

## 4. Per-notebook notes

**matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed (v7, 0.950).** nartaa base; goodpjw ConvNeXt via the glob of cnxt_v0_fold*.pt (three folds); TARGET_WEIGHTS per label (Baker's 0.35, Contusion 0.30, Medial OA 0.25, ACL 0.25, Medial Meniscus 0.20, MCL 0.20, Lateral Meniscus 0.08, Fracture 0.08, Lateral OA 0.08, PF OA 0.08, Effusion 0.04, Synovitis 0.03; default 0.15). The source comment attributes the weights to per-label gains (+0.036 to +0.071), with no stated data source; treat them as tuned [V for code, I for origin]. Fallback restores the 0.949 backup on any exception. Runtime: none on the card. The scorecard labels it "copied with edits from Mattia Angeli" [P]; the notebook itself has no credit cell.

**matterhorn3838/rsna-knee-d4 (v5, 0.950).** Flattened source identical to #1. The name "D4" is misleading: this is not pjmathematician's D4 blend. Six votes.

**sujanmajhisuzan/rsna-knee-apex-grandmaster-stack (v1, 0.950, 114 votes).** Same fusion cell as #1 (identical TARGET_WEIGHTS). Markdown identical to nartaa's 0.949 notebook. The most-voted copy.

**haideptry/rsna-knee-v2-vs-apex-comparative-study (v5, 0.950, 37 votes).** Three tiers (0.949 backup, verified 0.950 blend, and a tier-3 output). Tier 3 uses hard-coded weights (Baker's 0.48, ACL 0.38, Medial OA 0.35, Contusion 0.34, MCL 0.32, Medial Meniscus 0.22, Effusion 0.06, Synovitis 0.05, others 0.08) and runs a per-label grid over [0, 0.55] on the 58 gold studies from the attached mattiaangeli package, adopting it when the grid's macro-AUC is at least the V3 value. The gate has no mode check, so it also runs on the hidden set when the package is present [V]. The markdown claims "0.952-0.955+" and "~36-40 min" runtime [P]. It also says the hidden set has 2,000 studies; the official page says about 1,300 (brief section 3.1), so the 2,000 figure is wrong.

**kozykappa/rsna-knee-gold-gated-triple-reader (v2, 0.950, 5 votes).** The most careful source. Primary is the nartaa-equivalent CoAtNet-384 with mirror TTA and K94 (sha-verified checkpoint). goodpjw ConvNeXt enters at a global weight of 0.22, stated as "gold-selected" and retained only if the gold gate passes (gain at least 0.003, bootstrap sign share at least 0.85, 20% quantile above -0.004). The optional crop224 arm is gated separately (gain at least 0.002, share at least 0.93, 20% quantile above 0). A clinical co-label prior is present in the code but its strength is 0 in the default plan [V]. Time guard: 8.6 h total, and a crop run is skipped under 2,400 s remaining. The card does not show which arms actually ran on the hidden set.

**hitarthjain0/rsna-knee-apex-grandmaster-stack (v1, 0.950).** Same base as #3. Uses a logit/rank hybrid and weights about 0.03 higher than #1 per label. The header says "0.953+ Push" but the card shows 0.950. Final min-max rescaling per column is monotone and does not change ranks.

**nartaa/rsna-knee-0949-anatomical-mirror (v2, 0.949, 54 votes).** The single-arm reference. ARMS has one entry (the 384 SWA checkpoint, weight 1.0) and a comment that mislabels it as a 5-fold model. Native 384 full-field, 140 mm crop, 320 central crop, K94, mirror TTA with equal plain/mirror probability averaging, then per-finding percentile rank. Runtime: 26.1 min submit-to-score for the accuracy entry and 7.2 min for the 224 entry; the author says both include queueing and are not official runtimes [P, quoted from the author's note and forum post].

**xianhan/rsna-knee-0949-anatomical-mirror-candidate (v1, 0.949, 3 votes).** Byte-identical source to #7. Both score 0.949, which supports the view that identical code gives an identical displayed score.

**pjmathematician/rsna-knee-d4-blend (v3, 0.946, 261 votes, comparator).** Stages 1 to 5 are the public 0.943 family (DINO-20, A5, RadImageNet heads, Raptor, four CoAt). It then adds a private student fleet at a global 0.45 rank weight ("OUR LEG", with a documented claim of about 0.75 to 0.89 rank correlation with the public arms [P]). The fleet assets rsna-knee-eff6-assets and rsna-knee-ens14-assets are not attached in the pulled version and return 403 for us. On the hidden run, the code falls back to public-only if the assets are absent ("SOFT FAIL ... public only"). The 0.946 is therefore not reproducible from our side. Its attachments also include our own prvsiyan Rad heads dataset, and the RadImageNet encoder under CC BY-NC-SA 4.0.

## 5. Dates (ordering, 2026-10-05 to 2026-10-08, UTC)

| Time (UTC) | Event | Basis |
|---|---|---|
| 2026-10-04 | goodpjw V2 0.944 scored (ConvNeXt blend at 30% into 0.943) | [V] goodpjw card text |
| 2026-10-05 00:09 | goodpjw ConvNeXt reader dataset v1 published | [V] GetDataset |
| 2026-10-07 19:26:38 | nartaa SWA weights dataset v1 published (only version) | [V] GetDataset |
| 2026-10-07 19:34:51 | nartaa writeup post on the knee forum (0.949 ladder) | [V] forum post timestamp |
| about 22:30 on 10-07 to 08:30 on 10-08 | 0.950 copies appear (sujan about 12 h before the 10:30 scorecard; haideptry 4 h; kozykappa 6 h; matterhorn 2 h; hitarthjain0 2 h) | [I] from the scorecard "updated" ages supplied in the brief; API has no creation timestamps |

Reading: the 0.950 is a same-week recombination of two public artifacts released two days apart. It came within hours of the nartaa writeup. The nartaa weights dataset has no version after 10-07 19:26 UTC as of 10:31 UTC on 10-08 [V].

## 6. Forum and host statements (untrusted data; paraphrased)

- Host ruling on OAI (thread 741819, Po-Hao Chen, 2026-09-21): using OAI as external data does not violate the rules if it is free and generally accessible without institutional review or legal sign-off. The host disclaims OAI's own data-use terms [V, paraphrase].
- Host ruling on KneeCoT (thread 734109, quoted by others): not permitted because it needs an institutional data agreement [V, quoted by participants].
- Host guidance of 27 August (quoted by a participant in thread 743416): straightforward registration or click-through agreements are generally acceptable; institution-specific approvals are not. The host post itself was not located in the saved set [P].
- Open, no host answer found: whether prize-eligible submissions may use publicly shared weights trained on the competition data by other teams (thread 744056, opened 2026-09-28; follow-up 2026-10-01 directed at the host); the rule 2.5.a.4 versus 2.8.a.1 question on weights trained on no-redistribution data (743420, 2026-09-26, follow-ups 09-29); whether OAI access status is clear (743416, re-asked 2026-10-08); RadImageNet ResNet-50 licence under rule 6.c (745283, 2026-10-02).
- No host statement on blending public notebooks was found.
- Thread 745072 is a CASMI competition thread, excluded.

## 7. Caveats

- Displayed scores are rounded to three decimals.
- The 0.950 cards are one submission per notebook version; I could not see the submission history.
- Runtime claims on cards are author claims; nartaa states its times are not official.
- The two 0.949 notebooks are the same code; they should not be counted as independent evidence.
- BK-A receipts (see NOTES.md) give run times for three visible studies only; hidden-scale cost comes from the efficiency-table model in the brief and the cost constants in our notebook.
