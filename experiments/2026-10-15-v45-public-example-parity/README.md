# V45: exact public-example reproduction of the Helios blend

[Bee V45](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push?scriptVersionId=358775631) completed the three public example studies and produced a CSV byte-identical to [Heliosli's exact V1 reference](https://www.kaggle.com/code/heliosli/this-is-new-version-0-9576-in-one-click?scriptVersionId=358332966). This establishes public-example output parity. The official competition submission, row `57241363`, was accepted from V45's native submission menu and remained **PENDING as of 15 October 2026, 19:01:03 UTC**; no new official score is reported here. The upstream notebook's verified visible badge is 0.957; its title's extra digit is not a verified score.

| Completed public-example measurement | Result |
| --- | --- |
| Output shape | 3 studies, 12 findings; 13 CSV columns including the identifier |
| Comparison | All 36 predictions match; maximum absolute difference 0 |
| Exact CSV | 645 bytes; SHA-256 `1ebfe53cb93ba0b9705724721755b515e64f7703712bc6a1bdddbcd181a7342b` |
| Reported failures | R6: 0; Raptor384: 0; Raptor224: 0; R6 fallback studies: 0 |
| Time to validated output | 463.77 s; Kaggle's complete notebook display was 8m 6s on T4 ×2 |
| Sampled aggregate process-group RSS | 22,933,770,240 bytes (21.36 GiB), against a 28 GiB cap |

RSS is a sampled process measurement, not proof of an unseen instantaneous peak. Three public studies are neither cross-validation nor an estimate of hidden-test accuracy or total runtime. Matching the final ranked CSV also does not establish that every internal probability or installed dependency equals the upstream environment.

The recipe remains `0.500 × rank(R6) + 0.325 × rank(Raptor384) + 0.175 × rank(Raptor224)`. The [existing reproduction guide](https://github.com/cubres/rsna-knee-abnormality-detection/blob/af48d80410909758c7f8188c02afaebcf07942ff/tools/public0957-reproduction/2026-10-15/README.md) explains the models, anatomical reflection, execution order, credits and use conditions. [REPRODUCIBILITY.json](REPRODUCIBILITY.json) records this run's source, input, asset and environment hashes.

The active external inputs were three **public V1 datasets**, checked anonymously before launch: [rostigor's exact R6 V1 mirror](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406/versions/1), [nartaa's two Raptor checkpoints](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007/versions/1), and [Heliosli's auxiliary code, anatomy model and offline dependencies](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004/versions/1). The auxiliary card retains “Private Weights” in its title, but the checked access flag was public. R6 was bound to the historical V1 manifest and verified asset hashes; a dataset name alone is insufficient. The remaining data input was the [competition's original data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).

A useful failure lesson was the worker receipt contract. The upstream two-GPU controller constructs `worker_exit_codes` as a mapping. JSON serializes that as `{"0": 0, "1": 0}`. Our earlier wrapper incorrectly required a list. The repair requires exactly those two string keys and exact integer zero values, rejecting booleans, missing workers and nonzero exits. It preserves the failure gate and changes no prediction math. Validate wrappers against the producer's actual serialized receipt, not an assumed shape.

To compare your own locally retained files, use the [existing standard-library checker](https://github.com/cubres/rsna-knee-abnormality-detection/blob/af48d80410909758c7f8188c02afaebcf07942ff/tools/public0957-reproduction/2026-10-15/TOOLS.md) from the repository root:

```bash
python3 -B tools/public0957-reproduction/2026-10-15/validate_canary_parity.py CANARY.csv REFERENCE.csv
```

The names are placeholders. Independently obtain `REFERENCE.csv` from the exact upstream version above and bind its hash before comparison. The checker checks exact header and identifier order, shape, finite values in `[0,1]`, and all 36 differences against `1e-7`; it prints only aggregate results. Its invented-fixture demo is available with `--self-test`.

Credit Heliosli for the blend/R6 recipe, nartaa for the Raptor checkpoints and inference, dreaddevelopment for Raptor's architecture lineage, varshithakolupuri for the 224-slot contribution, and rostigor for the exact V1 mirror. The [full upstream credits and OAI/NDA acknowledgement](https://github.com/cubres/rsna-knee-abnormality-detection/blob/af48d80410909758c7f8188c02afaebcf07942ff/tools/public0957-reproduction/2026-10-15/README.md#credits-and-use-conditions) remain part of this reproduction. Apache-2.0 applies to the upstream notebook source; the dataset cards use “Other (specified in description)” and retain their own terms. Public accessibility does not itself grant redistribution rights. This record contains links, hashes and aggregate measurements; it rehosts no weights, CSV, identifiers, MRI data, reports or label tables.
