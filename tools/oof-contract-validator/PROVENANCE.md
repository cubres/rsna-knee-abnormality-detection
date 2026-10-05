# Implementation and evidence

This utility, its tests and the synthetic example are original code written for the `cubres` RSNA Knee campaign on 2026-10-05. They do not embed a third-party notebook, held prediction arrays, official labels, teacher outputs, patient IDs or model weights. The MIT license here covers this original software and documentation; it grants no rights to external competition data or checkpoints.

## Verified local observation

The read-only historical adapter joined 58 expert studies and 12 named targets against the retained official `train.csv`. The official CSV's SHA-256 was:

```text
8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33
```

The historical B3 CSV had 0/696 attached-label mismatches. The NPZ's attached labels had 145/696 mismatches under the exact target order declared by its historical consumer. Producer label order and the DINO generating-fold identity were not independently attested. The NPZ was read with `allow_pickle=False`; its study IDs were Unicode arrays.

All five retained B3 checkpoint hash pins matched. Expert fold counts were 12/11/12/11/12. Across the declared 4,407-study, 4,257-report-group split, the validator found zero train/held study overlaps and zero report-group overlaps in every fold. That observation does not establish patient independence or full runtime provenance.

The historical adapter returned **HOLD**, with 15 errors and one warning. Reasons included the 145 label disagreements, missing DINO generating folds/source/state/input closure, unpinned helper-module runtime hashes, unresolved historical Kaggle mount paths, unattested initialization/runtime closure, prior use of all five gold folds for blend selection, and unproven equivalence to the current ensemble. Unresolved mount paths mean those declared locations were unavailable locally; they do not prove the data is absent elsewhere.

No model was fitted, no input was corrected, and no official competition score was produced by this audit. Counts here describe local data-contract checks. The private historical adapter and data remain outside this publication packet.

## Exact reviewed artifacts

| Artifact | SHA-256 |
|---|---|
| `oof_contract.py` | `f4c8e5981e429c552b82f619d1f86a7708ba896c2390be9b8e45354400492901` |
| `test_oof_contract.py` | `022be59a7b121fd7b01ec6966dadd2f922ef0fb8d92a50aa1f4e6cb4feca1ab8` |
| Private historical evidence manifest | `39aace9fd5deac60a40794afefa37b0af9618cdbd11a87882483758e5f2a5bdc` |
| Historical selection source, read but not executed | `02b63f9a589d6fc3f6d2bc88254aaec4e835c37a74806966225d69d51638d5b5` |

The 20 unit tests passed in 0.025 seconds using Python 3.12.14 and NumPy 2.2.6. Execution time is a local observation, not a portable performance guarantee. Synthetic example output is a demonstration of the contract interface, not model validation evidence.
