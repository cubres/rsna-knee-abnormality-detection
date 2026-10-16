# Public update rejected before the canary

On 16 October 2026 at 11:49 UTC, Kaggle rejected one identity-preserving update of [Bee’s Knees](https://www.kaggle.com/code/prvsiyan/the-bee-s-knees-final-rsna-push?scriptVersionId=358804172):

> Notebooks with competition rsna-knee-abnormality-detection as a source may not be made public until after the competition ends

The response returned version 0 and no URL. A subsequent read-only source/status audit confirmed that the notebook remains public at V47, with the exact prior source hash and COMPLETE runtime. No new notebook version, GPU job or competition submission was created. The attempt is closed and will not be retried as recovery.

Both previously accepted versions 46 and 47 retain official public scores of **0.957**. See the [verified official results](../2026-10-16-v46-v47-official-results/README.md).

A faithful reproduction of Heliosli’s public 0.958 recipe had passed local source, preservation, transport and output-checker review. It remains staged, without a native canary or an owned 0.958 result. No notebook identity, artwork or visibility was changed.

The [official timeline](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview) still gives 22 October 23:59 UTC as the final submission deadline. An enabled submission action does not establish permission to publish a new notebook version. This record preserves the observed rejection; it does not infer an undocumented cause or a general change to all submission eligibility.
