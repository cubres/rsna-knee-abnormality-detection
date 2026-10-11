# Connect a model licence to the actual file

A public model can have several names, mirrors and conversions. A licence badge becomes useful when it can be connected to the exact artifact being used.

For each dependency, record:

1. The downstream file's declared SHA256, its byte size and the manifest that names it.
2. The upstream publisher, immutable revision, file path and published content hash.
3. The publisher's explicit model terms, separately from the software licence and training-data access terms.
4. Permission to use the artifact in the intended competition, separately from permission to redistribute it.
5. A final payload hash check in the authorized execution environment. Metadata comparison alone does not prove the mounted bytes.

A concrete example is the [S6 model manifest](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-v1/versions/13). Its declared anchor SHA256 matches the published hash of the original SKM-TEA author's V-Net E1 checkpoint in [arjundd/skm-tea-models revision 8b8b105835294fc966a912049d2ace6d0fee1068](https://huggingface.co/arjundd/skm-tea-models/tree/8b8b105835294fc966a912049d2ace6d0fee1068/neurips2021/segmentation-models/V-Net_E1). The [revision-pinned model README](https://huggingface.co/arjundd/skm-tea-models/blob/8b8b105835294fc966a912049d2ace6d0fee1068/README.md) declares Apache 2.0. This is a more precise connection than applying the GitHub software's MIT licence to an unidentified checkpoint.

That observation does not qualify the entire downstream bundle. Its initialization hash is unspecified, and a blending invitation is not a complete redistribution licence. The checkpoint publisher's terms also do not settle the competition host's treatment of historical training data. Those remaining questions need their own evidence.

This checklist is original prose. The example compares published metadata only: no weights were downloaded, loaded or executed, and no competition records or derived diagnostic results are included. It makes no score or native-runtime claim and does not authorize new checkpoint adoption.
