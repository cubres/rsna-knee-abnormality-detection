# Exact public Knee source: a reproducible starting point

The [Kaggle source dataset](https://www.kaggle.com/datasets/prvsiyan/rsna-knee-public-apache-source-20261012), version 1 / dataset 12522677, contains seven unmodified Apache 2.0 modules from [Helios Li’s credited notebook](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters), full LICENSE, NOTICE, README and a hash manifest. All eleven downloaded files matched the admitted bytes on 2026-10-11 at 23:16 UTC. It contains no weights, patient images, labels or prediction files.

Download it into a fresh directory with the Kaggle CLI, or add it as notebook input, then verify the files without importing the upstream code:

```sh
kaggle datasets download -d prvsiyan/rsna-knee-public-apache-source-20261012 -p ./knee-source-v1 --unzip
python verify_source_pack.py ./knee-source-v1
```

Keep LICENSE and NOTICE with reused source. Separately mounted weights and helper datasets retain their own terms.

![How the proposed fixed-half blend preserves score and row identity](https://raw.githubusercontent.com/cubres/rsna-knee-abnormality-detection/9c0ce7514423020f041d96c9afac6ccee4f5de21/raptor-s6-fixed-half-2026-10-12/figures/raptor-s6-fixed-half.svg)

The upstream 0.956 card is linked to V1/run 357284025 and its S6 plus DINOv3 student bundle. The current V2 fourfold S6 bundle differs. Access to these identical source modules does not reproduce that score; the old exact weight version was unavailable at our audit. Our historical three-study S6 canary passed numerical pooling checks in 133.079 seconds, which does not establish full hidden-set runtime or an official score. The current A + S6 canary is pending execution.

The first create-response verifier was stricter than the official optional-ref response contract. A read-only check reconciled the already-created dataset; no creation request was repeated. Its license metadata was corrected in place to Kaggle’s canonical `apache-2.0`, preserving id, title, version and all file bytes.
