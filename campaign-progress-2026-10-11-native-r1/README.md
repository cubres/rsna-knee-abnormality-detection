# Knee progress: a scored control, a pending blend check, and an honest training hold

This note records the `prvsiyan` campaign on 11 October 2026. The completed account rows were inspected at 19:11 UTC; the new submission was checked again at 19:32 UTC.

| Official evidence | Exact row | State |
|---|---:|---|
| Best completed result, bench Version 10 | 57043015 | COMPLETE — **0.949** |
| New accepted evaluation, bench Version 11 | 57083743 | UNKNOWN — score absent |

The [scored Version 10](https://www.kaggle.com/code/prvsiyan/zz-gpuchk-knee-bench-2?scriptVersionId=357057303) combines the credited nartaa anatomical reader with goodpjw2008's ConvNeXt reader. The new [Version 11](https://www.kaggle.com/code/prvsiyan/zz-gpuchk-knee-bench-2?scriptVersionId=357444375) tests a predeclared change to the global reader blend. Acceptance establishes that the evaluation was requested; its completed official row is still required. **0.956 is a target, not an achieved result.**

The separate training pilot, `prvsiyan/the-knee-is-not-the-machine-mc4-dual-t4-worker` Version 65, held at its strict missing-slot gate before the first optimizer update. It exported no checkpoint. This is a useful failure: an absent expected acquisition slot and an unreadable supplied acquisition need different evidence and policies. A replacement health policy must be registered as a new experiment, with actual decoding failures still visible. The failed run remains failed.

Once that control is qualified, the next original scientific factor is acquisition-coherent slice sampling. Adjacent channels can otherwise span boundaries between concatenated acquisitions. Restricting each sampled triplet to one acquisition, while preserving the window budget, labels, loss, initialization and training clock, gives a testable question: does channel coherence improve the reader? This proposal has no measured score yet. Prior checkpoint and development exposure must be accounted for before calling a validation panel unseen; selecting finding-specific blends on already inspected labels would not establish generalization.

For a fork, start with the original [optimizer-call witness](https://github.com/cubres/rsna-knee-abnormality-detection/tree/52023b65c4d4d8820c2a63137d79338a626d8448/optimizer-call-witness-2026-10-11), [factor/index binding utility](https://github.com/cubres/rsna-knee-abnormality-detection/tree/00b3911933f8032fb85fb7350d4f477e1d91d27a/native-factor-index-join-2026-10-11), and [checkpoint provenance guide](https://github.com/cubres/rsna-knee-abnormality-detection/tree/fbd1d0d2f144028b7f9a046c6693fd69b310e911/checkpoint-provenance-2026-10-11). Their software checks do not qualify MRI/CUDA execution. The [illustrated notebook companion](https://github.com/cubres/rsna-knee-abnormality-detection/blob/e3c5e5c0986f6b6633d4465da9da0552b22e2ba2/notebook-companion-2026-10-11-r1/COMPANION.md) provides the visual explanation and runnable invented example.

Upstream code and weights retain their own terms and attribution, including nartaa, goodpjw2008, the Raptor/MIL lineage from dreaddevelopment, and the documented per-finding table lineage from sujanmajhisuzan. These progress notes and utilities do not transfer ownership or expand those licenses.
