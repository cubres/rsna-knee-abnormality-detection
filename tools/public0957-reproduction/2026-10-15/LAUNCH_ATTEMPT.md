# One rejected publication attempt, with identity preserved

On 15 October 2026, the single attempt to publish the reviewed reproduction through the existing notebook update path returned HTTP 400. The response body was not retained, so the precise rejection reason is unknown. We did not resend that attempt, and no competition submission was made.

A fresh server pull and the native notebook page afterward still showed the previous public Version 38 (native version 358150421), COMPLETE, with unchanged source SHA-256 `9db7dfc45d9d1dc2ead51dc2a55c181cc3b48f17a846eaf05110279540aaccce`. GPU usage was unchanged and no GPU time was reserved. This is a failed publication attempt, not an inference result.

The preserved candidate was 1,642,705 bytes, SHA-256 `f5b1226d43a1e97425060f2c4ab5838f56a7df20118db37cccd765b80d7f15b8`. It retained all 145 existing cell sources and IDs, including the opening illustration, and appended the attributed reproduction. Independent source reviews and nine synthetic source-contract checks passed; native compatibility and numerical parity remain untested.

The next packaging investigation is lossless encoding of the added payloads while leaving every historical cell and decoded source byte intact. Kaggle staff have previously described a notebook source-size constraint in [this primary discussion](https://www.kaggle.com/discussions/questions-and-answers/103948). Size is a plausible explanation here, not a confirmed diagnosis of this HTTP response. Any new candidate needs its own exact-byte review and an explicit release decision; the failed attempt remains recorded.

The upstream visible score remains **0.957**; our verified notebook score remains **0.949**. The utility in this directory can be used independently, but its synthetic checks do not establish an improved competition score.
