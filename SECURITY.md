# Security

## Reporting a vulnerability

Please report security problems privately through GitHub: open the **Security** tab of this
repository and choose **Report a vulnerability**. Do not open a public issue for a security problem.

We will acknowledge the report, keep you informed while we fix it, and agree a disclosure date with
you once a fix is released.

## What is in scope

- The SDK code in this repository and the engine it builds (Quran-Lab/tajweedformer-engine).
- Model folder verification: the signed `manifest.json`, file sizes and SHA-256 checks, and the
  `min_sdk_version` guard. A way to make an SDK load a file that its manifest does not list, or a
  manifest that Quran Lab did not sign, is a vulnerability.
- The client side of the Quran Lab services the SDK talks to (`api.quranlab.ai`, `cdn.quranlab.ai`,
  `sdk.quranlab.ai`): signed configuration, developer keys, and the opt-in recording upload.

## Supported versions

| Version | Supported |
| - | - |
| 2.0.x | yes |
