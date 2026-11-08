# Conformance: what the 35 clips must decode to

The one copy of the expected outputs every build of the engine, and every SDK built on it, is
checked against, bit for bit: the greedy tokens, the chunk count and an FNV-1a fingerprint of every
output of every chunk (log_probs and the eleven heads).

| File | Input |
| - | - |
| `v4_c16_l64_expected.jsonl` | the 35 clips' features at c16_l64 (320 ms chunks, left context 64; the default) |
| `v4_c24_l72_expected.jsonl` | the same at c24_l72 (480 ms chunks, left context 72) |
| `v4_c24_l256_expected.jsonl` | the same at c24_l256 (480 ms chunks, left context 240) |
| `v4_c16_l64_audio_expected.jsonl` | the 35 clips as 16 kHz audio through the engine's own fbank and end-of-take flush, at c16_l64 |

The reference is ONNX Runtime 1.28.2 (armeabi-v7a) on an ARMv7 phone with one intra-op thread,
running the static int8 graph of the published model. The clips are the release asset
`conformance-data-v4-1` of this repository (`scripts/fetch-conformance-data.sh`); the model folder
is `sdk/v4.0` of Quran-Lab/tajweedformer-v4 on Hugging Face.

`bench/v4_conformance.sh BUILD MODEL DATA` runs every harness at every context and prints one line
per run; `bench/v4_conformance_check.py RUN EXPECTED` compares one run with one file.
