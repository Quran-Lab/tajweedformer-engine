# Changelog

## 3.0.0

Breaking: the engine is inference only. The tajweed scorer is gone; inference itself is unchanged.

- Removed from `zf_tajweed.h` (API version 4): the scorer (`zf_tajweed_score`, every `zf_tajweed_result_*`),
  the settings (every `zf_tajweed_config_*`), the class, check, preset and word preset name tables with their
  types, and the expected table as part of a model (`zf_tajweed_ayah_count`, `zf_tajweed_expected_tokens`,
  `zf_tajweed_expected_words`). `zf_tajweed_model_id`, `zf_tajweed_model_context_at` and
  `zf_tajweed_model_default_context` are gone too: they were read from `calibration.json`, which the engine no
  longer parses; the model folder's signed manifest has them. The header's opening comment has the full list.
- `zf_tajweed_model_open` reads `tokens.txt` and the context's engine files, and nothing else: a folder without
  `calibration.json` and without `expected_tj3.zfx` opens and runs. The context must now be given
  (`options.context`).
- `calibration.json` is data: `zf_tajweed_model_calibration` returns its bytes as shipped (or NULL). The engine
  never parses or applies it.
- The expected units of each ayah are data too, with a reader of their own that has no link to a model:
  `zf_tajweed_reference_open`, `_open_memory`, `_close`, `_ayah_count`, `_units`, `_tokens_sha256`. For the same
  file they return what the 2.x calls returned (checked on all 6,236 ayahs of the shipped table).
- `ZF_TAJWEED_HEADS_SCORER` is now `ZF_TAJWEED_HEADS_DEFAULT` (the same five heads).
- One CMake target to link: `zf_tajweed_inference` (it replaces `zf_tajweed_engine` and `zf_tajweed`).
- Inference is unchanged, and proven so: the 35 conformance clips give the same result before and after the
  removal, every output of every chunk bit for bit, at every context (`bench/v4_conformance.sh`: 27 runs at
  35/35 and the decoded layer and top-K checks at 0 mismatches), the second time on a model folder without
  `calibration.json` and `expected_tj3.zfx`.
- New: tests under `ctest` that need no model file and no network (`engine/tests`, `tajweed/tests`): SHA-256,
  the in-memory file table, the vendored `expf`, `logf` and `powf`, the greedy CTC decoder, the resampler, the
  fbank, the int8 kernels against a scalar reference, the decoded layer, the model folder, the reference data
  reader and the header's surface. CI runs them on Linux, macOS and Windows.

## 2.0.0 (first public release)

- The exact engine for TajweedFormer V4: all twelve outputs of the network (log_probs and the
  eleven heads) for every chunk, at the three shipped contexts c16_l64 (default), c24_l72 and c24_l256.
- `zf_tajweed.h`: model folders (layout 3: one shared `layers.zfw` and `frontend.zfw` for every context,
  per-context graphs and geometry; also from memory), streams with selectable heads (lazy: a head nobody
  asked for is not computed), the top-K tokens per frame, the layered tajweed settings (app code, then the
  portal's block, then `calibration.json`, including the word level) and the scorer (per-site verdicts with
  every LLR, per-word scores, madd counts, qalqalah and gemination evidence).
- Thread pinning: off by default on desktops and servers, an option per stream; a pinned caller's mask is
  restored when its stream ends.
- The Apple sidecar (`zf/zfs.h`): the CPU engine's weights rebuilt from a CoreML package's weights.
- Bit-exact with ONNX Runtime's ARM int8 reference on the 35 conformance clips at every context, on
  ARMv7, AArch64, x86-64 (through the NEON emulation) and WebAssembly.
- Builds with GCC, clang, MinGW-w64 and clang for the Microsoft C runtime; Android, Apple, Linux,
  Windows and Emscripten.
