# Changelog

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
