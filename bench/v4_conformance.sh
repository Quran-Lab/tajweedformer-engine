#!/bin/bash
# V4 (TajweedFormer) conformance on a host build: every context's twelve outputs, every chunk.
#
#   bench/v4_conformance.sh <dev build dir> <model folder> <data dir>
#
# <model folder> is sdk/v4.0's layout: calibration.json, tokens.txt, expected_tj3.zfx and one folder per
# context (c16_l64, c8_l64, c24_l72, c24_l256) with the engine's files (tools/zfv4.py). <data dir> holds the 35
# parity clips as feats/*.f32 and audio/*.pcm (conformance-data-2). The expected outputs are this
# repository's conformance/v4_*_expected.jsonl. Exits nonzero if any line is short of 35/35.
set -uo pipefail
abs() { (cd "$1" && { pwd -W 2>/dev/null || pwd; }); }
PY=${PYTHON:-python3}
"$PY" -c '' 2>/dev/null || PY=python
B=$(abs "$1")/bench/accel M=$(abs "$2") DATA=$(abs "$3")
HERE=$(abs "$(dirname "$0")")
EXE=; [ -f "$B/zf_set_rt.exe" ] && EXE=.exe
OUT=$(mktemp -d)
ls "$DATA"/feats/*.f32 > "$OUT/feats.txt"
ls "$DATA"/audio/*.pcm > "$OUT/audio.txt"
fails=0
# the scorer's heads (zf_tajweed.h's default, ZF_TAJWEED_HEADS_SCORER): the lazy-head lines below run
# only these and require the other seven outputs to be absent (not computed)
SCORER=inter1_log_probs,timing_log_probs,gem,attr_qalqalah
check() { # name run expected [subset]
  local r
  r=$("$PY" "$HERE/v4_conformance_check.py" "$2" "$3" ${4:+--subset "$4"} | tail -1 | cut -c1-90)
  printf '%-40s %s\n' "$1" "$r"
  case "$r" in 35/35*) ;; *) fails=$((fails + 1)) ;; esac
}
for c in c16_l64 c8_l64 c24_l72 c24_l256; do
  [ -d "$M/$c" ] || { echo "no $c in $M, skipped"; continue; }
  E=$HERE/../conformance/v4_${c}_expected.jsonl
  (cd "$M/$c" && ZF_FE_F32=1 ZF_CONV_ORT=1 "$B/zf_set_rt$EXE" model_engine2.onnx "$OUT/feats.txt" 4 "$OUT/set.jsonl" >/dev/null 2>&1)
  check "$c zf_set_rt" "$OUT/set.jsonl" "$E"
  (cd "$M/$c" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/feats.txt" 4 "$OUT/st.jsonl" >/dev/null 2>&1)
  check "$c stream, 1/7/13/3/29-frame pushes" "$OUT/st.jsonl" "$E"
  (cd "$M/$c" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/feats.txt" 4 "$OUT/b8.jsonl" --batch 8 >/dev/null 2>&1)
  check "$c stream, batch 8" "$OUT/b8.jsonl" "$E"
  (cd "$M/$c" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/feats.txt" 4 "$OUT/lz.jsonl" --outputs $SCORER >/dev/null 2>&1)
  check "$c stream, scorer heads only" "$OUT/lz.jsonl" "$E" "$SCORER"
  (cd "$M/$c" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/feats.txt" 4 "$OUT/lzb.jsonl" --batch 8 --outputs $SCORER >/dev/null 2>&1)
  check "$c stream, batch 8, scorer heads only" "$OUT/lzb.jsonl" "$E" "$SCORER"
done
# pinning is the platform's default (not pinned on a desktop, pinned on a phone) and an explicit choice
# overrides it: the other way round must give the same bits
E=$HERE/../conformance/v4_c16_l64_expected.jsonl
(cd "$M/c16_l64" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/feats.txt" 4 "$OUT/pin.jsonl" --pin >/dev/null 2>&1)
check "c16_l64 stream, pinned explicitly" "$OUT/pin.jsonl" "$E"
(cd "$M/c16_l64" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/feats.txt" 4 "$OUT/nopin.jsonl" --no-pin >/dev/null 2>&1)
check "c16_l64 stream, not pinned explicitly" "$OUT/nopin.jsonl" "$E"
A=$HERE/../conformance/v4_c16_l64_audio_expected.jsonl
(cd "$M/c16_l64" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/audio.txt" 4 "$OUT/au.jsonl" >/dev/null 2>&1)
check "c16_l64 stream, audio" "$OUT/au.jsonl" "$A"
"$B/zf_tajweed_check$EXE" "$M" c16_l64 "$OUT/audio.txt" "$OUT/tj.jsonl" 4 --heads all >/dev/null 2>"$OUT/tj.err"
check "c16_l64 zf_tajweed.h, audio" "$OUT/tj.jsonl" "$A"
"$B/zf_tajweed_check$EXE" "$M" c16_l64 "$OUT/audio.txt" "$OUT/tjs.jsonl" 4 --heads scorer >/dev/null 2>"$OUT/tjs.err"
check "c16_l64 zf_tajweed.h, audio, scorer heads" "$OUT/tjs.jsonl" "$A" "$SCORER"
# the decoded layer (zf_tajweed_stream_top2) against the greedy transcript and the log-probs
t2=$(grep -o '"top2_mismatch": [0-9]*' "$OUT/tjs.err" | sed 's/.*: //')
printf '%-40s %s\n' "c16_l64 decoded layer (top-1, runner-up)" "top2_mismatch ${t2:-missing}"
[ "${t2:-1}" = 0 ] || fails=$((fails + 1))
# top-K (API 3): k = 2 against top-2, k = cols against a full stable argsort, every head each run keeps
for r in tj tjs; do
  tk=$(grep -o '"topk_mismatch": [0-9]*' "$OUT/$r.err" | sed 's/.*: //')
  th=$(grep -o '"topk_heads_checked": [0-9]*' "$OUT/$r.err" | sed 's/.*: //')
  printf '%-40s %s
' "c16_l64 top-K, $([ $r = tj ] && echo all || echo scorer) heads" "topk_mismatch ${tk:-missing} (${th:-0} head rankings checked)"
  [ "${tk:-1}" = 0 ] && [ "${th:-0}" -gt 0 ] || fails=$((fails + 1))
done
# c8_l64 (160 ms) has its own audio reference (ONNX Runtime armv7, the c16_l64 recipe); only a folder that ships it runs these
if [ -d "$M/c8_l64" ]; then
  A8=$HERE/../conformance/v4_c8_l64_audio_expected.jsonl
  (cd "$M/c8_l64" && "$B/zf_stream_check$EXE" model_engine2.onnx "$OUT/audio.txt" 4 "$OUT/au8.jsonl" >/dev/null 2>&1)
  check "c8_l64 stream, audio" "$OUT/au8.jsonl" "$A8"
  "$B/zf_tajweed_check$EXE" "$M" c8_l64 "$OUT/audio.txt" "$OUT/tj8.jsonl" 4 --heads all >/dev/null 2>"$OUT/tj8.err"
  check "c8_l64 zf_tajweed.h, audio" "$OUT/tj8.jsonl" "$A8"
fi
rm -rf "$OUT"
echo "$fails failing"
exit $((fails > 0))
