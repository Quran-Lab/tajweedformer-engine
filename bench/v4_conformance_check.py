#!/usr/bin/env python3
"""V4 (TajweedFormer) conformance: every output of every chunk, bit for bit.

    v4_conformance_check.py <run.jsonl> <expected.jsonl>          compare, print N/N, exit 1 on any difference
    v4_conformance_check.py --make <reference.jsonl> <out.jsonl>   turn a reference run into an expected file
    v4_conformance_check.py <run.jsonl> <expected.jsonl> --subset a,b,c
        a run that computed only some heads (zf_stream_select_outputs, zf_tajweed_stream_set_heads):
        log_probs and the listed outputs must be bit-identical, and every other output must be absent
        or all FNV 0 (not computed), which proves the skipped heads were really skipped

A run is zf_set's or zf_stream_check's output: one line per clip with "features" (the clip's path),
"chunks", "tokens" (greedy CTC on log_probs, blank 0), "logp_fnv" (FNV-1a over every chunk's
log_probs) and "out_fnv": for each of the graph's outputs other than the carried states (log_probs
and the eleven tajweed heads), one FNV-1a per chunk. The expected file keeps "clip" (the feature
file's stem), "chunks", "tokens", "logp_fnv" and "out_fnv". A clip passes when all of them are
equal: the tokens, the chunk count, and every value of every output of every chunk.
"""
import json
import os
import sys

V4_OUTPUTS = ["attr_face", "attr_ghunna", "attr_qalqalah", "attr_tafkheem", "boundary", "conf", "gem",
              "id_log_probs", "inter1_log_probs", "inter4_log_probs", "log_probs", "timing_log_probs"]


def clip_of(r):
    return r["clip"] if "clip" in r else os.path.splitext(os.path.basename(r["features"].replace("\\", "/")))[0]


def load(path):
    rows = (json.loads(line) for line in open(path, encoding="utf-8") if line.strip())
    return {clip_of(r): r for r in rows if "features" in r or "clip" in r}


def make(ref_path, out_path):
    rows = load(ref_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for clip in sorted(rows):
            r = rows[clip]
            missing = [k for k in V4_OUTPUTS if k not in r.get("out_fnv", {})]
            if missing:
                raise SystemExit(f"{clip}: the reference lacks {missing}")
            f.write(json.dumps({"clip": clip, "chunks": r["chunks"], "tokens": r["tokens"],
                                "logp_fnv": r["logp_fnv"],
                                "out_fnv": {k: r["out_fnv"][k] for k in V4_OUTPUTS}}) + "\n")
    print(f"{len(rows)} clips -> {out_path}")


NOT_COMPUTED = "0000000000000000"


def check(run_path, exp_path, subset=None):
    exp, got = load(exp_path), load(run_path)
    want = V4_OUTPUTS if subset is None else sorted(set(subset) | {"log_probs"})
    unknown = [k for k in want if k not in V4_OUTPUTS]
    if unknown:
        raise SystemExit(f"--subset names unknown outputs: {unknown}")
    bad = []
    for clip, e in sorted(exp.items()):
        g = got.get(clip)
        why = None
        if g is None:
            why = "missing"
        elif g["chunks"] != e["chunks"]:
            why = f"chunks {g['chunks']} != {e['chunks']}"
        elif g["tokens"] != e["tokens"]:
            why = "tokens"
        elif g["logp_fnv"] != e["logp_fnv"]:
            why = "log_probs"
        else:
            go = g.get("out_fnv", {})
            for k in V4_OUTPUTS:
                if k not in want:
                    if any(v != NOT_COMPUTED for v in go.get(k, [])):
                        why = f"{k} was computed but not asked for"
                        break
                    continue
                if go.get(k) != e["out_fnv"][k]:
                    first = next((i for i, (a, b) in enumerate(zip(go.get(k) or [], e["out_fnv"][k])) if a != b), 0)
                    why = f"{k} from chunk {first}"
                    break
        if why:
            bad.append(f"{clip} ({why})")
    ok = len(exp) - len(bad)
    what = (f"all {len(V4_OUTPUTS)} outputs" if subset is None
            else f"the {len(want)} asked-for outputs, the other {len(V4_OUTPUTS) - len(want)} not computed")
    print(f"{ok}/{len(exp)} clips bit-identical (tokens, chunks, every value of {what} "
          f"of every chunk)" + (f"; differ: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--make":
        make(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        sys.exit(check(sys.argv[1], sys.argv[2]))
    elif len(sys.argv) == 5 and sys.argv[3] == "--subset":
        sys.exit(check(sys.argv[1], sys.argv[2], [k for k in sys.argv[4].split(",") if k and k != "none"]))
    else:
        raise SystemExit(__doc__)
