#!/usr/bin/env python3
"""The tajweed library (C, through ctypes) against the reference scorer.

  python -X utf8 tajweed/tests/run_vectors.py --lib <zf_tajweed_shared.dll|.so> --model <model dir>
        [--ref-dir <\1 dir>] [--synth <make_synth.py out dir>]
        [--presets balanced,very_sensitive]

Without --synth: the reference's vectors (<ref-dir>/vectors): each vector's saved heads and expected row, scored by the C
library from <model dir> (calibration.json, tokens.txt, expected_tj3.zfx) and compared with the reference verdicts
CAL saved for each preset (and, as a cross-check, with tajweed_ref.verdicts run now).
With --synth: make_synth.py's synthetic table and heads, the reference run now with the same inputs.

Compared per vector and preset: the scored path and its edit cost; every site (window): flagged, reported class, and
per class the best alternative, enabled, flagged and the per-head and fused LLRs (within 1e-3); madd: items, class,
count, flag; qalqalah and gemination: items and flags. Prints N/N per context and per preset; exits 1 on any
mismatch.
"""
import argparse
import collections
import ctypes as C
import glob
import gzip
import importlib.util
import json
import math
import os
import sys

import numpy as np

TOL = 1e-3
HEAD_ORDER = ("log_probs", "inter1_log_probs", "inter4_log_probs", "id_log_probs", "timing_log_probs", "gem",
              "attr_tafkheem", "attr_ghunna", "attr_qalqalah", "attr_face", "boundary", "conf")


class OpenOptions(C.Structure):
    _fields_ = [("struct_size", C.c_size_t), ("context", C.c_char_p), ("load_engine", C.c_int), ("threads", C.c_int),
                ("sample_rate", C.c_int), ("batch", C.c_int), ("no_pin", C.c_int)]  # API 3: no_pin appended


class Heads(C.Structure):
    _fields_ = [("n40", C.c_int), ("n20", C.c_int), ("vocab", C.c_int)] + [(k, C.POINTER(C.c_float)) for k in HEAD_ORDER]


def flat(v):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 3:
        return v.reshape(-1, v.shape[-1])
    if v.ndim == 2 and v.shape[0] > 1 and v.shape[-1] not in (2, 3, 4, 5, 38):
        return v.reshape(-1)
    return v


class Lib:
    def __init__(self, path):
        L = C.CDLL(path)
        L.zf_tajweed_open_default_options.restype = OpenOptions
        L.zf_tajweed_model_open.restype = C.c_void_p
        L.zf_tajweed_model_open.argtypes = [C.c_char_p, C.POINTER(OpenOptions), C.POINTER(C.c_char_p)]
        L.zf_tajweed_model_close.argtypes = [C.c_void_p]
        L.zf_tajweed_config_create.restype = C.c_void_p
        L.zf_tajweed_config_create.argtypes = [C.c_void_p]
        L.zf_tajweed_config_destroy.argtypes = [C.c_void_p]
        L.zf_tajweed_config_set_remote.argtypes = [C.c_void_p, C.c_char_p, C.c_size_t]
        L.zf_tajweed_score.restype = C.c_void_p
        L.zf_tajweed_score.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(Heads), C.c_int, C.c_int, C.POINTER(C.c_char_p)]
        L.zf_tajweed_result_json.restype = C.c_char_p
        L.zf_tajweed_result_json.argtypes = [C.c_void_p]
        L.zf_tajweed_result_free.argtypes = [C.c_void_p]
        self.L = L
        self.models = {}

    def model(self, model_dir, ctx):
        k = (model_dir, ctx)
        if k not in self.models:
            o = self.L.zf_tajweed_open_default_options()
            o.load_engine = 0
            o.context = ctx.encode()
            err = C.c_char_p()
            m = self.L.zf_tajweed_model_open(model_dir.encode(), C.byref(o), C.byref(err))
            if not m:
                raise RuntimeError(f"model_open({model_dir}, {ctx}): {err.value.decode()}")
            self.models[k] = m
        return self.models[k]

    def score(self, model_dir, ctx, heads, surah, ayah, remote):
        m = self.model(model_dir, ctx)
        cfg = self.L.zf_tajweed_config_create(m)
        try:
            if remote is not None:
                b = json.dumps(remote).encode()
                dropped = self.L.zf_tajweed_config_set_remote(cfg, b, len(b))
                if dropped != 0:
                    raise RuntimeError(f"remote block dropped {dropped} fields: {remote}")
            keep, h = [], Heads()
            for k in HEAD_ORDER:
                if k in heads:
                    a = np.ascontiguousarray(flat(heads[k]), dtype=np.float32)
                    keep.append(a)
                    setattr(h, k, a.ctypes.data_as(C.POINTER(C.c_float)))
            h.n40 = flat(heads["log_probs"]).shape[0]
            h.n20 = flat(heads["timing_log_probs"]).shape[0] if "timing_log_probs" in heads else 0
            h.vocab = 38
            err = C.c_char_p()
            r = self.L.zf_tajweed_score(m, cfg, C.byref(h), surah, ayah, C.byref(err))
            if not r:
                raise RuntimeError(f"score {surah}:{ayah}: {err.value.decode()}")
            out = json.loads(self.L.zf_tajweed_result_json(r).decode("utf-8"))
            self.L.zf_tajweed_result_free(r)
            return out
        finally:
            self.L.zf_tajweed_config_destroy(cfg)


def num_eq(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a) or isinstance(b, float) and math.isnan(b):
        return isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def compare(c, ref):
    """-> list of mismatch strings (empty = pass)"""
    bad = []
    if c["path"] != ref["path"]:
        bad.append(f"path differs: C {' '.join(c['path'])} | ref {' '.join(ref['path'])}")
        return bad
    if c.get("path_edit_cost") != ref.get("path_edit_cost"):
        bad.append(f"path_edit_cost C {c.get('path_edit_cost')} ref {ref.get('path_edit_cost')}")
    for wc, wr in zip(c["path_forms"], ref["path_forms"]):
        for k in ("pick", "L", "R", "mode", "wajh", "cross"):
            if wc.get(k) != wr.get(k):
                bad.append(f"path_forms word {wr['word']}: {k} C {wc.get(k)} ref {wr.get(k)}")
    if [s["i"] for s in c["sites"]] != [s["i"] for s in ref["sites"]]:
        bad.append("site windows differ")
        return bad
    for sc, sr in zip(c["sites"], ref["sites"]):
        tag = f"site i={sr['i']}"
        if sc["expected"] != sr["expected"]:
            bad.append(f"{tag}: expected C {sc['expected']} ref {sr['expected']}")
        if bool(sc["flagged"]) != bool(sr["flagged"]) or sc["class"] != sr["class"]:
            bad.append(f"{tag}: verdict C {sc['flagged']}/{sc['class']} ref {sr['flagged']}/{sr['class']}")
        if set(sc["per_class"]) != set(sr["per_class"]):
            bad.append(f"{tag}: classes C {sorted(sc['per_class'])} ref {sorted(sr['per_class'])}")
            continue
        for cls, pr in sr["per_class"].items():
            pc = sc["per_class"][cls]
            if pc["alt"] != pr["alt"] or pc["alt_window"] != pr["alt_window"]:
                bad.append(f"{tag} {cls}: alt C {pc['alt']} ref {pr['alt']}")
            if bool(pc["enabled"]) != bool(pr["enabled"]) or bool(pc["flagged"]) != bool(pr["flagged"]):
                bad.append(f"{tag} {cls}: enabled/flagged C {pc['enabled']}/{pc['flagged']} ref {pr['enabled']}/{pr['flagged']}"
                           f" (fused {pr['fused']} thr {pr['threshold']})")
            for f in ("llr_log_probs", "llr_inter1", "fused"):
                if not num_eq(pc[f], pr[f], TOL):
                    bad.append(f"{tag} {cls}: {f} C {pc[f]} ref {pr[f]}")
            if not num_eq(pc["threshold"], pr["threshold"], 1e-12):
                bad.append(f"{tag} {cls}: threshold C {pc['threshold']} ref {pr['threshold']}")
    for part, pname in (("madd", "ratio"), ("qalqalah", "p_burst"), ("gemination", "p_held")):
        if (part in c) != (part in ref):
            bad.append(f"{part}: present C {part in c} ref {part in ref}")
            continue
        if part not in ref:
            continue
        ic, ir = c[part]["items"], ref[part]["items"]
        if [x["i"] for x in ic] != [x["i"] for x in ir]:
            bad.append(f"{part}: items C {[x['i'] for x in ic]} ref {[x['i'] for x in ir]}")
            continue
        if part == "madd" and not num_eq(c[part]["unit_frames"], ref[part]["unit_frames"], 1e-9):
            bad.append(f"madd unit C {c[part]['unit_frames']} ref {ref[part]['unit_frames']}")
        for a, b in zip(ic, ir):
            keys = ("flagged",) + (("cls", "count", "aared_at_stop", "scoring") if part == "madd" else ())
            for k in keys:
                if a[k] != b[k]:
                    bad.append(f"{part} i={b['i']}: {k} C {a[k]} ref {b[k]}")
            if a.get("reported", True) != b.get("reported", True):
                bad.append(f"{part} i={b['i']}: reported C {a.get('reported', True)} ref {b.get('reported', True)}")
            if not num_eq(a[pname], b[pname], 1e-5):
                bad.append(f"{part} i={b['i']}: {pname} C {a[pname]} ref {b[pname]}")
    # API 3: the per-word score (tajweed_ref.word_scores): flags exact, scores and LLRs within TOL
    if ("words" in c) != ("words" in ref):
        bad.append(f"words: present C {'words' in c} ref {'words' in ref}")
    elif "words" in ref:
        wc, wr = c["words"], ref["words"]
        if not num_eq(wc["tau"], wr["tau"], 1e-12) or wc["preset"] != wr["preset"]:
            bad.append(f"words: tau/preset C {wc['tau']}/{wc['preset']} ref {wr['tau']}/{wr['preset']}")
        ic, ir = wc["items"], wr["items"]
        if [(x["word"], x["tokens"]) for x in ic] != [(x["word"], x["tokens"]) for x in ir]:
            bad.append("words: word windows differ")
        else:
            WORD_TALLY[0] += len(ir)
            for a, b in zip(ic, ir):
                tag = f"word {b['word']}"
                ok = bool(a["flagged"]) == bool(b["flagged"]) and num_eq(a.get("score"), b.get("score"), TOL)
                if b.get("score") is not None:
                    for k in ("alt", "alt_window", "class"):
                        ok = ok and a.get(k) == b.get(k)
                    for k in ("llr_log_probs", "llr_inter1"):
                        ok = ok and num_eq(a.get(k), b.get(k), TOL)
                if not ok:
                    bad.append(f"{tag}: C {a.get('score')}/{a.get('alt')}/{a['flagged']} ref {b.get('score')}/"
                               f"{b.get('alt')}/{b['flagged']}")
                else:
                    WORD_TALLY[1] += 1
                    WORD_TALLY[2] = max(WORD_TALLY[2], abs(a["score"] - b["score"]) if b.get("score") is not None else 0)
                    WORD_TALLY[3] += bool(b["flagged"])
    return bad


WORD_TALLY = [0, 0, 0.0, 0]  # word items compared, identical, max |score diff|, flagged in the reference


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_ref(ref_dir):
    spec = importlib.util.spec_from_file_location("tajweed_ref", os.path.join(ref_dir, "tajweed_ref.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tally_add(tally, keys, ok):
    for k in keys:
        tally[k][1] += 1
        tally[k][0] += ok


def run_synth(args, lib, R):
    model = os.path.join(args.synth, "model")
    cal = json.load(open(os.path.join(model, "calibration.json"), encoding="utf-8"))
    rows = {}
    with gzip.open(os.path.join(model, "expected_synth.jsonl.gz"), "rt", encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            rows[(r["surah"], r["ayah"])] = r
    tally, fails = collections.defaultdict(lambda: [0, 0]), 0
    for meta_path in sorted(glob.glob(os.path.join(args.synth, "vectors", "*.json"))):
        meta = json.load(open(meta_path))
        heads = load_npz(meta_path[:-5] + ".npz")
        row = rows[(meta["surah"], meta["ayah"])]
        for preset in args.presets:
            remote = {"v": 1, "preset": preset}
            ref = R.verdicts(R.load_heads(dict(heads)), row, cal, meta["context"], remote=remote)
            out = lib.score(model, meta["context"], heads, meta["surah"], meta["ayah"], remote)
            bad = compare(out, ref)
            tally_add(tally, (f"context {meta['context']}", f"preset {preset}"), not bad)
            if bad:
                fails += 1
                print(f"FAIL {os.path.basename(meta_path)} {preset}: " + "; ".join(bad[:4]))
    return tally, fails


def run_cal(args, lib, R):
    """the reference's vectors: layout read from <ref-dir>/vectors (see cal_layout below)."""
    import cal_layout
    return cal_layout.run(args, lib, R, compare, tally_add, load_npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--ref-dir", default="<\1 dir>")
    ap.add_argument("--synth", default=None)
    ap.add_argument("--presets", default="balanced,very_sensitive")
    ap.add_argument("--word-configs", action="store_true",
                    help="the reference's vectors also at the word presets and custom thresholds (fresh reference)")
    ap.add_argument("--work", default="", help="scratch folder for the test table")
    args = ap.parse_args()
    args.presets = args.presets.split(",")
    lib = Lib(os.path.abspath(args.lib))
    R = load_ref(args.ref_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    tally, fails = run_synth(args, lib, R) if args.synth else run_cal(args, lib, R)
    for k in sorted(tally):
        print(f"{k}: {tally[k][0]}/{tally[k][1]}")
    if WORD_TALLY[0]:
        print(f"words: {WORD_TALLY[1]}/{WORD_TALLY[0]} identical (flags exact, score/LLRs within {TOL}; "
              f"max |score diff| {WORD_TALLY[2]:.3g}; {WORD_TALLY[3]} flagged in the reference)")
    sys.exit(1 if fails or not tally else 0)


if __name__ == "__main__":
    main()
