#!/usr/bin/env python3
"""the settings vectors through the C library:
each case's remote block (zf_tajweed_config_set_remote), app options (the app setters), backend
(zf_tajweed_config_set_backend) and debug flag (zf_tajweed_config_set_debug_build), then zf_tajweed_config_json
compared with the expected resolution (every key of the expected output; thresholds and taus within 1e-9) and
zf_tajweed_config_effective cross-checked against the same JSON. The invalid app options must be refused (-1).

  python -X utf8 tajweed/tests/run_config_vectors.py --lib <zf_tajweed_shared> [--wrap <\1 dir>]

The fixture calibration's tokens_sha256 is the placeholder "fixture"; the copy the test opens carries the real
tokens.txt hash (the library refuses a malformed one), next to the synthetic test table of tests/data/model.
"""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_vectors as RV  # noqa: E402

CLASSES = ["vowel_sub", "vowel_del", "vowel_ins", "short_to_long", "long_to_short", "cons_sub", "cons_del", "insertion"]
CHECKS = ["letters", "vowels", "length", "madd", "gemination", "qalqalah", "lazim"]
SOURCES = ["calibration", "remote", "app"]
STATUS = ["off", "experimental", "validated"]


class Effective(C.Structure):
    _fields_ = [("preset", C.c_int), ("preset_source", C.c_int), ("class_enabled", C.c_int * 8),
                ("class_status", C.c_int * 8), ("class_threshold", C.c_double * 8), ("class_source", C.c_int * 8),
                ("check_enabled", C.c_int * 7), ("check_status", C.c_int * 7), ("check_source", C.c_int * 7),
                ("context", C.c_char * 16), ("context_source", C.c_int)]


def same(a, b, path, bad):
    if isinstance(b, dict):
        if not isinstance(a, dict):
            bad.append(f"{path}: C {a!r} expected {b!r}")
            return
        for k in b:
            if k not in a:
                bad.append(f"{path}.{k}: missing in C")
            else:
                same(a[k], b[k], f"{path}.{k}", bad)
    elif isinstance(b, float) or (isinstance(b, int) and not isinstance(b, bool) and isinstance(a, float)):
        if not (isinstance(a, (int, float)) and not isinstance(a, bool) and math.isfinite(a) and abs(a - b) <= 1e-9):
            bad.append(f"{path}: C {a!r} expected {b!r}")
    elif a != b:
        bad.append(f"{path}: C {a!r} expected {b!r}")


def apply_app(L, cfg, app):
    """-> the first setter's return code that is not 0 (an invalid option), else 0"""
    rc = []
    for k in app:
        if k not in ("preset", "overrides", "checks", "context"):
            rc.append(-1)  # no setter exists for an unknown option: refused by construction
    if "preset" in app:
        rc.append(L.zf_tajweed_config_set_preset(cfg, str(app["preset"]).encode()))
    if "context" in app:
        rc.append(L.zf_tajweed_config_set_context(cfg, str(app["context"]).encode()))
    for c, o in (app.get("overrides") or {}).items():
        if not isinstance(o, dict) or ("preset" in o) == ("threshold" in o):
            rc.append(-1)  # both or neither: no single setter call expresses it
        elif "preset" in o:
            rc.append(L.zf_tajweed_config_set_class_preset(cfg, c.encode(), str(o["preset"]).encode()))
        else:
            rc.append(L.zf_tajweed_config_set_class_threshold(cfg, c.encode(), C.c_double(float(o["threshold"]))))
    for k, v in (app.get("checks") or {}).items():
        rc.append(L.zf_tajweed_config_set_check(cfg, k.encode(), 1 if v else 0))
    return next((r for r in rc if r != 0), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--wrap", default="<\1 dir>")
    ap.add_argument("--work", default="")
    a = ap.parse_args()
    vec = json.load(open(os.path.join(a.wrap, "config_vectors.json"), encoding="utf-8"))
    cal = json.load(open(os.path.join(a.wrap, vec["calibration"]), encoding="utf-8"))
    os.makedirs(a.work, exist_ok=True)
    tok = os.path.join(HERE, "data", "model", "tokens.txt")
    cal["tokens_sha256"] = hashlib.sha256(open(tok, "rb").read()).hexdigest()
    json.dump(cal, open(os.path.join(a.work, "calibration.json"), "w", encoding="utf-8"))
    shutil.copyfile(tok, os.path.join(a.work, "tokens.txt"))
    shutil.copyfile(os.path.join(HERE, "data", "model", "expected_tj3.zfx"), os.path.join(a.work, "expected_tj3.zfx"))
    lib = RV.Lib(os.path.abspath(a.lib))
    L = lib.L
    L.zf_tajweed_config_json.restype = C.c_char_p
    L.zf_tajweed_config_json.argtypes = [C.c_void_p]
    for name in ("zf_tajweed_config_set_preset", "zf_tajweed_config_set_context", "zf_tajweed_config_set_backend"):
        getattr(L, name).argtypes = [C.c_void_p, C.c_char_p]
    L.zf_tajweed_config_set_class_preset.argtypes = [C.c_void_p, C.c_char_p, C.c_char_p]
    L.zf_tajweed_config_set_class_threshold.argtypes = [C.c_void_p, C.c_char_p, C.c_double]
    L.zf_tajweed_config_set_check.argtypes = [C.c_void_p, C.c_char_p, C.c_int]
    L.zf_tajweed_config_set_debug_build.argtypes = [C.c_void_p, C.c_int]
    L.zf_tajweed_config_effective.argtypes = [C.c_void_p, C.POINTER(Effective)]
    L.zf_tajweed_api_version.restype = C.c_int
    m = lib.model(a.work, cal["default_context"])
    ok = fails = 0
    for case in vec["cases"]:
        cfg = L.zf_tajweed_config_create(m)
        bad = []
        try:
            if case["remote"] is None:
                L.zf_tajweed_config_set_remote(cfg, None, 0)
            else:
                b = json.dumps(case["remote"]).encode()
                n = L.zf_tajweed_config_set_remote(cfg, b, len(b))
                if n != len(case["expect"]["dropped"]) and not (case["expect"]["dropped"] == ["remote.context"]
                                                                 and n == 1):
                    bad.append(f"set_remote returned {n}, expected dropped {case['expect']['dropped']}")
            if apply_app(L, cfg, case["app"] or {}) != 0:
                bad.append("an app setter refused a valid option")
            if L.zf_tajweed_config_set_backend(cfg, case["backend"].encode()) != 0:
                bad.append("set_backend refused")
            L.zf_tajweed_config_set_debug_build(cfg, 1 if case["debug"] else 0)
            out = json.loads(L.zf_tajweed_config_json(cfg).decode("utf-8"))
            same(out, case["expect"], "", bad)
            # the effective struct says the same as the JSON
            e = Effective()
            if L.zf_tajweed_config_effective(cfg, C.byref(e)) != 0:
                bad.append("config_effective failed")
            if e.context.decode() != out["context"]:
                bad.append(f"effective context {e.context.decode()} json {out['context']}")
            for i, c in enumerate(CLASSES):
                s = out["sites"][c]
                thr = e.class_threshold[i]
                if bool(e.class_enabled[i]) != s["enabled"] or STATUS[e.class_status[i]] != s["status"] \
                        or SOURCES[e.class_source[i]] != s["source"] \
                        or (s["threshold"] is None) != (not math.isfinite(thr)) \
                        or (s["threshold"] is not None and abs(thr - s["threshold"]) > 1e-12):
                    bad.append(f"effective {c} differs from the JSON")
            if bool(e.check_enabled[3]) != out["madd"]["enabled"] or bool(e.check_enabled[6]) != out["lazim"]["enabled"] \
                    or bool(e.check_enabled[4]) != out["gemination"]["enabled"] \
                    or bool(e.check_enabled[5]) != out["qalqalah"]["enabled"]:
                bad.append("effective checks differ from the JSON")
        finally:
            L.zf_tajweed_config_destroy(cfg)
        if bad:
            fails += 1
            print(f"FAIL {case['name']}: " + "; ".join(bad[:5]))
        else:
            ok += 1
    inv_ok = 0
    for case in vec.get("app_invalid", []):
        cfg = L.zf_tajweed_config_create(m)
        rc = apply_app(L, cfg, case["app"])
        L.zf_tajweed_config_destroy(cfg)
        if rc == -1:
            inv_ok += 1
        else:
            print(f"FAIL invalid app option accepted: {case['name']}")
    n_inv = len(vec.get("app_invalid", []))
    print(f"api version {L.zf_tajweed_api_version()}")
    print(f"config vectors: {ok}/{ok + fails}")
    print(f"invalid app options refused: {inv_ok}/{n_inv}")
    sys.exit(1 if fails or inv_ok != n_inv else 0)


if __name__ == "__main__":
    main()
