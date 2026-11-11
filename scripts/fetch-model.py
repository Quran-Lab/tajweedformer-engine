#!/usr/bin/env python3
"""Download the TajweedFormer V4 model folder from Quran Lab with your app key.

    QURANLAB_KEY=qlk_v1_... python3 scripts/fetch-model.py DIR [--revision REV] [--decode]

Get access at https://sdk.quranlab.ai (NPL-1.2), create an app key on the portal, then run
this. It writes DIR/sdk/v4.0/... exactly as published: manifest.json first, then every file the manifest
lists, each checked against its size and SHA-256 from the manifest (a mismatch stops with an error).
--decode also writes the decoded copy of each gzip file next to it (its signed decoded size and SHA-256
are checked too), for tools that read a plain folder.

The revision: --revision, else QURANLAB_MODEL_REVISION, else the current release that the CDN's signed
file list names for the model. Files come from https://cdn.quranlab.ai/models/{repo}/{revision}/{path}.
Python 3.8+, standard library only.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import urllib.request

BASE = os.environ.get("QURANLAB_MODELS", "https://cdn.quranlab.ai/models")
CDN = BASE[: -len("/models")] if BASE.endswith("/models") else BASE  # /manifest.json is at the root of the CDN
REPO = "Quran-Lab/tajweedformer-v4"
FOLDER = "sdk/v4.0"


def get(url, key):
    req = urllib.request.Request(url, headers={"x-qlab-key": key, "user-agent": "tajweedformer-fetch-model"})
    return urllib.request.urlopen(req, timeout=120)


def fetch(url, key, dest, size, sha256):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h, n = hashlib.sha256(), 0
    with get(url, key) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            f.write(b)
    if n != size or h.hexdigest() != sha256:
        os.remove(dest + ".part")
        sys.exit(f"fetch-model: {url} does not match the manifest (size {n} vs {size} or sha256)")
    os.replace(dest + ".part", dest)


def main():
    ap = argparse.ArgumentParser(description="Download the V4 model folder with an app key.")
    ap.add_argument("dir")
    ap.add_argument("--revision", default=os.environ.get("QURANLAB_MODEL_REVISION", ""))
    ap.add_argument("--decode", action="store_true", help="also write decoded copies of gzip files")
    a = ap.parse_args()
    key = os.environ.get("QURANLAB_KEY", "")
    if not key:
        sys.exit("fetch-model: set QURANLAB_KEY to your app key (sdk.quranlab.ai)")
    rev = a.revision
    if not rev:
        with get(f"{CDN}/manifest.json?repo={REPO}", key) as r:
            rev = json.load(r).get("releases", {}).get(REPO, "")
        if not rev:
            sys.exit("fetch-model: no released revision found; pass --revision")
    base = f"{BASE}/{REPO}/{rev}/{FOLDER}"
    out = os.path.join(a.dir, *FOLDER.split("/"))
    os.makedirs(out, exist_ok=True)
    with get(f"{base}/manifest.json", key) as r:
        raw = r.read()
    m = json.loads(raw)
    with open(os.path.join(out, "manifest.json"), "wb") as f:
        f.write(raw)
    for e in m["files"]:
        dest = os.path.join(out, *e["path"].split("/"))
        fetch(f"{base}/{e['path']}", key, dest, e["size"], e["sha256"].lower())
        d = e.get("decoded")
        if a.decode and e.get("encoding") == "gzip" and d:
            plain = os.path.join(out, *d["path"].split("/"))
            h, n = hashlib.sha256(), 0
            with gzip.open(dest, "rb") as src, open(plain, "wb") as dst:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    h.update(b)
                    n += len(b)
                    dst.write(b)
            if n != d["size"] or h.hexdigest() != d["sha256"]:
                sys.exit(f"fetch-model: {e['path']} does not decode to the manifest's {d['path']}")
    print(f"{REPO}@{rev}: {len(m['files'])} files in {out}")


if __name__ == "__main__":
    main()
