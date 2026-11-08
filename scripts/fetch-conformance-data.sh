#!/bin/bash
# The 35 conformance clips (features and 16 kHz audio) into DIR/feats and DIR/audio:
#   scripts/fetch-conformance-data.sh DIR
# They are a release asset of this repository (conformance-data-v4-1). With the GitHub CLI logged in
# it uses `gh release download`; otherwise the public download URL.
set -euo pipefail
DIR=${1:?usage: fetch-conformance-data.sh DIR}
TAG=conformance-data-v4-1
ASSET=conformance-v4-clips.tar.gz
SHA=8b2a38d271535f1d81673c5d55b4dd78b8ada46138603e9997c157613027078b
mkdir -p "$DIR"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh release download "$TAG" -R Quran-Lab/tajweedformer-engine -p "$ASSET" -D "$DIR" --clobber
else
  curl -fL -o "$DIR/$ASSET" "https://github.com/Quran-Lab/tajweedformer-engine/releases/download/$TAG/$ASSET"
fi
if command -v sha256sum >/dev/null; then got=$(sha256sum "$DIR/$ASSET" | cut -d' ' -f1); else got=$(shasum -a 256 "$DIR/$ASSET" | cut -d' ' -f1); fi
[ "$got" = "$SHA" ] || { echo "checksum mismatch for $ASSET: $got" >&2; exit 1; }
tar -xzf "$DIR/$ASSET" -C "$DIR"
ls "$DIR/feats" | wc -l
