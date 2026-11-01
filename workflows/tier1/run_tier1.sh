#!/usr/bin/env bash
# Tier 1, shipped configuration. --segskip-n 6 --fail-closed came after the reported end-to-end
# renders, which ran without them, so their utility numbers do not describe this mask.
#
#   ./run_tier1.sh <input.mp4> <output_dir> [n_people] [ttp_url]
#
# A TTP public-key endpoint must be reachable: Tier 1 refuses to mint a keypair locally, because
# holding the private key next to the data it protects defeats the third-party split. Start one with
#   python ../../tier3_restoration/scripts/ttp_stub.py <ttp_private_key.pem> 8843
set -euo pipefail

CLIP="${1:?usage: run_tier1.sh <input.mp4> <output_dir> [n_people] [ttp_url]}"
OUT="${2:?usage: run_tier1.sh <input.mp4> <output_dir> [n_people] [ttp_url]}"
PEOPLE="${3:-1}"
# 🔴 The default is a LOCAL TTP over plain HTTP, and --ttp-http below disables TLS verification.
# That is appropriate for a loopback development server and NOWHERE ELSE: the fetched RSA key is
# what every per-person recovery envelope is wrapped to, so an adversary who can answer this
# request substitutes its own key and can later decrypt every envelope. Against a remote TTP, drop
# --ttp-http and use a CA-validated https:// URL (or pin the fingerprint - see
# tier1/src/mirage/encryption.py, which documents that TOFU pinning is the intended end state).
TTP="${4:-http://127.0.0.1:8843}"

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

python "$REPO/tier1/scripts/run_tier1.py" "$CLIP" \
    --headless --no-save \
    --anonymizer yolo11n_boxfill \
    --segskip-n 6 --fail-closed \
    --gait-anon --gait-preset e2 \
    --mask-shape-mode bbox --mask-temporal-win 2 \
    --score-binarize-thresh 0.5 \
    --export-dir "$OUT" --export-people "$PEOPLE" \
    --ttp-server "$TTP" --ttp-http
