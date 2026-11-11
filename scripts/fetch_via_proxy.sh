#!/usr/bin/env bash
# Fetch /workspace/results_sem.tgz from a pod that only has RunPod's SSH proxy (no scp):
# base64 chunks are printed between markers, md5-checked, then reassembled and extracted here.
# Usage: bash scripts/fetch_via_proxy.sh <pod_id-userhash>
set -euo pipefail
T=$1
cd "$(dirname "$0")/.."
D=$(mktemp -d)
n=$(bash scripts/pod_cmd.sh "$T" 'cd /tmp && rm -rf fx && mkdir fx && cd fx && md5sum /workspace/results_sem.tgz | cut -c1-32 > all.md5 && split -b 3M -d -a 3 /workspace/results_sem.tgz p_ && echo NCHUNK $(ls p_* | wc -l)' | tr -d '\r' | awk '/^NCHUNK [0-9]/{print $2}')
echo "chunks: $n"
want=$(bash scripts/pod_cmd.sh "$T" 'cat /tmp/fx/all.md5' | tr -d '\r' | grep -E '^[0-9a-f]{32}$')
for i in $(seq 0 $((n - 1))); do
    p=$(printf 'p_%03d' "$i")
    for try in 1 2 3 4 5; do
        bash scripts/pod_cmd.sh "$T" "echo BEGIN_$p; base64 -w 0 /tmp/fx/$p; echo; echo END_$p; md5sum /tmp/fx/$p | cut -c1-32" \
            | tr -d '\r' > "$D/$p.out" || true
        awk "/^BEGIN_$p\$/{f=1;next}/^END_$p\$/{f=0}f" "$D/$p.out" | tr -d '\n' | base64 -d > "$D/$p" 2>/dev/null || true
        m=$(awk "/^END_$p\$/{getline; print; exit}" "$D/$p.out")
        [ "$(md5sum < "$D/$p" | cut -c1-32)" = "$m" ] && break
        echo "retry $p ($try)"
    done
done
cat "$D"/p_[0-9][0-9][0-9] > "$D/all.tgz"
got=$(md5sum < "$D/all.tgz" | cut -c1-32)
[ "$got" = "$want" ] || { echo "md5 mismatch $got != $want"; exit 1; }
tar xzf "$D/all.tgz" --no-same-owner
mkdir -p ckpt/dino_physics && [ -f ckpt/dino_physics/teacher_backbone.pt ] || true
rm -rf "$D"
echo "FETCH_OK $got"
