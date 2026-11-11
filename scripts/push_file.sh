#!/usr/bin/env bash
# Copy one local file to the pod through the SSH proxy (base64 in small chunks).
# Usage: bash scripts/push_file.sh <pod_id-userhash> <local> <remote>
T=$1; L=$2; R=$3
tmp=/tmp/push_$$.b64
bash scripts/pod_cmd.sh "$T" "rm -f $tmp" >/dev/null
base64 -w0 "$L" | { fold -w 2500; echo; } | while read -r c; do [ -n "$c" ] || continue
  bash scripts/pod_cmd.sh "$T" "printf '%s' '$c' >> $tmp" >/dev/null
done
want=$(md5sum "$L" | cut -c1-32)
got=$(bash scripts/pod_cmd.sh "$T" "base64 -d $tmp > $R && rm $tmp && md5sum $R" | tr -d '\r' | grep -ao '[0-9a-f]\{32\}' | head -1)
[ "$want" = "$got" ] && echo "pushed $L OK" || { echo "push FAILED $L ($want vs $got)"; exit 1; }
