#!/usr/bin/env bash
# Robust copy pod -> laptop over a flaky link: tar.gz on the pod, split into 4 MB chunks,
# fetch each chunk with retries, verify md5, then extract here.
# Usage: bash scripts/fetch_from_pod.sh <host> <port> <path relative to repo> [more paths...]
set -euo pipefail
HOST=$1; PORT=$2; shift 2
KEY="$HOME/.ssh/id_ed25519_runpod"
O="-i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
cd "$(dirname "$0")/.."
ID=xfer_$$
ssh $O -p "$PORT" root@"$HOST" "cd /workspace/sem-ssl-vit && mkdir -p /tmp/$ID && \
  tar czf /tmp/$ID/all.tgz --exclude='*.lmdb' --exclude='*.lmdb-lock' $* && cd /tmp/$ID && \
  md5sum all.tgz > all.md5 && split -b 4M -d -a 4 all.tgz part_ && rm all.tgz && ls part_*" > /tmp/$ID.list
mkdir -p /tmp/$ID
for p in $(cat /tmp/$ID.list); do
    for try in 1 2 3 4 5 6; do
        scp -q $O -P "$PORT" root@"$HOST":/tmp/$ID/$p /tmp/$ID/ && break
        echo "retry $p ($try)"; sleep 3
    done
done
scp -q $O -P "$PORT" root@"$HOST":/tmp/$ID/all.md5 /tmp/$ID/
cat /tmp/$ID/part_* > /tmp/$ID/all.tgz
(cd /tmp/$ID && md5sum -c all.md5)
tar xzf /tmp/$ID/all.tgz --no-same-owner
ssh $O -p "$PORT" root@"$HOST" "rm -rf /tmp/$ID" || true
rm -rf /tmp/$ID /tmp/$ID.list
echo "fetched: $*"
