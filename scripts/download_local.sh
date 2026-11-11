#!/usr/bin/env bash
# Resumable, checksum-verified download of the NFFA tars (for slow / flaky connections).
cd "$(dirname "$0")/.."
mkdir -p data/tars data/raw
one() {
  key=$1; md5=$2; url=$3; cls=${key%.tar}
  [ -f "data/raw/$cls/.done" ] && return 0
  for i in $(seq 1 40); do
    curl -sfL -C - --retry 5 --retry-delay 5 -o "data/tars/$key" "$url"
    [ "$(md5sum "data/tars/$key" | cut -c1-32)" = "$md5" ] && break
    sleep 5
  done
  [ "$(md5sum "data/tars/$key" | cut -c1-32)" = "$md5" ] || { echo "FAILED $key"; return 1; }
  rm -rf "data/raw/$cls"; tar xf "data/tars/$key" --force-local -C data/raw && touch "data/raw/$cls/.done" && rm "data/tars/$key" && echo "ok $key"
}
export -f one
xargs -P 3 -L 1 bash -c 'one "$0" "$1" "$2"' < <(tr -d '' < data/tars.txt)
ls data/raw/*/.done | wc -l
