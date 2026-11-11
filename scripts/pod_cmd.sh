#!/usr/bin/env bash
# Run commands on a RunPod pod through the SSH proxy (no direct TCP needed).
# Usage: bash scripts/pod_cmd.sh <pod_id-userhash> 'command; command'
T=$1; shift
printf '%s\nexit\n' "stty -echo; $*" | timeout ${POD_CMD_TIMEOUT:-300} ssh -tt -i ~/.ssh/id_ed25519_runpod \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=30 "$T@ssh.runpod.io" 2>/dev/null \
  | sed -e 's/\x1b\[[0-9;?]*[a-zA-Z]//g' -e 's/\x1b\][^\x07]*\x07//g' | grep -av '^root@\|^exit$\|Connection to'
