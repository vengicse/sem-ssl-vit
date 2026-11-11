#!/usr/bin/env bash
# Wait for the pod run to finish, fetch results, then let the pod stop itself.
T=$1
cd "$(dirname "$0")/.."
while true; do
    s=$(timeout 120 bash scripts/pod_cmd.sh "$T" '[ -f /workspace/RUN_FINISHED ] && echo RUN_IS_""COMPLETE; tail -2 /workspace/sem-ssl-vit/logs/run.log' | tr -d '\r')
    echo "$(date +%H:%M) $(echo "$s" | grep -a "ep \|stage\|frozen\|finetune\|Error\|PIPELINE" | tail -1)"
    echo "$s" | grep -q "^RUN_IS_COMPLETE" && break
    if [ -z "$s" ]; then echo "pod unreachable"; fi
    sleep 600
done
ok=0
for i in 1 2 3; do timeout 3600 bash scripts/fetch_via_proxy.sh "$T" && { ok=1; break; }; sleep 60; done
# Only let the pod stop once results are safely local (container disk is wiped on stop).
if [ $ok = 1 ]; then timeout 120 bash scripts/pod_cmd.sh "$T" 'touch /workspace/FETCHED; echo marked'; echo FETCH_OK; else echo FETCH_FAILED; fi
