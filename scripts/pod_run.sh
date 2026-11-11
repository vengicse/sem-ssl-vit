#!/usr/bin/env bash
# Full pipeline on a RunPod GPU pod. The pod STOPS ITSELF when this script exits.
#   cd /workspace/sem-ssl-vit && nohup bash scripts/pod_run.sh > logs/run.log 2>&1 &
# Results are packed into /workspace/results_sem.tgz. Because the container disk is erased on
# stop, the script waits (max FETCH_WAIT_H hours) for the laptop to fetch the archive and
# create /workspace/FETCHED, then stops the pod either way.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
finish() {
    tar czf /workspace/results_sem.tgz results logs ckpt/dino_physics/teacher_backbone.pt 2>/dev/null
    touch /workspace/RUN_FINISHED
    for _ in $(seq 1 $(( ${FETCH_WAIT_H:-3} * 60 ))); do
        [ -f /workspace/FETCHED ] && break
        sleep 60
    done
    STOP_DELAY=5 bash scripts/pod_stop_self.sh
}
trap finish EXIT
export DATA_DIR=/workspace/data CKPT_DIR=/workspace/sem-ssl-vit/ckpt RESULTS_DIR=/workspace/sem-ssl-vit/results
export PROFILE=${PROFILE:-low} PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1 HF_HOME=/workspace/hf_cache
pip install -q -r requirements.txt
nvidia-smi --query-gpu=name,memory.total --format=csv
STAGES="${*:-data pretrain frozen finetune morph xai}"
timeout ${MAX_HOURS:-12}h python run_all.py --stages $STAGES   # hard wall-clock cap on spend
echo "PIPELINE_EXIT $?"
