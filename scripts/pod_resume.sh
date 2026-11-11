#!/usr/bin/env bash
# Resume after pretrain/frozen: priority order so the core results finish inside the budget cap.
# Finished fine-tuning runs are skipped automatically. Pod STOPS ITSELF when this exits.
set -uo pipefail
cd "$(dirname "$0")/.."
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
export PROFILE=low PYTHONUNBUFFERED=1 HF_HOME=/workspace/hf_cache
END=$(( $(date +%s) + ${MAX_MIN:-190} * 60 ))
left() { echo $(( END - $(date +%s) )); }
run() { local l; l=$(left); [ "$l" -gt 60 ] || return 0; timeout "$l" python run_all.py "$@"; echo "STEP_EXIT $? :: $*"; }
VITS=("ViT-S/16 (sup. IN)" "DINO ViT-S/16 (IN)" "DINO-generic (SEM)" "Ours: DINO-physics (SEM)")
FT_SEEDS=0 run --stages finetune --only "${VITS[@]}"
run --stages morph xai
FT_SEEDS=1,2 FT_BUDGETS=0.01,0.05,0.1 run --stages finetune --only "${VITS[@]}"
FT_SEEDS=0 run --stages finetune
echo "PIPELINE_EXIT 0"
