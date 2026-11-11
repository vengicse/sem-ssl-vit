#!/usr/bin/env bash
# Stop this RunPod pod from inside (GPU billing ends; disk is kept).
sleep "${STOP_DELAY:-90}"   # grace period so results can be fetched if someone is watching
if command -v runpodctl >/dev/null; then
    runpodctl stop pod "$RUNPOD_POD_ID"
else
    curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
fi
