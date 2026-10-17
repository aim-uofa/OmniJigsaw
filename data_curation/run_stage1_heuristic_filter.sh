#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

command -v ffmpeg >/dev/null 2>&1 || {
    echo "Error: ffmpeg is required but was not found in PATH." >&2
    exit 1
}
command -v ffprobe >/dev/null 2>&1 || {
    echo "Error: ffprobe is required but was not found in PATH." >&2
    exit 1
}

PYTHON_BIN=${PYTHON_BIN:-python3}
INPUT_DIR=${INPUT_DIR:-./datasets/raw_videos}
OUTPUT_DIR=${OUTPUT_DIR:-./datasets/omnijigsaw_stage1}
SILERO_REPO=${SILERO_REPO:-}
DEVICE=${DEVICE:-auto}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "Error: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
}

MAX_DURATION_SEC=${MAX_DURATION_SEC:-200}
TARGET_SAMPLE_RATE=${TARGET_SAMPLE_RATE:-16000}
RMS_THRESHOLD_DB=${RMS_THRESHOLD_DB:--40}
MAX_SILENCE_RATIO=${MAX_SILENCE_RATIO:-0.70}
MIN_FLUX_VARIANCE=${MIN_FLUX_VARIANCE:-0.5}
VAD_MIN_RATIO=${VAD_MIN_RATIO:-0.30}
VAD_MAX_RATIO=${VAD_MAX_RATIO:-0.80}
VISUAL_SAMPLE_INTERVAL=${VISUAL_SAMPLE_INTERVAL:-1.0}
VISUAL_DIFF_THRESHOLD=${VISUAL_DIFF_THRESHOLD:-5.0}
MAX_STATIC_RATIO=${MAX_STATIC_RATIO:-0.70}

args=(
    --input-dir "${INPUT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --device "${DEVICE}"
    --max-duration-sec "${MAX_DURATION_SEC}"
    --target-sample-rate "${TARGET_SAMPLE_RATE}"
    --rms-threshold-db "${RMS_THRESHOLD_DB}"
    --max-silence-ratio "${MAX_SILENCE_RATIO}"
    --min-flux-variance "${MIN_FLUX_VARIANCE}"
    --vad-min-ratio "${VAD_MIN_RATIO}"
    --vad-max-ratio "${VAD_MAX_RATIO}"
    --visual-sample-interval "${VISUAL_SAMPLE_INTERVAL}"
    --visual-diff-threshold "${VISUAL_DIFF_THRESHOLD}"
    --max-static-ratio "${MAX_STATIC_RATIO}"
)

if [[ -n "${SILERO_REPO}" ]]; then
    args+=(--silero-repo "${SILERO_REPO}")
fi

echo "Starting OmniJigsaw stage-1 heuristic filtering"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
"${PYTHON_BIN}" data_curation/stage1_heuristic_filter.py "${args[@]}"
