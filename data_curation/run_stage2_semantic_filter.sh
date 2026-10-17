#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-./weights/Qwen2.5-VL-7B-Instruct}
INPUT_DIR=${INPUT_DIR:-./datasets/omnijigsaw_stage1}
RESULTS_FILE=${RESULTS_FILE:-./outputs/data_curation/stage2_semantic_results.json}

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || {
    echo "Error: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-50000}

GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
SEED=${SEED:-1234}
NFRAMES=${NFRAMES:-200}
MAX_PIXELS=${MAX_PIXELS:-100352}
MAX_TOKENS=${MAX_TOKENS:-2048}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.05}

echo "Starting OmniJigsaw stage-2 semantic filtering"
echo "Model: ${MODEL_PATH}"
echo "Input directory: ${INPUT_DIR}"
echo "Results file: ${RESULTS_FILE}"

"${PYTHON_BIN}" data_curation/stage2_semantic_filter.py \
    --model-path "${MODEL_PATH}" \
    --input-dir "${INPUT_DIR}" \
    --results-file "${RESULTS_FILE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --seed "${SEED}" \
    --nframes "${NFRAMES}" \
    --max-pixels "${MAX_PIXELS}" \
    --max-tokens "${MAX_TOKENS}" \
    --repetition-penalty "${REPETITION_PENALTY}"
