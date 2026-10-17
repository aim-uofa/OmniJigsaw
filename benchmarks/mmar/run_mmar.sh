#!/bin/bash

# MMAR Benchmark Evaluation Suite

# Environment and version settings.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-"50000"}

# Resolve the result version.
VERSION=${1:-${VERSION:-"results"}}
BASE_BENCH_DIR="benchmarks/mmar"
SAVE_DIR="${BASE_BENCH_DIR}/${VERSION}"

# Create the result directory.
mkdir -p "${SAVE_DIR}"

# Shared benchmark parameters.
MODEL_PATH=${MODEL_PATH:-"./weights/Qwen3-Omni-30B-A3B-Instruct"}
BENCHMARK_DIR=${BENCHMARK_DIR_MMAR:-"./datasets/MMAR"}

# Define result files.
OUTPUT_JSON="${SAVE_DIR}/mmar_results.json"
RESULT_REPORT="${SAVE_DIR}/mmar_results.txt"

# Inference engine parameters.
TP_SIZE=${TP_SIZE:-1}
GPU_UTIL=${GPU_UTIL:-0.95}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}

# Logging and status helpers.
log_info() { echo -e "\033[1;34m[INFO]\033[0m $1"; }
log_succ() { echo -e "\033[1;32m[SUCCESS]\033[0m $1"; }
log_err()  { echo -e "\033[1;31m[ERROR]\033[0m $1"; }
log_task() { echo -e "\n\033[1;33m==== $1 ====\033[0m"; }

check_status() {
    if [ $? -ne 0 ]; then
        log_err "$1 failed. Terminating script."
        exit 1
    fi
}

# Evaluation workflow.

log_info "Model: ${MODEL_PATH}"
log_info "Save Directory: ${SAVE_DIR}"

# Step 1: run inference.
log_task "STEP 1: Running MMAR Inference"

python3 -m benchmarks.mmar.eval_mmar \
    --model_path "${MODEL_PATH}" \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --output_file "${OUTPUT_JSON}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --gpu_memory_utilization "${GPU_UTIL}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
check_status "MMAR Inference"

# Step 2: compute benchmark metrics.
log_task "STEP 2: Accuracy Calculation"

python3 "${BASE_BENCH_DIR}/get_mmar_accuracy.py" \
    --input "${OUTPUT_JSON}" \
    --output_txt "${RESULT_REPORT}"
check_status "MMAR Accuracy Calculation"

# Report generated artifacts.
log_succ "Full MMAR Evaluation Completed Successfully!"
log_info "Final Report: ${RESULT_REPORT}"
