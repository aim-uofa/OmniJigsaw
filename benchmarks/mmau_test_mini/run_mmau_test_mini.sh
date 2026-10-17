#!/bin/bash

# MMAU test_mini Benchmark Evaluation Suite

# Environment and version settings.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-"50000"}

VERSION=${1:-${VERSION:-"results"}}
BASE_BENCH_DIR="benchmarks/mmau_test_mini"
SAVE_DIR="${BASE_BENCH_DIR}/${VERSION}"

mkdir -p "${SAVE_DIR}"

# Shared benchmark parameters.
MODEL_PATH=${MODEL_PATH:-"./weights/Qwen3-Omni-30B-A3B-Instruct"}
BENCHMARK_DIR=${BENCHMARK_DIR_MMAU_TEST_MINI:-"./datasets/MMAU_test_mini"}

# Define result files.
OUTPUT_JSON="${SAVE_DIR}/mmau_test_mini_results.json"
RESULT_REPORT="${SAVE_DIR}/mmau_test_mini_results.txt"

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
log_task "STEP 1: Running MMAU test_mini Inference"

python3 -m benchmarks.mmau_test_mini.eval_mmau_test_mini \
    --model_path "${MODEL_PATH}" \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --output_file "${OUTPUT_JSON}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --gpu_memory_utilization "${GPU_UTIL}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
check_status "MMAU test_mini Inference"

# Step 2: compute benchmark metrics.
log_task "STEP 2: Accuracy Calculation"

python3 "${BASE_BENCH_DIR}/get_mmau_test_mini_accuracy.py" \
    --input "${OUTPUT_JSON}" \
    --output_txt "${RESULT_REPORT}"
check_status "MMAU test_mini Accuracy Calculation"

# Report generated artifacts.
log_succ "Full MMAU test_mini Evaluation Completed Successfully!"
log_info "Final Report: ${RESULT_REPORT}"
