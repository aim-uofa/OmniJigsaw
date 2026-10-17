#!/bin/bash

# MLVU Multiple-Choice Evaluation Suite

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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export DECORD_EOF_RETRY_MAX=${DECORD_EOF_RETRY_MAX:-"50000"}

VERSION=${1:-${VERSION:-"results"}}
BASE_BENCH_DIR="benchmarks/mlvu"
SAVE_DIR="${BASE_BENCH_DIR}/${VERSION}"
mkdir -p "${SAVE_DIR}"

MODEL_PATH=${MODEL_PATH:-"./weights/Qwen3-Omni-30B-A3B-Instruct"}
BENCHMARK_DIR=${BENCHMARK_DIR_MLVU:-"./datasets/MLVU"}

TP_SIZE=${TP_SIZE:-1}
GPU_UTIL=${GPU_UTIL:-0.95}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
NUM_CLIPS=${NUM_CLIPS:-6}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
DISABLE_AUDIO=${DISABLE_AUDIO:-"false"}
AUDIO_FLAG=""
if [ "$DISABLE_AUDIO" = "true" ]; then
    AUDIO_FLAG="--disable_audio"
    log_info "Audio is DISABLED for this run."
else
    log_info "Audio is ENABLED for this run."
fi

MCQ_JSON="${SAVE_DIR}/mlvu_results_mcq.json"
MCQ_REPORT="${SAVE_DIR}/mlvu_results_mcq.txt"

log_info "Model: ${MODEL_PATH}"
log_info "Save Directory: ${SAVE_DIR}"
log_task "MLVU Multiple-Choice Inference"

python3 -m benchmarks.mlvu.eval_mlvu \
    --model_path "${MODEL_PATH}" \
    --benchmark_dir "${BENCHMARK_DIR}" \
    --output_file "${MCQ_JSON}" \
    --num_clips "${NUM_CLIPS}" \
    --tensor_parallel_size "${TP_SIZE}" \
    --gpu_memory_utilization "${GPU_UTIL}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    ${AUDIO_FLAG}
check_status "MLVU MCQ Inference"

python3 "${BASE_BENCH_DIR}/get_mlvu_mcq_accuracy.py" \
    --input "${MCQ_JSON}" \
    --output_txt "${MCQ_REPORT}"
check_status "MLVU MCQ Accuracy Calculation"

log_succ "MLVU MCQ Evaluation Completed Successfully!"
log_info "Final Report: ${MCQ_REPORT}"
