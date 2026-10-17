#!/bin/bash

set -eo pipefail

# Shared defaults; callers may override any value through the environment.
export VERSION=${1:-"results"}
export MODEL_PATH="${MODEL_PATH:-./weights/Qwen3-Omni-30B-A3B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-50000}"

export TP_SIZE="${TP_SIZE:-1}"
export GPU_UTIL="${GPU_UTIL:-0.95}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export NUM_CLIPS="${NUM_CLIPS:-6}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export DISABLE_AUDIO="${DISABLE_AUDIO:-false}"

# Dataset placeholders follow the expected on-disk directory names.
export BENCHMARK_DIR_MLVU="${BENCHMARK_DIR_MLVU:-./datasets/MLVU}"
export BENCHMARK_DIR_MLVU_TEST_MCQ="${BENCHMARK_DIR_MLVU_TEST_MCQ:-./datasets/MLVU_Test}"
export BENCHMARK_DIR_MMAU_PRO="${BENCHMARK_DIR_MMAU_PRO:-./datasets/MMAU-Pro}"
export BENCHMARK_DIR_MMAU_TEST_MINI="${BENCHMARK_DIR_MMAU_TEST_MINI:-./datasets/MMAU_test_mini}"
export BENCHMARK_DIR_MMSU="${BENCHMARK_DIR_MMSU:-./datasets/MMSU}"
export BENCHMARK_DIR_VIDEOMME="${BENCHMARK_DIR_VIDEOMME:-./datasets/Video-MME}"
export BENCHMARK_DIR_MMAR="${BENCHMARK_DIR_MMAR:-./datasets/MMAR}"
export BENCHMARK_DIR_VIDEO_HOLMES="${BENCHMARK_DIR_VIDEO_HOLMES:-./datasets/Video-Holmes}"
export BENCHMARK_DIR_AOT="${BENCHMARK_DIR_AOT:-./datasets/AoTBench}"
export BENCHMARK_DIR_TEMPCOMPASS="${BENCHMARK_DIR_TEMPCOMPASS:-./datasets/TempCompass}"
export BENCHMARK_DIR_TUNA="${BENCHMARK_DIR_TUNA:-./datasets/TUNA-Bench}"
export BENCHMARK_DIR_VIDEO_TT="${BENCHMARK_DIR_VIDEO_TT:-./datasets/Video-TT}"


# Per-benchmark overrides: benchmark_name MAX_MODEL_LEN BATCH_SIZE.
declare -A BENCH_CONFIGS=(
    ["mlvu"]="32768 4"
    ["mlvu_test_mcq"]="32768 4"
    ["video_mme"]="32768 4"
    ["mmau_pro"]="65536 2"
    ["mmau_test_mini"]="65536 2"
    ["mmsu"]="65536 2"
    ["mmar"]="65536 2"
    ["video_holmes"]="32768 4"
    ["aot_bench"]="32768 4"
    ["tempcompass"]="32768 4"
    ["tuna_bench"]="32768 4"
    ["video_tt"]="32768 4"
)

log_header() { echo -e "\n\033[1;44m >>> FINAL CONTROL: STARTING EVALUATION [VERSION: $VERSION] <<< \033[0m"; }
log_item() { echo -e "\033[1;36m > $1\033[0m"; }

run_benchmark() {
    local bench_name=$1
    local bench_script_path="benchmarks/${bench_name}/run_${bench_name}.sh"

    if [[ -n "${BENCH_CONFIGS[$bench_name]}" ]]; then
        read -r custom_max_model_len custom_batch_size <<< "${BENCH_CONFIGS[$bench_name]}"

        log_item "Starting ${bench_name} (MAX_MODEL_LEN=${custom_max_model_len}, BATCH_SIZE=${custom_batch_size})..."

        # Scope overrides to this benchmark invocation.
        MAX_MODEL_LEN=$custom_max_model_len \
        BATCH_SIZE=$custom_batch_size \
        bash "$bench_script_path" "$VERSION"
    else
        log_item "Starting ${bench_name} (using global defaults: MAX_MODEL_LEN=${MAX_MODEL_LEN}, BATCH_SIZE=${BATCH_SIZE})..."
        bash "$bench_script_path" "$VERSION"
    fi
}

# The optional second argument selects a comma-separated benchmark subset.
SELECTED_BENCHS=${2:-"mlvu,mlvu_test_mcq,mmau_pro,mmau_test_mini,mmsu,video_mme,mmar,video_holmes,aot_bench,tempcompass,tuna_bench,video_tt"}

log_header
log_item "Selected Benchmarks: $SELECTED_BENCHS"
log_item "Model Path: $MODEL_PATH"
log_item "GPU Units: $CUDA_VISIBLE_DEVICES"
log_item "Global Defaults: MAX_MODEL_LEN=${MAX_MODEL_LEN}, BATCH_SIZE=${BATCH_SIZE}"

IFS=',' read -ra ADDR <<< "$SELECTED_BENCHS"
for bench in "${ADDR[@]}"; do
    case $bench in
        "mlvu")
            run_benchmark "mlvu"
            ;;
        "mlvu_test_mcq")
            run_benchmark "mlvu_test_mcq"
            ;;
        "mmau_pro")
            run_benchmark "mmau_pro"
            ;;
        "mmau_test_mini")
            run_benchmark "mmau_test_mini"
            ;;
        "mmsu")
            run_benchmark "mmsu"
            ;;
        "video_mme")
            run_benchmark "video_mme"
            ;;
        "mmar")
            run_benchmark "mmar"
            ;;
        "video_holmes")
            run_benchmark "video_holmes"
            ;;
        "aot_bench")
            run_benchmark "aot_bench"
            ;;
        "tempcompass")
            run_benchmark "tempcompass"
            ;;
        "tuna_bench")
            run_benchmark "tuna_bench"
            ;;
        "video_tt")
            run_benchmark "video_tt"
            ;;
        *)
            echo -e "\033[1;31m[ERROR]\033[0m Unknown benchmark: $bench" >&2
            exit 2
            ;;
    esac
done

echo -e "\n\033[1;32m[ALL FINISHED] Evaluation for version '$VERSION' completed.\033[0m"
