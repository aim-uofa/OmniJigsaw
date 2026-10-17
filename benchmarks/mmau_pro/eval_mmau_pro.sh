#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=50000

MODEL_PATH="${MODEL_PATH:-./weights/Qwen3-Omni-30B-A3B-Instruct}"
BENCHMARK_DIR="${BENCHMARK_DIR_MMAU_PRO:-./datasets/MMAU-Pro}"
OUTPUT_FILE="benchmarks/mmau_pro/results_new/mmau_pro_results.parquet"

TP_SIZE=1
GPU_UTIL=0.95
MAX_NUM_SEQS=1
MAX_MODEL_LEN=65536

echo "Running MMAU-Pro Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data: $BENCHMARK_DIR"

python3 -m benchmarks.mmau_pro.eval_mmau_pro \
    --model_path $MODEL_PATH \
    --benchmark_dir $BENCHMARK_DIR \
    --output_file $OUTPUT_FILE \
    --tensor_parallel_size $TP_SIZE \
    --gpu_memory_utilization $GPU_UTIL \
    --max_num_seqs $MAX_NUM_SEQS \
    --max_model_len $MAX_MODEL_LEN
