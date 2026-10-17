#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=50000

MODEL_PATH="${MODEL_PATH:-./weights/Qwen3-Omni-30B-A3B-Instruct}"
BENCHMARK_DIR="${BENCHMARK_DIR_MMAU_TEST_MINI:-./datasets/MMAU_test_mini}"
OUTPUT_JSON="benchmarks/mmau_test_mini/results_new/mmau_test_mini_results.json"

TP_SIZE=1
GPU_UTIL=0.95
MAX_NUM_SEQS=1
MAX_MODEL_LEN=65536

echo "Running MMAU test_mini Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data: $BENCHMARK_DIR"

python3 -m benchmarks.mmau_test_mini.eval_mmau_test_mini \
    --model_path $MODEL_PATH \
    --benchmark_dir $BENCHMARK_DIR \
    --output_file $OUTPUT_JSON \
    --tensor_parallel_size $TP_SIZE \
    --gpu_memory_utilization $GPU_UTIL \
    --max_num_seqs $MAX_NUM_SEQS \
    --max_model_len $MAX_MODEL_LEN
