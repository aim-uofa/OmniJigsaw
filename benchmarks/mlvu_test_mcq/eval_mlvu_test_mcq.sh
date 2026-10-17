#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=50000

MODEL_PATH="${MODEL_PATH:-./weights/Qwen3-Omni-30B-A3B-Instruct}"
BENCHMARK_DIR="${BENCHMARK_DIR_MLVU_TEST_MCQ:-./datasets/MLVU_Test}"
OUTPUT_JSON="benchmarks/mlvu_test_mcq/results_new/mlvu_test_results.json"

TP_SIZE=1
GPU_UTIL=0.95
MAX_NUM_SEQS=1
MAX_MODEL_LEN=65536

NUM_CLIPS=6

echo "Running MLVU Test (MCQ) Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data: $BENCHMARK_DIR"

python3 -m benchmarks.mlvu_test_mcq.eval_mlvu_test_mcq \
    --model_path $MODEL_PATH \
    --benchmark_dir $BENCHMARK_DIR \
    --output_file $OUTPUT_JSON \
    --num_clips $NUM_CLIPS \
    --tensor_parallel_size $TP_SIZE \
    --gpu_memory_utilization $GPU_UTIL \
    --max_num_seqs $MAX_NUM_SEQS \
    --max_model_len $MAX_MODEL_LEN
