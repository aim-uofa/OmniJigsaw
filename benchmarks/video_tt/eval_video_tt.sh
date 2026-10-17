#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=50000

MODEL_PATH="${MODEL_PATH:-./weights/Qwen3-Omni-30B-A3B-Instruct}"
BENCHMARK_DIR="${BENCHMARK_DIR_VIDEO_TT:-./datasets/Video-TT}"
OUTPUT_JSON="benchmarks/video_tt/results/video_tt_results.json"

TP_SIZE=1
GPU_UTIL=0.95
MAX_NUM_SEQS=1
MAX_MODEL_LEN=65536
NUM_CLIPS=6

echo "Running Video-TT Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data: $BENCHMARK_DIR"

python3 -m benchmarks.video_tt.eval_video_tt \
    --model_path $MODEL_PATH \
    --benchmark_dir $BENCHMARK_DIR \
    --output_file $OUTPUT_JSON \
    --num_clips $NUM_CLIPS \
    --tensor_parallel_size $TP_SIZE \
    --gpu_memory_utilization $GPU_UTIL \
    --max_num_seqs $MAX_NUM_SEQS \
    --max_model_len $MAX_MODEL_LEN
