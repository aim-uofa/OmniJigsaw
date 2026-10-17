# OmniJigsaw Data Curation

This directory contains the two-stage puzzle-quality curation pipeline used to
prepare videos for OmniJigsaw post-training.

| Stage | Entry point | Output |
| --- | --- | --- |
| Heuristic filtering | `run_stage1_heuristic_filter.sh` | Retained videos standardized as H.264/AAC MP4 files |
| Semantic filtering | `run_stage2_semantic_filter.sh` | Qwen2.5-VL judgments in JSON format |

## Environment

Data curation uses Qwen2.5-VL-7B-Instruct and should run in an environment
separate from the OmniJigsaw post-training environment.

Use Python 3.10 or later. Create an isolated environment, then install PyTorch
and vLLM builds compatible with the CUDA toolkit and NVIDIA driver on the target
machine. Install the remaining dependencies after those two packages:

```bash
CURATION_VENV=/path/to/omnijigsaw-curation-env
python3 -m venv "${CURATION_VENV}"
source "${CURATION_VENV}/bin/activate"
python -m pip install --upgrade pip

# Install CUDA-compatible PyTorch and vLLM builds before these packages.
python -m pip install transformers qwen-vl-utils decord av
python -m pip install numpy librosa opencv-python-headless ffmpeg-python tqdm
```

The host system must also provide `ffmpeg` and `ffprobe` on `PATH`:

```bash
ffmpeg -version
ffprobe -version
```

## Stage 1: Heuristic Filtering

The first stage recursively checks video-audio stream integrity, duration,
visual dynamics, silence ratio, spectral-flux variance, and speech ratio. Every
retained sample is written as an H.264/AAC MP4 file while preserving its path
relative to the input directory.

Run from the repository root:

```bash
INPUT_DIR=./datasets/raw_videos \
OUTPUT_DIR=./datasets/omnijigsaw_stage1 \
DEVICE=auto \
./data_curation/run_stage1_heuristic_filter.sh
```

By default, Silero VAD is loaded through PyTorch Hub. For an offline run, point
`SILERO_REPO` to a local checkout:

```bash
INPUT_DIR=./datasets/raw_videos \
OUTPUT_DIR=./datasets/omnijigsaw_stage1 \
SILERO_REPO=/path/to/silero-vad \
./data_curation/run_stage1_heuristic_filter.sh
```

The launcher uses the following default settings: a 200-second duration limit,
1-second visual sampling interval, MAD threshold of 5.0, maximum static and
silence ratios of 0.70, 16 kHz audio, -40 dB RMS threshold, minimum
spectral-flux variance of 0.5, and a VAD speech-ratio range of 0.30--0.80. Each
setting can be overridden through the correspondingly named environment
variable in the launcher.

## Stage 2: Semantic Filtering

The second stage uses Qwen2.5-VL-7B-Instruct to determine whether each retained
video has an unambiguous chronological progression suitable for temporal
reordering.

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=./weights/Qwen2.5-VL-7B-Instruct \
INPUT_DIR=./datasets/omnijigsaw_stage1 \
RESULTS_FILE=./outputs/data_curation/stage2_semantic_results.json \
./data_curation/run_stage2_semantic_filter.sh
```

For tensor-parallel inference, expose the required GPUs and set the matching
parallel size, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TENSOR_PARALLEL_SIZE=2 \
MODEL_PATH=./weights/Qwen2.5-VL-7B-Instruct \
INPUT_DIR=./datasets/omnijigsaw_stage1 \
RESULTS_FILE=./outputs/data_curation/stage2_semantic_results.json \
./data_curation/run_stage2_semantic_filter.sh
```

The default inference configuration uses 200 sampled frames, 100,352 maximum
pixels, greedy decoding, 2,048 maximum new tokens, a 1.05 repetition penalty,
and a 32,768-token context limit.

A video is accepted only when the parsed answer is exactly `YES`; the derived
`accepted` field makes this decision explicit.
