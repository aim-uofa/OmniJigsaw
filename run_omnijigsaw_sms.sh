#!/usr/bin/env bash

set -eo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

WORLD_SIZE=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PROJECT_NAME="omnijigsaw"
export EXPERIMENT_NAME="sms_qwen3_omni"
export PYDEVD_DISABLE_FILE_VALIDATION=1

SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-./weights/sms_qwen3_omni}
ROLLOUT_DATA_DIR=omnijigsaw/sms/rollout_data_sms_qwen3_omni
VALIDATION_DATA_DIR=omnijigsaw/sms/validate_res_sms_qwen3_omni

mkdir -p ${SAVE_CHECKPOINT_DIR}
mkdir -p ${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME}

DATASET_TRAIN=data/sms_train.parquet
DATASET_TEST=data/sms_test.parquet
DATA_FOLDER=${DATA_FOLDER:-./datasets/omni_jigsaw}

REF_MODEL_PATH=${REF_MODEL_PATH:-./weights/Qwen3-Omni-30B-A3B-Instruct}

STAGE1_MAX_FRAMES=80
STAGE1_SAMPLE_FPS=1.0
STAGE1_TEMPERATURE=0.0
STAGE1_TOP_P=1.0
STAGE1_TOP_K=-1
STAGE1_REPETITION_PENALTY=1.05
STAGE1_MAX_TOKENS=2048

STAGE2_MAX_AUDIO_SECONDS=36.0
STAGE2_SAMPLING_FPS=2.0

REWARD_FORMAT_SCORE=0.2
REWARD_GLOBAL_POS_WEIGHT=0.5
REWARD_CONTINUITY_WEIGHT=0.5

# Train with Sample-level Modality Selection (SMS), validate with video-only jigsaw data.
TRAIN_DATASET_CLS_PATH='pkg://omnijigsaw.sms.dataset.omnijigsaw_sms_dataset'
TRAIN_DATASET_CLS_NAME='SMSDataset'
VAL_DATASET_CLS_PATH='pkg://omnijigsaw.sms.dataset.video_jigsaw_dataset'
VAL_DATASET_CLS_NAME='VideoJigsawDataset'

REWARD_FUNC_PATH='reward_fn/custom_reward_entry.py'
REWARD_FUNC_NAME='compute_score'

FREEZE_VISION_TOWER=True
FREEZE_AUDIO_TOWER=True
FREEZE_ROUTER=True

python3 -m omnijigsaw.sms.main \
    +ray_init.num_cpus=80 \
    algorithm.adv_estimator=grpo \
    data.dataloader_num_workers=16 \
    data.train_files=[${DATASET_TRAIN}] \
    data.val_files=[${DATASET_TEST}] \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    +data.multimodal_folder=${DATA_FOLDER} \
    +data.custom_cls_train.path=${TRAIN_DATASET_CLS_PATH} \
    +data.custom_cls_train.name=${TRAIN_DATASET_CLS_NAME} \
    +data.custom_cls_val.path=${VAL_DATASET_CLS_PATH} \
    +data.custom_cls_val.name=${VAL_DATASET_CLS_NAME} \
    +data.stage1_max_frames=${STAGE1_MAX_FRAMES} \
    +data.stage1_sample_fps=${STAGE1_SAMPLE_FPS} \
    +data.stage1_do_sample=False \
    +data.stage1_temperature=${STAGE1_TEMPERATURE} \
    +data.stage1_top_p=${STAGE1_TOP_P} \
    +data.stage1_top_k=${STAGE1_TOP_K} \
    +data.stage1_repetition_penalty=${STAGE1_REPETITION_PENALTY} \
    +data.stage1_max_tokens=${STAGE1_MAX_TOKENS} \
    +data.stage2_max_audio_seconds=${STAGE2_MAX_AUDIO_SECONDS} \
    +data.stage2_sampling_fps=${STAGE2_SAMPLING_FPS} \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.use_audio_in_video=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=60000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.freeze_vision_tower=${FREEZE_VISION_TOWER} \
    +actor_rollout_ref.actor.freeze_audio_tower=${FREEZE_AUDIO_TOWER} \
    actor_rollout_ref.actor.freeze_router=${FREEZE_ROUTER} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=80000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.video=6 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.audio=6 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=196608 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.max_num_seqs=128 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=80000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model'] \
    custom_reward_function.path=${REWARD_FUNC_PATH} \
    custom_reward_function.name=${REWARD_FUNC_NAME} \
    +custom_reward_function.reward_kwargs.format_score=${REWARD_FORMAT_SCORE} \
    +custom_reward_function.reward_kwargs.global_pos_weight=${REWARD_GLOBAL_POS_WEIGHT} \
    +custom_reward_function.reward_kwargs.continuity_weight=${REWARD_CONTINUITY_WEIGHT} \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.max_actor_ckpt_to_keep=20 \
    trainer.test_freq=100 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME} \
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR} \
    trainer.validation_data_dir=${VALIDATION_DATA_DIR} \
    trainer.total_training_steps=1000 \
    trainer.total_epochs=10 2>&1 | tee ${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME}/logs.log
