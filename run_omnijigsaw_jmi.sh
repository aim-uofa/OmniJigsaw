#!/usr/bin/env bash

set -eo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

WORLD_SIZE=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PROJECT_NAME="omnijigsaw"
export EXPERIMENT_NAME="jmi_qwen3_omni"
export PYDEVD_DISABLE_FILE_VALIDATION=1

SAVE_CHECKPOINT_DIR=${SAVE_CHECKPOINT_DIR:-./weights/jmi_qwen3_omni}
ROLLOUT_DATA_DIR=omnijigsaw/jmi/rollout_data_jmi_qwen3_omni
VALIDATION_DATA_DIR=omnijigsaw/jmi/validate_res_jmi_qwen3_omni

mkdir -p ${SAVE_CHECKPOINT_DIR}
mkdir -p ${SAVE_CHECKPOINT_DIR}/${EXPERIMENT_NAME}

DATASET_TRAIN=data/jmi_train.parquet
DATASET_TEST=data/jmi_test.parquet
DATA_FOLDER=${DATA_FOLDER:-./datasets/omni_jigsaw}

REF_MODEL_PATH=${REF_MODEL_PATH:-./weights/Qwen3-Omni-30B-A3B-Instruct}

DATASET_CLS_PATH='pkg://omnijigsaw.jmi.dataset.omnijigsaw_jmi_dataset'
DATASET_CLS_NAME='JMIDataset'

REWARD_FUNC_PATH='reward_fn/custom_reward_entry.py'
REWARD_FUNC_NAME='compute_score'

REWARD_FORMAT_SCORE=0.2
REWARD_GLOBAL_POS_WEIGHT=0.5
REWARD_CONTINUITY_WEIGHT=0.5

FREEZE_VISION_TOWER=True
FREEZE_AUDIO_TOWER=True
FREEZE_ROUTER=True

python3 -m omnijigsaw.jmi.main \
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
    data.custom_cls.path=${DATASET_CLS_PATH} \
    data.custom_cls.name=${DATASET_CLS_NAME} \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.use_audio_in_video=True \
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
