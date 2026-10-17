import json
import os
import random
import re
import uuid
from copy import deepcopy
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
import verl.utils.torch_functional as verl_F
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.single_controller.ray import RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


STAGE2_INSTRUCTIONS_TEMPLATE = """You are given {num_clips} **shuffled** video clips that were created by extracting key temporal segments from one original video.

**Important Note on Modalities:** To test your cross-modal understanding, some clips may have missing audio (the audio is silent) or missing video (the video frames are entirely black). Use the non-missing and informative modalities in each clip to identify its content.

Here are the clips, each tagged with an index reflecting the current (shuffled) order in which they are shown. Please pay attention to both the visual actions and the specific content of the audio.

{clip_list}

Task:
1. **For each clip**, describe its specific visual events and audio features (e.g., distinct speech phrases or sound patterns). If a modality is missing, focus on the available one.
2. Analyze the temporal logic by comparing these features to determine how the story or action progresses from one segment to another.
3. Based on this content-driven analysis, reassemble the original video and output the indices in correct order, separated by commas.

Answer format example:
2, 3, 1, 4, 6, 5

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. **Ensure your reasoning is strictly based on visual facts and auditory clues, following a clear logical progression and avoiding unrelated digressions.** The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE enclosed within <answer> </answer> tags."""


STAGE2_CROSS_MODALITY_INSTRUCTIONS_TEMPLATE = """You are given {num_clips} **shuffled** clips that were created by slicing one original video into {num_clips} equal-length temporal segments.

**Important Note on Modalities:** To test your cross-modal understanding, some clips may have missing audio (the audio is silent) or missing video (the video frames are entirely black). Use the non-missing and informative modalities in each clip to identify its content.

Here are the clips, each tagged with an index reflecting the current (shuffled) order in which they are shown. Please pay attention to both the visual facts and auditory clues.

{clip_list}

Task:
1. **For each clip**, describe its specific visual events and audio features (e.g., distinct speech phrases or sound patterns). If a modality is missing, focus on the available one. **For clip connections**, identify subtle transition clues (e.g., action/state continuity, unfinished sentences, background sounds) and analyze the causal, logical, or emotional progression.
2. Analyze the temporal logic by integrating individual clip content with inter-clip relationships, explicitly chaining together both specific continuity details and thematic progression.
3. Based on this evidence-driven analysis, reassemble the original video and output the indices in correct order, separated by commas.

Answer format example:
2, 3, 1, 4, 6, 5

You **FIRST** think about the reasoning process as an internal monologue and **THEN** provide the final answer. **Ensure your reasoning is strictly based on clip content features, inter-clip relationships, and specific connection evidence, avoiding unrelated digressions.** The reasoning process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE enclosed within <answer> </answer> tags."""


class OmniPPOTrainer(RayPPOTrainer):
    
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        
        
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Create the colocated actor and rollout worker.
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # Create the critic when required by the advantage estimator.
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # Create a standalone reference policy when required.
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # Create an optional learned reward-model worker.
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # Roles sharing a resource pool are initialized as colocated workers.
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        # Create colocated workers for each configured role.
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # Initialize rollout last so vLLM can estimate available KV-cache memory.
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # Create the asynchronous rollout manager when configured.
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _is_two_stage_cross_modality_jigsaw_batch(self, batch: DataProto) -> bool:
        """Detect whether the current batch uses the two-stage CMM pipeline."""
        extra_infos = batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is None or len(extra_infos) == 0:
            return False
        first = extra_infos[0]
        return isinstance(first, dict) and bool(first.get("cross_modality_jigsaw", False))

    def _extract_tag(self, text: str, tag: str) -> str | None:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
        if not m:
            return None
        return m.group(1).strip()

    def _decode_responses_to_text(self, gen_out: DataProto) -> list[str]:
        # Responses are padded; skip_special_tokens drops PAD/EOS, leaving clean text.
        resp = gen_out.batch["responses"]
        texts = self.tokenizer.batch_decode(resp, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _parse_stage1_modality_selection(self, text: str, num_clips: int) -> dict:
        """Parse stage-1 clip-level modality selections for CMM.

        Expected JSON (inside <answer>...</answer>):
            {"modalities": ["V", "A", "VA", ...]}
        """
        answer = self._extract_tag(text, "answer")
        if answer is None:
            answer = text.strip()

        if answer.startswith("```"):
            answer = re.sub(r"^```[a-zA-Z0-9]*\n?", "", answer)
            answer = re.sub(r"\n?```$", "", answer).strip()

        parsed_obj = None
        parse_ok = False
        err = None
        try:
            parsed_obj = json.loads(answer)
            parse_ok = True
        except Exception as e:
            try:
                l = answer.find("{")
                r = answer.rfind("}")
                if l != -1 and r != -1 and r > l:
                    parsed_obj = json.loads(answer[l : r + 1])
                    parse_ok = True
                else:
                    err = f"json_parse_failed: {e}"
            except Exception as e2:
                err = f"json_parse_failed: {e2}"

        if not parse_ok:
            return {"parse_ok": False, "modalities": [], "error": err or "json_parse_failed"}

        modalities_raw = None
        if isinstance(parsed_obj, dict) and "modalities" in parsed_obj:
            modalities_raw = parsed_obj.get("modalities")
        elif isinstance(parsed_obj, list):
            modalities_raw = parsed_obj

        if not isinstance(modalities_raw, list):
            return {"parse_ok": False, "modalities": [], "error": "modalities_not_a_list"}

        modalities = []
        for item in modalities_raw:
            mod = str(item).strip().upper()
            if mod not in {"V", "A", "VA"}:
                return {"parse_ok": False, "modalities": [], "error": "invalid_modality"}
            modalities.append(mod)

        if len(modalities) != int(num_clips):
            return {"parse_ok": False, "modalities": [], "error": "modality_len_mismatch"}

        return {"parse_ok": True, "modalities": modalities, "error": None}

    def _uniform_time_ranges(self, duration_sec: float, clip_num: int, trim_ratio: float = 0.1) -> list[tuple[float, float]]:
        clip_num = int(max(1, clip_num))
        seg_len = float(duration_sec) / float(clip_num)
        ranges = []
        for i in range(clip_num):
            start = i * seg_len
            end = (i + 1) * seg_len
            trim = seg_len * float(trim_ratio)
            start = start + trim / 2.0
            end = end - trim / 2.0
            if end <= start:
                end = min(duration_sec, start + 0.1)
            ranges.append((start, end))
        return ranges

    def _build_stage2_jigsaw_messages(self, num_clips: int) -> list[dict]:
        num_clips = int(num_clips)

        clip_list_rows = [f"Clip {i}: <video>" for i in range(1, num_clips + 1)]
        instructions = STAGE2_INSTRUCTIONS_TEMPLATE.format(num_clips=num_clips, clip_list="\n".join(clip_list_rows))

        content_list = []
        segments = re.split("(<image>|<video>|<audio>)", instructions)
        segments = [item for item in segments if item != ""]
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image"})
            elif segment == "<video>":
                content_list.append({"type": "video"})
            elif segment == "<audio>":
                content_list.append({"type": "audio"})
            else:
                content_list.append({"type": "text", "text": segment})

        return [{"role": "user", "content": content_list}]

    def _build_stage2_cross_modality_messages(self, num_clips: int) -> list[dict]:
        """Build stage-2 prompt with <video> placeholders only."""
        clip_rows = [f"Clip {i}: <video>" for i in range(1, num_clips + 1)]

        instructions = STAGE2_CROSS_MODALITY_INSTRUCTIONS_TEMPLATE.format(
            num_clips=num_clips,
            clip_list="\n".join(clip_rows),
        )

        content_list = []
        segments = re.split("(<image>|<video>|<audio>)", instructions)
        segments = [item for item in segments if item != ""]
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image"})
            elif segment == "<video>":
                content_list.append({"type": "video"})
            elif segment == "<audio>":
                content_list.append({"type": "audio"})
            else:
                content_list.append({"type": "text", "text": segment})

        return [{"role": "user", "content": content_list}]

    def _prepare_cross_modality_stage2_gen_batch(
        self,
        base_batch: DataProto,
        stage1_gen_batch: DataProto,
        stage1_texts: list[str],
    ) -> DataProto:
        """Prepare stage-2 jigsaw gen_batch from stage-1 modality selection outputs."""
        from verl.utils.dataset.rl_dataset import collate_fn

        stage2_max_audio_seconds = self.config.data.get("stage2_max_audio_seconds", 12.0)
        fps_default = float(self.config.data.get("stage2_sampling_fps", 1.0))

        stage2_samples: list[dict] = []
        new_reward_model = []
        new_extra_info = []

        mm_data_list = stage1_gen_batch.non_tensor_batch.get("multi_modal_data", [])
        mm_kwargs_list = stage1_gen_batch.non_tensor_batch.get("mm_processor_kwargs", [])

        for i, stage1_text in enumerate(stage1_texts):
            extra_info_i = {}
            if "extra_info" in base_batch.non_tensor_batch and base_batch.non_tensor_batch["extra_info"][i] is not None:
                extra_info_i = dict(base_batch.non_tensor_batch["extra_info"][i])

            mm_data = mm_data_list[i] if i < len(mm_data_list) else None
            if not isinstance(mm_data, dict):
                raise ValueError("Missing multi_modal_data for cross-modality stage2 preparation")

            raw_videos = mm_data.get("video", []) or []
            raw_audios = mm_data.get("audio", []) or []

            clip_videos = [
                torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in raw_videos
            ]
            clip_audios = [np.asarray(a, dtype=np.float32) for a in raw_audios]

            clip_num = int(extra_info_i.get("stage1_clip_num", 0))
            if clip_num <= 0:
                clip_num = len(clip_videos) if clip_videos else len(clip_audios)
            if clip_num <= 0:
                raise ValueError("No clips available for cross-modality stage2 preparation")

            # Align lengths defensively.
            if clip_videos and clip_audios:
                clip_num = min(clip_num, len(clip_videos), len(clip_audios))
            elif clip_videos:
                clip_num = min(clip_num, len(clip_videos))
            elif clip_audios:
                clip_num = min(clip_num, len(clip_audios))
            if clip_videos:
                clip_videos = clip_videos[:clip_num]
            if clip_audios:
                clip_audios = clip_audios[:clip_num]

            parse_res = self._parse_stage1_modality_selection(stage1_text, num_clips=clip_num)
            parse_ok = bool(parse_res.get("parse_ok", False))
            modalities = parse_res.get("modalities", [])
            if not parse_ok or len(modalities) != clip_num:
                modalities = ["VA"] * clip_num
                parse_ok = False

            masked_videos: list[torch.Tensor] = []
            masked_audios: list[np.ndarray] = []
            for vid, aud, mod in zip(clip_videos, clip_audios, modalities, strict=True):
                mod_norm = str(mod).upper()
                if mod_norm == "A":
                    vid = torch.zeros_like(vid)
                elif mod_norm == "V":
                    aud = np.zeros_like(aud)
                elif mod_norm != "VA":
                    # Defensive fallback: keep both modalities if an unexpected tag appears.
                    mod_norm = "VA"
                masked_videos.append(vid)
                masked_audios.append(aud)

            perm = list(range(clip_num))
            random.shuffle(perm)
            gt = [perm.index(j) + 1 for j in range(clip_num)]
            jigsaw_videos = [masked_videos[j] for j in perm]
            jigsaw_audios = [masked_audios[j] for j in perm]
            stage2_modalities = [modalities[j] for j in perm]

            messages = self._build_stage2_cross_modality_messages(num_clips=clip_num)
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            fps_for_processor = fps_default
            if i < len(mm_kwargs_list):
                mm_kwargs_stage1 = mm_kwargs_list[i]
                if isinstance(mm_kwargs_stage1, dict) and "fps" in mm_kwargs_stage1:
                    fps_for_processor = float(mm_kwargs_stage1["fps"])

            audio_max_len = int(float(stage2_max_audio_seconds) * 16000)
            mm_kwargs = {
                "use_audio_in_video": True,
                "fps": fps_for_processor,
                "text_kwargs": {"padding": True},
                "audio_kwargs": {"padding": "max_length", "max_length": audio_max_len},
            }

            processor_kwargs = {
                "text": [text_prompt],
                "videos": jigsaw_videos,
                "audio": jigsaw_audios,
                "return_tensors": "pt",
                "use_audio_in_video": True,
                "fps": fps_for_processor,
                "text_kwargs": {"padding": True},
                "audio_kwargs": {"padding": "max_length", "max_length": audio_max_len},
            }
            model_inputs = self.processor(**processor_kwargs)

            multi_modal_inputs = dict(model_inputs)
            input_ids = multi_modal_inputs.pop("input_ids")
            attention_mask = multi_modal_inputs.pop("attention_mask")

            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.truncation,
            )

            audio_seqlens = multi_modal_inputs["feature_attention_mask"].sum(dim=-1)
            from model.qwen3_omni_moe import get_rope_index

            video_second_per_grid = multi_modal_inputs.get("video_second_per_grid")
            if video_second_per_grid is None:
                raise RuntimeError("Processor did not return `video_second_per_grid` for stage2 video input")

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=multi_modal_inputs.get("image_grid_thw"),
                video_grid_thw=multi_modal_inputs.get("video_grid_thw"),
                second_per_grids=video_second_per_grid.to(dtype=torch.bfloat16),
                attention_mask=attention_mask,
                use_audio_in_video=True,
                audio_seqlens=audio_seqlens,
            )

            multi_modal_data = {
                "video": [v.numpy() if isinstance(v, torch.Tensor) else v for v in jigsaw_videos],
                "audio": jigsaw_audios,
            }

            stage2_samples.append(
                {
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "text_prompt": text_prompt,
                    "multi_modal_data": multi_modal_data,
                    "mm_processor_kwargs": mm_kwargs,
                    "multi_modal_inputs": multi_modal_inputs,
                }
            )

            extra_info_i.update(
                {
                    "stage1_output": stage1_text,
                    "stage1_parse_ok": parse_ok,
                    "stage1_error": parse_res.get("error"),
                    "stage1_modalities": modalities,
                    "stage2_jigsaw_num_clips": clip_num,
                    "stage2_jigsaw_shuffle_perm": perm,
                    "stage2_jigsaw_modalities": stage2_modalities,
                }
            )
            new_extra_info.append(extra_info_i)

            reward_model_i = {}
            if "reward_model" in base_batch.non_tensor_batch and base_batch.non_tensor_batch["reward_model"][i] is not None:
                reward_model_i = dict(base_batch.non_tensor_batch["reward_model"][i])
            reward_model_i["ground_truth"] = gt
            new_reward_model.append(reward_model_i)

        base_batch.non_tensor_batch["extra_info"] = np.array(new_extra_info, dtype=object)
        base_batch.non_tensor_batch["reward_model"] = np.array(new_reward_model, dtype=object)

        stage2_batch_dict = collate_fn(stage2_samples)
        return DataProto.from_single_dict(stage2_batch_dict)

    def fit(self):
        """Run PPO training while coordinating worker groups through RPC."""

        
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Restore state before validation or optimization.
        self._load_checkpoint()

        # Run reward-function validation before training when configured.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Training steps are one-indexed.
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # Assign stable IDs for group-wise advantage computation.
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Two-stage CMM pipeline (stage 1 modality selection -> stage 2 jigsaw).
                if self._is_two_stage_cross_modality_jigsaw_batch(batch):
                    if self.async_rollout_mode:
                        raise NotImplementedError("Two-stage jigsaw is only implemented for sync rollout.")

                    # Generate one modality decision per sample before rollout repetition.
                    stage1_gen_batch = self._get_gen_batch(batch)
                    stage1_gen_batch.meta_info["global_steps"] = self.global_steps
                    stage1_gen_batch.meta_info["do_sample"] = self.config.data.get("stage1_do_sample", False)
                    stage1_gen_batch.meta_info["sampling_params"] = {
                        "temperature": float(self.config.data.get("stage1_temperature", 0.0)),
                        "top_p": float(self.config.data.get("stage1_top_p", 1.0)),
                        "top_k": int(self.config.data.get("stage1_top_k", -1)),
                        "repetition_penalty": float(self.config.data.get("stage1_repetition_penalty", 1.05)),
                        "max_tokens": int(
                            self.config.data.get("stage1_max_tokens", self.config.data.max_response_length)
                        ),
                    }
                    
                    stage1_gen_out = self.actor_rollout_wg.stage1_generate_sequences(stage1_gen_batch)

                    # Keep timing for profiling.
                    timing_raw.update(stage1_gen_out.meta_info.get("timing", {}))
                    stage1_gen_out.meta_info.pop("timing", None)

                    stage1_texts = self._decode_responses_to_text(stage1_gen_out)

                    # Stage-2: build jigsaw inputs + prompt based on stage-1 outputs.
                    gen_batch = self._prepare_cross_modality_stage2_gen_batch(
                        base_batch=batch,
                        stage1_gen_batch=stage1_gen_batch,
                        stage1_texts=stage1_texts,
                    )

                    # Free the temporary stage-1 objects early (can be large).
                    del stage1_gen_batch, stage1_gen_out, stage1_texts
                else:
                    # Execute the single-stage pipeline.
                    gen_batch = self._get_gen_batch(batch)

                # Propagate the global step to rollout tracing.
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # Generate rollout responses.
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    # Repeat inputs to align with the rollout response count.
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balancing preserves UID-based advantage groups but may change
                    # mini-batch composition during loss computation.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Record global valid-token counts for FLOP estimation.
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # Compute reward-model scores when they are not precomputed.
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Recompute actor log probabilities for PPO updates.
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # Add rollout/training probability mismatch diagnostics.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # Compute reference-policy log probabilities.
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                            
                            # Drop the standalone reference output after merging it.
                            del ref_log_prob

                    # Compute critic values when required.
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # Combine learned and rule-based reward scores.
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if "extra_info" in batch.non_tensor_batch:
                            extra_infos = batch.non_tensor_batch["extra_info"]
                            if reward_extra_infos_dict is None:
                                reward_extra_infos_dict = {}
                            if len(extra_infos) == len(batch):
                                stage1_outputs = []
                                for info in extra_infos:
                                    if isinstance(info, dict):
                                        stage1_outputs.append(info.get("stage1_output"))
                                    else:
                                        stage1_outputs.append(None)
                                reward_extra_infos_dict.setdefault("stage1_output", stage1_outputs)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Apply the configured KL reward penalty.
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance weights and policy-mismatch metrics once per batch.
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # Importance-sampling metrics already include their namespace prefix.
                        metrics.update(is_metrics)

                        # Compute advantages on the driver process.
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # Configure within-group advantage normalization for GRPO.

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # Update the critic when enabled.
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # Delay actor updates until critic warmup is complete.
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Save rollout generations when configured.
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # Run scheduled validation.
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Detect imminent elastic-instance expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Save on schedule, at the final step, or before resource expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # Record training-state metrics.
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # Aggregate batch and timing metrics.
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            
                
                reward_extra_keys = [
                    "acc_reward",
                    "format_reward",
                    "global_reward",
                    "continuity_reward",
                    "acc",
                    "rep_penalty"
                ]
                for key in reward_extra_keys:
                    if key in batch.non_tensor_batch:
                        values = batch.non_tensor_batch[key]
                        metrics[f"critic/rewards/{key}_mean"] = float(np.mean(values))
                        metrics[f"critic/rewards/{key}_max"] = float(np.max(values))
                        metrics[f"critic/rewards/{key}_min"] = float(np.min(values))
       
                # Record realized throughput metrics.
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Update curriculum state after consuming the current batch.
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                
                # Periodically reclaim fragmented host and device memory.
                if self.global_steps % 5 == 0:
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # Notify datasets that adapt between training batches.
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
