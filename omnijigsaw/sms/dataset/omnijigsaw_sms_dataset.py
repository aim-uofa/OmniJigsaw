import logging
import os
import random
import re
from typing import Optional

import datasets
import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset

from omnijigsaw_utils import process_full_video_with_audio

logger = logging.getLogger(__name__)


STAGE1_INSTRUCTIONS_TEMPLATE = """
### Role
You are a Multi-modal Content Analyst. Your goal is to evaluate the **visual stream** and **audio track** of the provided video to determine the optimal modality for a "Temporal Jigsaw Puzzle".

### Task
You must decide whether the **Visual stream** or the **Audio stream** contains stronger, more deterministic clues for reconstructing the chronological order if this video were sliced into shuffled clips.

### Analysis Guidelines

**1. Evaluate Visual Suitability (For Video Jigsaw):**
*   **Look for:** Clear physical actions (e.g., pouring water), scene transitions, camera movement, or object state changes (e.g., assembling a puzzle).
*   **Penalty:** Give low priority to "V" if the video is static (e.g., a person sitting still with minimal movement), repetitive (loops), mostly black/blurry, or shows minimal change from beginning to end.

**2. Evaluate Audio Suitability (For Audio Jigsaw):**
*   **Look for:** Continuous Speech (e.g., narrative/dialogue with logical flow), Musical Progression (e.g., verse -> chorus), or Sequential Sound Events (e.g., footsteps -> door open -> slam).
*   **Penalty:** Give low priority to "A" if the audio is constant background noise, a repetitive music loop without progression, or silence.

**3. Comparison & Decision:**
*   **Select "V"** if the visual evolution provides a stricter, unambiguous timeline (e.g., a silent movie with clear plot actions).
*   **Select "A"** if the auditory narrative provides the primary logical thread (e.g., a podcast, a speech, or a static shot of a narrator).
*   *Tie-Breaker:* If both are good, choose the one that requires less ambiguous guessing for solving the "Temporal Jigsaw Puzzle".

### Input Data
<video>

### Output Format
Output **only** the single character decision inside valid tags.
<answer>V</answer>
or
<answer>A</answer>

You **FIRST** think about the comparison process as an internal monologue **(Analyze Visual Suitability -> Analyze Audio Suitability -> Final Decision)** and **THEN** provide the final answer. The thinking process MUST BE enclosed within <thinking> </thinking> tags. The final answer MUST BE enclosed within <answer> </answer> tags.
"""


class SMSDataset(RLHFDataset):
    """Two-stage SMS dataset that prepares stage-1 modality-selection inputs.

    This dataset prepares the *first-stage* input for Sample-level Modality Selection:
    - a low-res, low-frame-count approximation of the full video
    - the full audio
    - a prompt that asks the model to select V (video) or A (audio)

    The *second-stage* jigsaw inputs are dynamically created in the trainer based on the
    stage-1 model output, so this dataset intentionally does NOT construct the jigsaw clips.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        self.multimodal_folder = config.get("multimodal_folder", None)

        # Stage-1 (modality selection) media budgets.
        self.stage1_max_pixels = config.get("stage1_max_pixels", config.get("max_pixels", 100352))
        self.stage1_max_frames = config.get("stage1_max_frames", config.get("max_frames", 100))
        self.stage1_sample_fps = float(config.get("stage1_sample_fps", 1.0))

        # Apply per-source repetition for multi-file training data.
        self.dataset_repeat_times = config.get("dataset_repeat_times", None)
        if self.dataset_repeat_times is not None and isinstance(self.dataset_repeat_times, str):
            self.dataset_repeat_times = eval(self.dataset_repeat_times)

        if processor is None:
            raise ValueError("A Hugging Face processor is required for SMSDataset.")

        super().__init__(data_files, tokenizer, config, processor)

    def _read_files_and_tokenize(self):
        dataframes = []
        repeat_times = [1] * len(self.data_files)
        if self.dataset_repeat_times is not None:
            repeat_times = self.dataset_repeat_times

        dataset_index = 0
        all_columns = set()

        for parquet_file in self.data_files:
            df = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            all_columns.update(df.column_names)

        for parquet_file, repeat_time in zip(self.data_files, repeat_times):
            df = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            df = df.add_column("dataset_index", [dataset_index] * len(df))

            # Fill in missing columns with None so concatenate_datasets is safe.
            for col in all_columns:
                if col not in df.column_names:
                    df = df.add_column(col, [None] * len(df))

            if repeat_time >= 1:
                for _ in range(int(repeat_time)):
                    dataframes.append(df)
            elif 0 < repeat_time < 1:
                n_samples = int(len(df) * repeat_time)
                sampled = df.shuffle(seed=42).select(range(n_samples))
                dataframes.append(sampled)

            dataset_index += 1

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _build_stage1_modality_messages(self) -> list[dict]:
        instructions = STAGE1_INSTRUCTIONS_TEMPLATE

        content_list = []
        segments = re.split("(<image>|<video>)", instructions)
        segments = [item for item in segments if item != ""]

        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image"})
            elif segment == "<video>":
                content_list.append({"type": "video"})
            else:
                content_list.append({"type": "text", "text": segment})

        return [
            {
                "role": "user",
                "content": content_list,
            }
        ]

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        # Drop the stage-1 prompt before constructing the stage-2 sample.
        row_dict.pop(self.prompt_key, None)

        # Load full video + full audio (low-res, sampled frames).
        if self.video_key not in row_dict or not row_dict.get(self.video_key, None):
            raise ValueError(f"Missing `{self.video_key}` for modality selection at index {item}")

        video_info = row_dict[self.video_key][0]
        if not isinstance(video_info, dict) or "path" not in video_info:
            raise TypeError(f"Unexpected video_info format at index {item}: {type(video_info)} {video_info}")

        # Resolve relative path using multimodal_folder.
        if not video_info["path"].startswith("/"):
            if not self.multimodal_folder:
                raise ValueError("multimodal_folder is required when video paths are relative")
            video_info["path"] = os.path.join(self.multimodal_folder, video_info["path"])

        video_path = video_info["path"]

        try:
            full_video, full_audio, sample_fps, duration_sec, video_fps = process_full_video_with_audio(
                video_path=video_path,
                max_pixels=self.stage1_max_pixels,
                max_frames=self.stage1_max_frames,
                sample_fps=self.stage1_sample_fps,
                add_timestamps=False,
            )
        except Exception as e:
            logger.error(f"Failed to process full video/audio for modality selection at index {item}: {e}", exc_info=True)
            random_idx = random.randint(0, self.__len__() - 1)
            return self.__getitem__(random_idx)

        videos_list = [full_video]
        audios_list = [full_audio]

        # Build stage-1 modality selection prompt.
        messages = self._build_stage1_modality_messages()

        try:
            text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            row_dict["text_prompt"] = text_prompt

            model_inputs = self.processor(
                text=[text_prompt],
                videos=videos_list,
                audio=audios_list,
                return_tensors="pt",
                use_audio_in_video=True,
                fps=float(sample_fps),
                padding=True,
            )
        except Exception as e:
            logger.error(f"Qwen3-Omni Processor failed at index {item}: {e}", exc_info=True)
            random_idx = random.randint(0, self.__len__() - 1)
            return self.__getitem__(random_idx)

        # Minimal non-tensor multimodal payload for vLLM.
        row_dict["multi_modal_data"] = {
            "video": [full_video.numpy() if isinstance(full_video, torch.Tensor) else full_video],
            "audio": audios_list,
        }
        row_dict["mm_processor_kwargs"] = {
            "use_audio_in_video": True,
            "fps": float(sample_fps),
            "padding": True,
        }

        # Mark this sample as "two-stage jigsaw" so trainer can switch pipeline.
        extra_info = row_dict.get("extra_info", {}) or {}
        extra_info = dict(extra_info)
        extra_info.update(
            {
                "omni_jigsaw_two_stage": True,
                "source_video_path": video_path,
                "source_video_duration_sec": float(duration_sec),
                "source_video_fps": float(video_fps),
            }
        )
        row_dict["extra_info"] = extra_info

        # Placeholder ground truth; will be replaced after stage-1 in trainer.
        row_dict.setdefault("reward_model", {})
        row_dict["reward_model"]["ground_truth"] = []

        # Convert to model inputs (tensor) for rollout.
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # Position IDs for Qwen3-Omni mRoPE.
        audio_seqlens = model_inputs["feature_attention_mask"].sum(dim=-1)
        from model.qwen3_omni_moe import get_rope_index

        video_second_per_grid = model_inputs.get("video_second_per_grid")
        if video_second_per_grid is None:
            raise RuntimeError("Processor did not return `video_second_per_grid` for video input")

        position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            second_per_grids=video_second_per_grid.to(dtype=torch.bfloat16),
            attention_mask=attention_mask,
            use_audio_in_video=True,
            audio_seqlens=audio_seqlens,
        )

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        return row_dict
