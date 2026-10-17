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

from omnijigsaw_utils import process_video_jigsaw_with_audio

logger = logging.getLogger(__name__)


STAGE1_INSTRUCTIONS_TEMPLATE = """
### Role
You are a Multi-modal Jigsaw Puzzle Expert. Your goal is to curate the modalities for a high-quality "Omni Jigsaw" puzzle based on 6 sequential video clips.

### What is Omni Jigsaw?
Omni Jigsaw challenges a solver to reconstruct the chronological timeline of shuffled clips using a curated mix of Visual (V), Audio (A), and Cross-modal (VA) clues.

### Input Data
You are given 6 **ordered** video clips (each contains both video and audio). These clips were extracted from one original video in chronological order and have equal length.

Clip list:
{clip_list}

### Task
Assign the most appropriate modality (**"V"**, **"A"**, or **"VA"**) to **each** clip based on its information richness and its role in the temporal sequence.

**Analysis Steps:**
1.  **Content Evaluation**: For each clip, assess the richness and distinctiveness of its Visual stream vs. its Audio stream.
2.  **Modality Decision Logic**:
    - **Select "V"**: If the **Visual** evidence is distinct (e.g., scene change, physical motion) while the audio is generic, ambient, or redundant.
    - **Select "A"**: If the **Audio** evidence is distinct (e.g., continuous sentence, distinct sound cue) while the visual is static, blurry, or less informative.
    - **Select "VA"**: If **BOTH** video and audio provide rich, critical clues for placing this clip in the timeline (e.g., a specific character action synchronized with a plot-critical dialogue). Use VA when the clip serves as a strong "anchor" that requires full context.
3.  **Sequence Verification**: 
    - Review your selections. Do they allow the solver to deduce the order through natural content progression? Ensure the sequence isn't rendered unsolvable by over-hiding critical cues.

### Constraints
- **Avoid** outputting all "V", all "A", or all "VA".
- Strive for a natural mix. While VA is valuable, distinct V and A clips are also essential for testing specific modality understanding.

### Output Format
You MUST provide the final result in valid JSON format inside <answer>...</answer> tags following this structure:
{{
  "modalities": ["V", "A", "VA", "V", "A", "V"]
}}

You **FIRST** think about the **Content Richness and Modality Strategy** as an internal monologue (Content Evaluation -> Modality Selection -> Sequence Verification) and **THEN** provide the final design. The thinking process MUST BE enclosed within <thinking> </thinking> tags. The final design MUST BE enclosed within <answer> </answer> tags.
"""


class CMMDataset(RLHFDataset):
    """Two-stage CMM dataset that prepares stage-1 modality-selection inputs."""

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        self.multimodal_folder = config.get("multimodal_folder", None)
        self.max_pixels = config.get("max_pixels", None)
        self.max_frames = config.get("max_frames", None)
        self.stage1_sampling_fps = float(config.get("stage1_sampling_fps", config.get("stage2_sampling_fps", 1.0)))
        self.stage1_max_audio_seconds = float(
            config.get("stage1_max_audio_seconds", config.get("stage2_max_audio_seconds", 12.0))
        )

        self.dataset_repeat_times = config.get("dataset_repeat_times", None)
        if self.dataset_repeat_times is not None and isinstance(self.dataset_repeat_times, str):
            self.dataset_repeat_times = eval(self.dataset_repeat_times)

        if processor is None:
            raise ValueError("A Hugging Face processor is required for CMMDataset.")

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

    def _build_stage1_messages(self, num_clips: int) -> list[dict]:
        clip_list_rows = [f"Clip {i}: <video>" for i in range(1, num_clips + 1)]
        instructions = STAGE1_INSTRUCTIONS_TEMPLATE.format(
            num_clips=num_clips,
            clip_list="\n".join(clip_list_rows),
        )

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

        return [{"role": "user", "content": content_list}]

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        if self.video_key not in row_dict or not row_dict.get(self.video_key, None):
            raise ValueError(f"Missing `{self.video_key}` for cross-modality jigsaw at index {item}")

        video_info = row_dict[self.video_key][0]
        if not isinstance(video_info, dict) or "path" not in video_info:
            raise TypeError(f"Unexpected video_info format at index {item}: {type(video_info)} {video_info}")

        if not video_info["path"].startswith("/"):
            if not self.multimodal_folder:
                raise ValueError("multimodal_folder is required when video paths are relative")
            video_info["path"] = os.path.join(self.multimodal_folder, video_info["path"])

        if not isinstance(video_info.get("fps", None), (int, float)) or float(video_info.get("fps", 0)) <= 0:
            video_info["fps"] = self.stage1_sampling_fps

        try:
            videos, audios, fps_list = process_video_jigsaw_with_audio(
                video_info,
                max_pixels=self.max_pixels,
                fps_max_frames=self.max_frames,
                max_audio_seconds=self.stage1_max_audio_seconds,
            )
        except Exception as e:
            logger.error(f"Failed to process video/audio for cross-modality jigsaw at index {item}: {e}")
            random_idx = random.randint(0, self.__len__() - 1)
            return self.__getitem__(random_idx)

        clip_num = int(video_info.get("jigsaw_clip_num", len(videos)))

        fps = 1.0
        try:
            if not fps_list:
                raise ValueError("FPS list is empty")
            ref_fps = fps_list[0]
            assert all(abs(x - ref_fps) < 1e-5 for x in fps_list), f"FPS values vary: {fps_list}"
            fps = ref_fps
        except (AssertionError, ValueError):
            if fps_list:
                fps = sum(fps_list) / len(fps_list)
                print(f"[Warning] Detected inconsistent FPS in clips: {fps_list}. Using average FPS: {fps:.4f}")
            else:
                fps = 1.0
                print("[Warning] FPS list is empty. Defaulting to 1.0")

        messages = self._build_stage1_messages(num_clips=clip_num)
        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        row_dict["text_prompt"] = text_prompt

        audio_max_len = int(self.stage1_max_audio_seconds * 16000)
        model_inputs = self.processor(
            text=[text_prompt],
            videos=videos,
            audio=audios,
            return_tensors="pt",
            use_audio_in_video=True,
            fps=fps,
            text_kwargs={"padding": True},
            audio_kwargs={"padding": "max_length", "max_length": audio_max_len},
        )

        row_dict["multi_modal_data"] = {
            "video": [v.numpy() if isinstance(v, torch.Tensor) else v for v in videos],
            "audio": [a for a in audios],
        }
        row_dict["mm_processor_kwargs"] = {
            "use_audio_in_video": True,
            "fps": fps,
            "text_kwargs": {"padding": True},
            "audio_kwargs": {"padding": "max_length", "max_length": audio_max_len},
        }

        extra_info = row_dict.get("extra_info", {}) or {}
        extra_info = dict(extra_info)
        extra_info.update(
            {
                "omni_jigsaw_two_stage": True,
                "cross_modality_jigsaw": True,
                "stage1_clip_num": clip_num,
                "stage1_clip_fps": float(fps),
            }
        )
        row_dict["extra_info"] = extra_info

        row_dict.setdefault("reward_model", {})
        row_dict["reward_model"]["ground_truth"] = []

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

        audio_feature_mask = model_inputs.get("feature_attention_mask")
        audio_seqlens = audio_feature_mask.sum(dim=-1) if audio_feature_mask is not None else None
        from model.qwen3_omni_moe import get_rope_index

        video_second_per_grid = model_inputs.get("video_second_per_grid")
        if video_second_per_grid is None:
            raise RuntimeError("Processor did not return `video_second_per_grid` for stage1 video input")

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
