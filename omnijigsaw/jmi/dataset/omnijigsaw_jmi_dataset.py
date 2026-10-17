import logging
import os
import random
from typing import Optional
import datasets
import torch

from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset 
from omnijigsaw_utils import process_video_jigsaw_with_audio
import verl.utils.torch_functional as verl_F

logger = logging.getLogger(__name__)


class JMIDataset(RLHFDataset):
    """Prepare synchronized audio-video jigsaws for Qwen3-Omni training.

    Clips share one permutation across both modalities. The processor produces
    training tensors, while raw media are retained for the rollout backend.
    """
    
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
        self.dataset_repeat_times = config.get("dataset_repeat_times", None)
        if self.dataset_repeat_times is not None:
            if type(self.dataset_repeat_times) == str:
                self.dataset_repeat_times = eval(self.dataset_repeat_times)
        if processor is None:
            raise ValueError("A Hugging Face processor is required for JMIDataset.")
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

            # Fill missing columns with None before concatenation.
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

        # Concatenate datasets with aligned columns.
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def __getitem__(self, item):
        """
        Processes a single data item to create a jigsaw puzzle and prepares it 
        as input for the Qwen3-Omni model.
        """
        # Read a copy because sample preparation mutates metadata fields.
        row_dict: dict = self.dataframe[item]
        
        videos_list = None
        audios_list = None

        # Resolve multimodal inputs for jigsaw samples.
        if self.video_key in row_dict and row_dict.get(self.video_key, None):
            if 'data_source' in row_dict and 'jigsaw' in row_dict.get('data_source', ''):
                video_info = row_dict[self.video_key][0]
                if not video_info['path'].startswith('/'):
                    video_info['path'] = os.path.join(self.multimodal_folder, video_info['path'])
                
                try:
                    videos, audios, fps_list = process_video_jigsaw_with_audio(
                        video_info, 
                        max_pixels=self.max_pixels, 
                        fps_max_frames=self.max_frames,
                        max_audio_seconds=36.0
                    )
                except Exception as e:
                    logger.error(f"Failed to process video/audio for jigsaw at index {item}: {e}")
                    # Replace unreadable samples without aborting a training batch.
                    random_idx = random.randint(0, self.__len__() - 1)
                    return self.__getitem__(random_idx)

                num_clips = video_info['jigsaw_clip_num']
                clip_indices = list(range(num_clips)) 
                
                random.shuffle(clip_indices)
                
                gt = [clip_indices.index(i) + 1 for i in range(num_clips)]
                
                if 'reward_model' not in row_dict: 
                    row_dict['reward_model'] = {}
                row_dict['reward_model']['ground_truth'] = gt
                
                # Apply one permutation to both modalities.
                videos_list = [videos[i] for i in clip_indices]
                audios_list = [audios[i] for i in clip_indices]
                fps = 1.0
                try:
                    if not fps_list:
                        raise ValueError("FPS list is empty")
                        
                    ref_fps = fps_list[0]
                    assert all(abs(x - ref_fps) < 1e-5 for x in fps_list), f"FPS values vary: {fps_list}"
                    fps = ref_fps
                except (AssertionError, ValueError) as e:
                    if fps_list:
                        fps = sum(fps_list) / len(fps_list)
                        print(f"[Warning] Detected inconsistent FPS in clips: {fps_list}. Using average FPS: {fps:.4f}")
                    else:
                        fps = 1.0
                        print("[Warning] FPS list is empty. Defaulting to 1.0")
            else:
                pass

        messages = self._build_messages(row_dict)
        
        # Build model inputs with the native multimodal processor.
        try:
            # Render the chat template before multimodal tokenization.
            text_prompt = self.processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )

            row_dict["text_prompt"] = text_prompt
            
            # Tokenize the prompt and process its media inputs.
            model_inputs = self.processor(
                text=[text_prompt],
                videos=videos_list, # Each video tensor has shape (frames, channels, height, width).
                audio=audios_list,
                return_tensors="pt",
                use_audio_in_video=True,
                fps=fps,
                text_kwargs={
                    "padding": True
                },
                audio_kwargs={
                    "padding": "max_length",
                    "max_length": 576000
                }
            )
            
        except Exception as e:
            logger.error(f"Qwen3-Omni Processor failed at index {item}: {e}", exc_info=True)
            random_idx = random.randint(0, self.__len__() - 1)
            return self.__getitem__(random_idx)
        
        # Retain raw modalities for the rollout backend.
        multi_modal_data = {}
        
        if videos_list is not None and len(videos_list) > 0:
            multi_modal_data["video"] = [
                v.numpy() if isinstance(v, torch.Tensor) else v 
                for v in videos_list
            ]

        if audios_list is not None and len(audios_list) > 0:
            multi_modal_data["audio"] = audios_list

        if len(multi_modal_data) > 0:
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["mm_processor_kwargs"] = {
                "use_audio_in_video": True,
                "fps": fps,
                "text_kwargs": {
                    "padding": True
                },
                "audio_kwargs": {
                    "padding": "max_length",
                    "max_length": 576000
                }
            }
            
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)
                row_dict["multi_modal_inputs"].pop("input_ids", None)
                row_dict["multi_modal_inputs"].pop("attention_mask", None)

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

        audio_seqlens = model_inputs["feature_attention_mask"].sum(dim=-1)
        # Qwen3-Omni uses multimodal rotary position IDs.
        from model.qwen3_omni_moe import get_rope_index
        position_ids = get_rope_index(
                        self.processor,
                        input_ids=input_ids,
                        image_grid_thw=model_inputs.get("image_grid_thw"),
                        video_grid_thw=model_inputs.get("video_grid_thw"),
                        second_per_grids=model_inputs.get("video_second_per_grid").to(dtype=torch.bfloat16),
                        attention_mask=attention_mask,
                        use_audio_in_video=True,
                        audio_seqlens=audio_seqlens
                    )

        row_dict["input_ids"] = input_ids[0]            # [padded_len]
        row_dict["attention_mask"] = attention_mask[0]  # [padded_len]
        row_dict["position_ids"] = position_ids[0]      # [3, padded_len]
        
        raw_prompt_ids = self.tokenizer.encode(text_prompt, add_special_tokens=False)
        
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = text_prompt

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
