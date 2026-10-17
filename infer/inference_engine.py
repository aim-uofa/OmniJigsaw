import os
os.environ['VLLM_USE_V1'] = '0'

import re
from typing import List, Optional, Tuple, Dict
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
import torch.distributed as dist
import numpy as np

from .video_utils import prepare_jigsaw_inputs, prepare_original_inputs, load_video_and_audio_data, load_audio_with_av, load_video, prepare_video_input
from .audio_utils import prepare_audio_input
from .templates import STAGE_1_PROMPT, STAGE_2_PROMPT, STANDARD_PROMPT, AUDIO_INFER_PROMPT, VIDEO_INFER_PROMPT, GENERATION_PROMPT, VIDEO_GENERATION_PROMPT, AUDIO_GENERATION_PROMPT


def extract_xml_tag(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_candidates(candidates):
    return "\n".join(candidates)


def build_multimodal_content(content_str):
    content_list = []
    segments = re.split("(<image>|<video>)", content_str)
    segments = [item for item in segments if item != ""]
    for segment in segments:
        if segment == "<image>":
            content_list.append({"type": "image"})
        elif segment == "<video>":
            content_list.append({"type": "video"})
        else:
            content_list.append({"type": "text", "text": segment})
    return content_list


class TTSInferenceEngine:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_num_seqs: int = 8,
        max_model_len: int = 65536,
        num_clips: int = 6,
        seed: int = 1234
    ):
        self.num_clips = num_clips
        self.model_path = model_path
        
        print(f">>> Initializing OmniJigsaw inference engine with model: {model_path}")
        
        # Initialize a single reusable vLLM engine.
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={'image': 0, 'video': num_clips, 'audio': num_clips},
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=seed,
            disable_mm_preprocessor_cache=True
        )
        
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        
        self.s1_sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1.0,
            top_k=-1,
            max_tokens=2048,
            repetition_penalty=1.05
        )
        self.s2_sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1.0,
            top_k=-1,
            max_tokens=2048,
            repetition_penalty=1.05
        )

    def run_inference(
        self,
        question: str,
        candidates: List[str],
        video_path: Optional[str] = None,
        loaded_data: Optional[Tuple] = None,
        answer_idx: Optional[str] = None,  # Used only to identify log entries.
    ) -> str:
        """
        Run two-stage inference from a path or preloaded media tuple.

        Passing ``loaded_data`` avoids decoding the same media more than once.
        """
        
        # Reuse preloaded media when provided by a batched evaluator.
        if loaded_data:
            vr, audio_wav, video_fps = loaded_data
        elif video_path:
            print(f">>> Loading video: {video_path}")
            vr, audio_wav, video_fps = load_video_and_audio_data(video_path)
            print(f">>> Loaded video. Total frames: {len(vr)}, FPS: {video_fps:.2f}, Audio shape: {audio_wav.shape}")
        else:
            raise ValueError("Either video_path or loaded_data must be provided.")

        # Stage 1: jigsaw reordering.
        shuffled_videos, fps, shuffled_audios, gt_order_str = prepare_jigsaw_inputs(
            vr=vr,
            audio_wav=audio_wav,
            video_fps=video_fps,
            num_clips=self.num_clips,
            max_audio_seconds_per_clip=36.0 
        )

        clip_placeholders = "\n".join([f"Clip {i+1}: <video>" for i in range(self.num_clips)])
        stage1_text = STAGE_1_PROMPT.format(num_clips=self.num_clips, clip_placeholders=clip_placeholders)
        
        s1_content_list = build_multimodal_content(stage1_text)
    
        s1_messages = [
            {"role": "user", "content": s1_content_list}
        ]

        s1_prompt_text = self.processor.apply_chat_template(
            s1_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        s1_inputs = {
            'prompt': s1_prompt_text,
            'multi_modal_data': {'video': shuffled_videos, 'audio': shuffled_audios},
            "mm_processor_kwargs": {"use_audio_in_video": True, "fps": fps, "padding": True}
        }

        s1_outputs = self.llm.generate([s1_inputs], sampling_params=self.s1_sampling_params, use_tqdm=False)
        s1_response = s1_outputs[0].outputs[0].text
        
        stage1_thinking = extract_xml_tag(s1_response, "thinking")
        stage1_prediction = extract_xml_tag(s1_response, "answer")
        if not stage1_thinking: 
            stage1_thinking = s1_response

        # Stage 2: question answering with the reordering context.
        candidates_str = format_candidates(candidates)
        stage2_text = STAGE_2_PROMPT.format(
            stage1_thinking=stage1_thinking,
            stage1_prediction=stage1_prediction,
            stage1_gt=gt_order_str,
            question=question,
            candidates_str=candidates_str
        )

        # Stage 2 consumes the original media alongside the stage-1 result.
        s2_videos, s2_audios, s2_fps = prepare_original_inputs(
            vr=vr,
            audio_wav=audio_wav,
            video_fps=video_fps,
            max_frames=200,
            max_audio_seconds=600.0
        )

        s2_messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "video", "video": "placeholder"},
                    {"type": "text", "text": stage2_text}
                ]
            }
        ]

        s2_prompt_text = self.processor.apply_chat_template(
            s2_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        s2_inputs = {
            'prompt': s2_prompt_text,
            'multi_modal_data': {'video': s2_videos, 'audio': s2_audios},
            "mm_processor_kwargs": {"use_audio_in_video": True, "fps": s2_fps}
        }

        s2_outputs = self.llm.generate([s2_inputs], sampling_params=self.s2_sampling_params, use_tqdm=False)
        raw_response = s2_outputs[0].outputs[0].text

        extracted_answer = extract_xml_tag(raw_response, "answer")
        if extracted_answer:
            return extracted_answer
        else:
            return raw_response


    def close(self):
        if dist.is_initialized():
            print(">>> Cleaning up distributed process group...")
            dist.destroy_process_group()


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.95,
        max_num_seqs: int = 8,
        max_model_len: int = 65536,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        seed: int = 1234
    ):
        self.model_path = model_path
        
        print(f">>> Initializing Standard Inference Engine with Model: {model_path}")

        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {'image': 1, 'video': 1, 'audio': 1}
        
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=limit_mm_per_prompt,
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            seed=seed,
            disable_mm_preprocessor_cache=True
        )
        
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1.0,
            top_k=-1,
            max_tokens=2048,
            repetition_penalty=1.05
        )
        

    def run_inference(
        self,
        question: str,
        candidates: List[str],
        video_path: Optional[str] = None,
        loaded_data: Optional[Tuple] = None,
        answer_idx: Optional[str] = None
    ) -> str:
        
        if loaded_data:
            vr, audio_wav, video_fps = loaded_data
        elif video_path:
            print(f">>> Loading video: {video_path}")
            vr, audio_wav, video_fps = load_video_and_audio_data(video_path)
            print(f">>> Loaded video. Total frames: {len(vr)}, FPS: {video_fps:.2f}, Audio shape: {audio_wav.shape}")
        else:
            raise ValueError("Either video_path or loaded_data must be provided.")

        videos, audios, fps = prepare_original_inputs(
            vr=vr,
            audio_wav=audio_wav,
            video_fps=video_fps,
            max_frames=200,
            max_audio_seconds=600.0
        )

        if candidates and len(candidates) > 0:
            candidates_str = format_candidates(candidates)
            raw_prompt_text = STANDARD_PROMPT.format(
                question=question,
                candidates_str=candidates_str
            )
        else:
            raw_prompt_text = GENERATION_PROMPT.format(question=question)

        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "video", "video": "placeholder"},
                    {"type": "text", "text": raw_prompt_text}
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = {
            'prompt': prompt_text,
            'multi_modal_data': {'video': videos, 'audio': audios},
            "mm_processor_kwargs": {"use_audio_in_video": True, "fps": fps}
        }

        outputs = self.llm.generate([inputs], sampling_params=self.sampling_params, use_tqdm=False)
        
        raw_response = outputs[0].outputs[0].text
        
        extracted_answer = extract_xml_tag(raw_response, "answer")
        
        if extracted_answer:
            return extracted_answer
        else:
            return raw_response
        
    def run_inference_batch(
        self,
        questions: List[str],
        candidates_list: List[List[str]],
        video_paths: Optional[List[str]] = None,
        loaded_data_list: Optional[List[Tuple]] = None,
        answer_idx_list: Optional[List[str]] = None
    ) -> List[str]:
        
        batch_size = len(questions)

        if video_paths is None and loaded_data_list is None:
            raise ValueError("Either video_paths or loaded_data_list must be provided.")

        all_inputs = []

        for i in range(batch_size):
            question = questions[i]
            candidates = candidates_list[i]
            if loaded_data_list is not None:
                vr, audio_wav, video_fps = loaded_data_list[i]
            else:
                video_path = video_paths[i]
                print(f">>> Loading video [{i+1}/{batch_size}]: {video_path}")
                vr, audio_wav, video_fps = load_video_and_audio_data(video_path)

            videos, audios, fps = prepare_original_inputs(
                vr=vr,
                audio_wav=audio_wav,
                video_fps=video_fps,
                max_frames=200,
                max_audio_seconds=600.0
            )

            if candidates and len(candidates) > 0:
                candidates_str = format_candidates(candidates)
                raw_prompt_text = STANDARD_PROMPT.format(
                    question=question,
                    candidates_str=candidates_str
                )
            else:
                raw_prompt_text = GENERATION_PROMPT.format(question=question)

            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "video", "video": "placeholder"},
                        {"type": "text", "text": raw_prompt_text}
                    ]
                }
            ]

            prompt_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = {
                'prompt': prompt_text,
                'multi_modal_data': {'video': videos, 'audio': audios},
                "mm_processor_kwargs": {"use_audio_in_video": True, "fps": fps}
            }

            all_inputs.append(inputs)

        outputs = self.llm.generate(all_inputs, sampling_params=self.sampling_params, use_tqdm=False)

        results = []
        
        for output in outputs:
            raw_response = output.outputs[0].text
            extracted_answer = extract_xml_tag(raw_response, "answer")
            if extracted_answer:
                results.append(extracted_answer)
            else:
                results.append(raw_response)

        return results
    
    def run_audio_inference(
        self,
        question: str,
        candidates: List[str],
        audio_path: Optional[str] = None,
        loaded_data: Optional[np.ndarray] = None,
        answer_idx: Optional[str] = None 
    ) -> str:
        
        if loaded_data is not None:
            audio_wav = loaded_data
        elif audio_path:
            print(f">>> Loading audio: {audio_path}")
            audio_wav = load_audio_with_av(audio_path)
            print(f">>> Loaded audio. Audio shape: {audio_wav.shape}")
        else:
            raise ValueError("Either audio_path or loaded_data must be provided.")

        audio_input = prepare_audio_input(
            audio_wav=audio_wav,
            max_audio_seconds=3600.0
        )

        if candidates and len(candidates) > 0:
            candidates_str = format_candidates(candidates)
            raw_prompt_text = AUDIO_INFER_PROMPT.format(
                question=question,
                candidates_str=candidates_str
            )
        else:
            raw_prompt_text = AUDIO_GENERATION_PROMPT.format(question=question)

        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "audio", "audio": "placeholder"},
                    {"type": "text", "text": raw_prompt_text}
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = {
            'prompt': prompt_text,
            'multi_modal_data': {'audio': audio_input}
        }

        outputs = self.llm.generate([inputs], sampling_params=self.sampling_params, use_tqdm=False)
        
        raw_response = outputs[0].outputs[0].text
        
        extracted_answer = extract_xml_tag(raw_response, "answer")
        
        if extracted_answer:
            return extracted_answer
        else:
            return raw_response
        
    def run_audio_inference_batch(
        self,
        questions: List[str],
        candidates_list: List[List[str]],
        audio_paths: Optional[List[str]] = None,
        loaded_data_list: Optional[List[np.ndarray]] = None,
        answer_idx_list: Optional[List[str]] = None 
    ) -> List[str]:
        
        batch_size = len(questions)

        if audio_paths is None and loaded_data_list is None:
            raise ValueError("Either audio_paths or loaded_data_list must be provided.")
        
        all_inputs = []

        for i in range(batch_size):
            question = questions[i]
            candidates = candidates_list[i]
            if loaded_data_list is not None:
                audio_wav = loaded_data_list[i]
            else:
                audio_path = audio_paths[i]
                print(f">>> Loading audio [{i+1}/{batch_size}]: {audio_path}")
                audio_wav = load_audio_with_av(audio_path)

            audio_input = prepare_audio_input(
                audio_wav=audio_wav,
                max_audio_seconds=3600.0
            )

            if candidates and len(candidates) > 0:
                candidates_str = format_candidates(candidates)
                raw_prompt_text = AUDIO_INFER_PROMPT.format(
                    question=question,
                    candidates_str=candidates_str
                )
            else:
                raw_prompt_text = AUDIO_GENERATION_PROMPT.format(question=question)

            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "audio", "audio": "placeholder"},
                        {"type": "text", "text": raw_prompt_text}
                    ]
                }
            ]

            prompt_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = {
                'prompt': prompt_text,
                'multi_modal_data': {'audio': audio_input}
            }

            all_inputs.append(inputs)
        
        outputs = self.llm.generate(all_inputs, sampling_params=self.sampling_params, use_tqdm=False)

        results = []

        for output in outputs:
            raw_response = output.outputs[0].text
            extracted_answer = extract_xml_tag(raw_response, "answer")
            if extracted_answer:
                results.append(extracted_answer)
            else:
                results.append(raw_response)
        
        return results
        
    def run_video_inference(
        self,
        question: str,
        candidates: List[str],
        video_path: Optional[str] = None,
        loaded_data: Optional[Tuple] = None,
        answer_idx: Optional[str] = None
    ) -> str:
        
        if loaded_data is not None:
            vr, video_fps = loaded_data
        elif video_path:
            print(f">>> Loading video: {video_path}")
            vr, video_fps = load_video(video_path)
            print(f">>> Loaded video. Total frames: {len(vr)}, FPS: {video_fps:.2f}")
        else:
            raise ValueError("Either video_path or loaded_data must be provided.")

        video_input, fps = prepare_video_input(
            vr=vr,
            video_fps=video_fps,
            max_frames=200
        )

        if candidates and len(candidates) > 0:
            candidates_str = format_candidates(candidates)
            raw_prompt_text = VIDEO_INFER_PROMPT.format(
                question=question,
                candidates_str=candidates_str
            )
        else:
            raw_prompt_text = VIDEO_GENERATION_PROMPT.format(question=question)

        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "video", "video": "placeholder"},
                    {"type": "text", "text": raw_prompt_text}
                ]
            }
        ]

        prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = {
            'prompt': prompt_text,
            'multi_modal_data': {'video': video_input},
            "mm_processor_kwargs": {"fps": fps}
        }

        outputs = self.llm.generate([inputs], sampling_params=self.sampling_params, use_tqdm=False)
        
        raw_response = outputs[0].outputs[0].text
        
        extracted_answer = extract_xml_tag(raw_response, "answer")
        
        if extracted_answer:
            return extracted_answer
        else:
            return raw_response
        
    def run_video_inference_batch(
        self,
        questions: List[str],
        candidates_list: List[List[str]],
        video_paths: Optional[List[str]] = None,
        loaded_data_list: Optional[List[Tuple]] = None,
        answer_idx_list: Optional[List[str]] = None
    ) -> List[str]:
        
        batch_size = len(questions)

        if video_paths is None and loaded_data_list is None:
            raise ValueError("Either video_paths or loaded_data_list must be provided.")
        
        all_inputs = []

        for i in range(batch_size):
            question = questions[i]
            candidates = candidates_list[i]
            if loaded_data_list is not None:
                vr, video_fps = loaded_data_list[i]
            else:
                video_path = video_paths[i]
                print(f">>> Loading video [{i+1}/{batch_size}]: {video_path}")
                vr, video_fps = load_video(video_path)

            video_input, fps = prepare_video_input(
                vr=vr,
                video_fps=video_fps,
                max_frames=200
            )

            if candidates and len(candidates) > 0:
                candidates_str = format_candidates(candidates)
                raw_prompt_text = VIDEO_INFER_PROMPT.format(
                    question=question,
                    candidates_str=candidates_str
                )
            else:
                raw_prompt_text = VIDEO_GENERATION_PROMPT.format(question=question)

            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "video", "video": "placeholder"},
                        {"type": "text", "text": raw_prompt_text}
                    ]
                }
            ]

            prompt_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            inputs = {
                'prompt': prompt_text,
                'multi_modal_data': {'video': video_input},
                "mm_processor_kwargs": {"fps": fps}
            }

            all_inputs.append(inputs)
        
        outputs = self.llm.generate(all_inputs, sampling_params=self.sampling_params, use_tqdm=False)

        results = []

        for output in outputs:
            raw_response = output.outputs[0].text
            extracted_answer = extract_xml_tag(raw_response, "answer")
            if extracted_answer:
                results.append(extracted_answer)
            else:
                results.append(raw_response)
        
        return results

    def close(self):
        if dist.is_initialized():
            print(">>> Cleaning up distributed process group...")
            dist.destroy_process_group()
