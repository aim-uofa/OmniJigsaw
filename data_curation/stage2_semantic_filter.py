#!/usr/bin/env python3
"""Stage-2 semantic filtering for OmniJigsaw data curation."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


LOGGER = logging.getLogger("omnijigsaw.data_curation.stage2")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi"}

VIDEO_JUDGE_PROMPT = """You are an expert video analyst tasked with determining if a video is suitable for a "Video Jigsaw" (temporal re-ordering) task.

**Task Definition**:
A video is suitable (YES) if it contains a clear, **directionally unambiguous** chronological progression with **distinct visual states**. If this video were cut into 3-6 segments and shuffled, a human should be able to restore the original order purely by identifying specific visual cues and logical flow.

**Evaluation Criteria**:
1. **Causal Progression**: Does action A lead to result B? (e.g., ingredients being mixed then put into an oven).
2. **Visual State Changes**: Is there a clear evolution of an object or environment? (e.g., a house being built, a drawing being completed, a plant growing).
3. **Logical Narrative**: Is there a sequential flow of events? (e.g., a person walking into a building, then sitting at a desk).
4. **Temporal Markers**: Are there environmental changes like lighting (day to night) or physical wear/consumption?

**Exclusion Criteria (Answer NO if)**:
1. **Repetitive Loops**: The person is doing the same action over and over (e.g., a gym workout loop, a person just nodding/talking without changing position).
2. **Static/Low-Dynamic Content**: Talking heads with static backgrounds, or videos with minimal movement/change.
3. **Parallel/Unrelated Clips**: A compilation of unrelated highlights where order doesn't matter.
4. **Visual Ambiguity**: Different parts of the video look too similar (e.g., a person walking in a desert with no landmarks), making re-ordering a guess.

**Output Format**:
You MUST follow this format:
<think>
1. Briefly describe the key actions and state changes observed.
2. Evaluate against the Evaluation and Exclusion criteria.
3. Conclude whether the video is suitable for the "Video Jigsaw" task.
</think>
<answer>YES or NO</answer>"""


def extract_xml_tag(text: str, tag: str) -> str | None:
    """Extract the first non-greedy occurrence of an XML-style tag."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def parse_response(raw_response: str) -> dict[str, Any]:
    """Parse the required reasoning and binary-decision fields."""
    tagged_thought = extract_xml_tag(raw_response, "think")
    tagged_answer = extract_xml_tag(raw_response, "answer")
    thought = tagged_thought
    answer = tagged_answer

    # Accept a bare YES or NO when answer tags are absent.
    if thought is None and answer is not None:
        thought = raw_response.split("<answer>", maxsplit=1)[0].strip()
    if answer is None:
        answer = raw_response.strip()
    if thought is None:
        thought = raw_response.strip()

    answer = answer.strip()
    format_valid = bool(tagged_thought) and tagged_answer in {"YES", "NO"}
    return {
        "thought": thought,
        "answer": answer,
        "format_valid": format_valid,
        "accepted": answer == "YES",
        "raw_response": raw_response,
    }


class Qwen25VLEngine:
    """Run deterministic Qwen2.5-VL judgments through vLLM."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        seed: int,
        nframes: int,
        max_pixels: int,
        max_tokens: int,
        repetition_penalty: float,
    ) -> None:
        LOGGER.info("Loading Qwen2.5-VL model from %s", model_path)
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 0, "video": 1},
            max_model_len=max_model_len,
            max_num_seqs=1,
            seed=seed,
            disable_mm_preprocessor_cache=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty,
        )
        self.nframes = nframes
        self.max_pixels = max_pixels

    def run_inference(self, video_path: Path) -> dict[str, Any]:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "nframes": self.nframes,
                        "max_pixels": self.max_pixels,
                    },
                    {"type": "text", "text": VIDEO_JUDGE_PROMPT},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        _, video_inputs = process_vision_info(messages)
        llm_inputs = {
            "prompt": prompt_text,
            "multi_modal_data": {"video": video_inputs} if video_inputs is not None else {},
        }
        outputs = self.llm.generate(
            [llm_inputs],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        return parse_response(outputs[0].outputs[0].text.strip())


def discover_videos(input_dir: Path) -> list[Path]:
    """Recursively discover supported video files."""
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def load_results(results_file: Path) -> dict[str, dict[str, Any]]:
    """Load an existing result map for resumable processing."""
    if not results_file.exists():
        return {}
    with results_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {results_file}.")
    return data


def save_results(results_file: Path, results: dict[str, dict[str, Any]]) -> None:
    """Atomically persist judgments to avoid a partial checkpoint."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = results_file.with_name(f".{results_file.name}.tmp")
    with temporary_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary_file, results_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply OmniJigsaw stage-2 semantic filtering to a video directory."
    )
    parser.add_argument("--model-path", type=str, required=True, help="Qwen2.5-VL model path or identifier.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing stage-1 videos.")
    parser.add_argument("--results-file", type=Path, required=True, help="JSON file for semantic judgments.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nframes", type=int, default=200)
    parser.add_argument("--max-pixels", type=int, default=100_352)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be within (0, 1].")
    for name in (
        "tensor_parallel_size",
        "max_model_len",
        "nframes",
        "max_pixels",
        "max_tokens",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    input_dir = args.input_dir.resolve()
    results_file = args.results_file.resolve()
    video_paths = discover_videos(input_dir)
    results = load_results(results_file)
    pending_paths = []
    for video_path in video_paths:
        result_key = video_path.relative_to(input_dir).as_posix()
        existing = results.get(result_key)
        if not existing or existing.get("answer") == "ERROR":
            pending_paths.append(video_path)

    LOGGER.info(
        "Discovered %d videos under %s; %d require inference.",
        len(video_paths),
        input_dir,
        len(pending_paths),
    )
    if not pending_paths:
        save_results(results_file, results)
        return

    engine = Qwen25VLEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
        nframes=args.nframes,
        max_pixels=args.max_pixels,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    processed_since_save = 0
    for video_path in tqdm(pending_paths, desc="Stage-2 filtering"):
        result_key = video_path.relative_to(input_dir).as_posix()
        try:
            result = engine.run_inference(video_path)
            results[result_key] = {"path": str(video_path), **result}
        except Exception as exc:
            LOGGER.error("Failed to judge %s: %s", video_path, exc)
            results[result_key] = {
                "path": str(video_path),
                "answer": "ERROR",
                "accepted": False,
                "error": str(exc),
            }

        processed_since_save += 1
        if processed_since_save >= 5:
            save_results(results_file, results)
            processed_since_save = 0

    save_results(results_file, results)
    accepted_count = sum(result.get("accepted") is True for result in results.values())
    rejected_count = sum(result.get("answer") == "NO" for result in results.values())
    error_count = sum(result.get("answer") == "ERROR" for result in results.values())
    LOGGER.info(
        "Stage-2 filtering completed: accepted=%d, rejected=%d, errors=%d; results=%s",
        accepted_count,
        rejected_count,
        error_count,
        results_file,
    )


if __name__ == "__main__":
    main()
