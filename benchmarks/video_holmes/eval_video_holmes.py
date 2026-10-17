import os
import json
import argparse
from tqdm import tqdm
import torch
import gc

from infer.inference_engine import TTSInferenceEngine, InferenceEngine
from infer.video_utils import load_video_and_audio_data, load_video, load_video_and_audio_data_batch, load_video_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Video-Holmes Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Root dir containing 'test_Video-Holmes.json', 'videos_cropped'")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save result JSON")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_num_seqs", type=int, default=8, help="Max number of sequences per iteration (batch size)")
    parser.add_argument("--max_model_len", type=int, default=65536, help="Maximum context length for the model")

    parser.add_argument("--num_clips", type=int, default=6, help="Number of clips for Jigsaw stage")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--disable_audio", action="store_true", help="Whether to disable audio during video inference. If set, processes silent video; otherwise, processes video with audio.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for data loading")

    return parser.parse_args()


def main():
    args = parse_args()

    json_file = os.path.join(args.benchmark_dir, "test_Video-Holmes.json")
    video_root = os.path.join(args.benchmark_dir, "videos_cropped")

    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Annotation file not found: {json_file}")
    if not os.path.exists(video_root):
        raise FileNotFoundError(f"Video directory not found at: {video_root}")

    print(f">>> Loading Video-Holmes Data from {json_file}")
    with open(json_file, 'r') as f:
        data_list = json.load(f)

    print(">>> [Mode] Standard Inference (Baseline) Enabled")
    limit_mm_per_prompt = {'image': 0, 'video': 1, 'audio': 0} if args.disable_audio else {'image': 0, 'video': 1, 'audio': 1}
    engine = InferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt=limit_mm_per_prompt,
        seed=args.seed
    )

    results = []
    print(f">>> Starting Evaluation on {len(data_list)} questions...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(data_list), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(data_list))
        batch_items = data_list[batch_start:batch_end]

        batch_questions = []
        batch_candidates = []
        batch_loaded_data = []
        batch_gt_letters = []
        batch_meta = []

        video_paths = []
        valid_items = []
        for item in batch_items:
            vid = item.get("video ID")
            qid = item.get("Question ID")

            if vid.endswith(".mp4") or vid.endswith(".mkv"):
                video_filename = vid
            else:
                video_filename = f"{vid}.mp4"

            video_path = os.path.join(video_root, video_filename)

            if not os.path.exists(video_path):
                if os.path.exists(os.path.join(video_root, f"{vid}.mkv")):
                    video_path = os.path.join(video_root, f"{vid}.mkv")
                else:
                    print(f"[Warning] Video missing: {video_path}, skipping QID {qid}")
                    continue

            video_paths.append(video_path)
            valid_items.append(item)

        if len(video_paths) == 0:
            continue

        if not args.disable_audio:
            load_results, load_errors = load_video_and_audio_data_batch(
                video_paths, num_workers=args.num_workers
            )
        else:
            load_results, load_errors = load_video_batch(
                video_paths, num_workers=args.num_workers
            )

        for i, item in enumerate(valid_items):
            if load_errors[i] is not None:
                print(f"[Error] Failed to load video {video_paths[i]}: {load_errors[i]}")
                continue

            loaded_data = load_results[i]

            question_text = item.get("Question")
            raw_options_dict = item.get("Options", {})

            formatted_candidates = []
            sorted_keys = sorted(raw_options_dict.keys())
            for key in sorted_keys:
                content = raw_options_dict[key]
                formatted_candidates.append(f"{key}. {content}")

            gt_letter = item.get("Answer")

            batch_questions.append(question_text)
            batch_candidates.append(formatted_candidates)
            batch_loaded_data.append(loaded_data)
            batch_gt_letters.append(gt_letter)
            batch_meta.append(item.copy())

        if len(batch_questions) == 0:
            continue

        try:
            if not args.disable_audio:
                batch_responses = engine.run_inference_batch(
                    questions=batch_questions,
                    candidates_list=batch_candidates,
                    loaded_data_list=batch_loaded_data,
                    answer_idx_list=batch_gt_letters
                )
            else:
                batch_responses = engine.run_video_inference_batch(
                    questions=batch_questions,
                    candidates_list=batch_candidates,
                    loaded_data_list=batch_loaded_data,
                    answer_idx_list=batch_gt_letters
                )
        except Exception as e:
            print(f"[Error] Batch inference failed: {e}")
            batch_responses = ["Error during generation"] * len(batch_questions)

        for i, response_text in enumerate(batch_responses):
            res_item = batch_meta[i].copy()
            res_item["response"] = response_text
            results.append(res_item)

        del batch_loaded_data
        gc.collect()
        torch.cuda.empty_cache()

        if len(results) % 20 == 0:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    print(f">>> Evaluation finished. Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    engine.close()


if __name__ == "__main__":
    main()
