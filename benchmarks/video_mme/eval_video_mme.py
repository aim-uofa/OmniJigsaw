import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import gc

from infer.inference_engine import TTSInferenceEngine, InferenceEngine
from infer.video_utils import load_video_and_audio_data, load_video, load_video_and_audio_data_batch, load_video_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Video-MME Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Root dir containing 'data' and 'videomme' folders")
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

    parquet_path = os.path.join(args.benchmark_dir, "videomme", "test-00000-of-00001.parquet")
    video_root = os.path.join(args.benchmark_dir, "data")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at: {parquet_path}")

    print(f">>> Loading Video-MME data from {parquet_path}")
    df = pd.read_parquet(parquet_path)

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

    print(f">>> Preparing data - Total questions: {len(df)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    all_questions = []
    for idx, row in df.iterrows():
        all_questions.append({
            "video_id": row['video_id'],
            "videoID": row['videoID'],
            "duration": row['duration'],
            "domain": row['domain'],
            "sub_category": row['sub_category'],
            "question_id": row['question_id'],
            "task_type": row['task_type'],
            "question": row['question'],
            "options": row['options'].tolist() if hasattr(row['options'], 'tolist') else list(row['options']),
            "answer": row['answer']
        })

    print(f">>> Starting Evaluation on {len(all_questions)} questions with batch_size={args.batch_size}...")

    all_results = []

    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(all_questions), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(all_questions))
        batch_items = all_questions[batch_start:batch_end]

        batch_questions = []
        batch_candidates = []
        batch_loaded_data = []
        batch_gt_answers = []
        batch_meta = []

        video_path_map = {}  # video_filename -> video_path
        video_filename_to_items = {}  # video_filename -> list of items

        for item in batch_items:
            video_filename = f"{item['videoID']}.mp4"
            video_path = os.path.join(video_root, video_filename)

            if not os.path.exists(video_path):
                print(f"[Warning] Video file missing: {video_path}, skipping question {item['question_id']}")
                continue

            if video_filename not in video_path_map:
                video_path_map[video_filename] = video_path
                video_filename_to_items[video_filename] = []

            video_filename_to_items[video_filename].append(item)

        if len(video_path_map) == 0:
            continue

        unique_video_filenames = list(video_path_map.keys())
        unique_video_paths = [video_path_map[vf] for vf in unique_video_filenames]

        if not args.disable_audio:
            load_results, load_errors = load_video_and_audio_data_batch(
                unique_video_paths, num_workers=args.num_workers
            )
        else:
            load_results, load_errors = load_video_batch(
                unique_video_paths, num_workers=args.num_workers
            )

        video_data_cache = {}
        for i, video_filename in enumerate(unique_video_filenames):
            if load_errors[i] is not None:
                print(f"[Error] Failed to load video {unique_video_paths[i]}: {load_errors[i]}")
                continue
            video_data_cache[video_filename] = load_results[i]

        for video_filename, items_for_video in video_filename_to_items.items():
            if video_filename not in video_data_cache:
                continue

            loaded_data = video_data_cache[video_filename]

            for item in items_for_video:
                question_text = item['question']
                candidates = item['options']
                gt_answer = item['answer']

                batch_questions.append(question_text)
                batch_candidates.append(candidates)
                batch_loaded_data.append(loaded_data)
                batch_gt_answers.append(gt_answer)
                batch_meta.append({
                    "video_id": item['video_id'],
                    "duration": item['duration'],
                    "domain": item['domain'],
                    "sub_category": item['sub_category'],
                    "question_id": item['question_id'],
                    "task_type": item['task_type'],
                    "question": question_text,
                    "options": candidates,
                    "answer": gt_answer
                })

        if len(batch_questions) == 0:
            continue

        try:
            if not args.disable_audio:
                batch_responses = engine.run_inference_batch(
                    questions=batch_questions,
                    candidates_list=batch_candidates,
                    loaded_data_list=batch_loaded_data,
                    answer_idx_list=batch_gt_answers
                )
            else:
                batch_responses = engine.run_video_inference_batch(
                    questions=batch_questions,
                    candidates_list=batch_candidates,
                    loaded_data_list=batch_loaded_data,
                    answer_idx_list=batch_gt_answers
                )
        except Exception as e:
            print(f"[Error] Batch inference failed: {e}")
            batch_responses = ["Error during generation"] * len(batch_questions)

        for i, response_text in enumerate(batch_responses):
            result_item = batch_meta[i].copy()
            result_item["response"] = response_text
            all_results.append(result_item)

        del batch_loaded_data
        gc.collect()
        torch.cuda.empty_cache()

        if len(all_results) % 20 == 0:
            results_by_video = {}
            for res in all_results:
                vid = res['video_id']
                if vid not in results_by_video:
                    results_by_video[vid] = {
                        "video_id": vid,
                        "duration": res['duration'],
                        "domain": res['domain'],
                        "sub_category": res['sub_category'],
                        "questions": []
                    }
                results_by_video[vid]["questions"].append({
                    "question_id": res['question_id'],
                    "task_type": res['task_type'],
                    "question": res['question'],
                    "options": res['options'],
                    "answer": res['answer'],
                    "response": res['response']
                })

            results_list = list(results_by_video.values())
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results_list, f, ensure_ascii=False, indent=4)

    print(f">>> Evaluation finished. Saving results to {args.output_file}")
    results_by_video = {}
    for res in all_results:
        vid = res['video_id']
        if vid not in results_by_video:
            results_by_video[vid] = {
                "video_id": vid,
                "duration": res['duration'],
                "domain": res['domain'],
                "sub_category": res['sub_category'],
                "questions": []
            }
        results_by_video[vid]["questions"].append({
            "question_id": res['question_id'],
            "task_type": res['task_type'],
            "question": res['question'],
            "options": res['options'],
            "answer": res['answer'],
            "response": res['response']
        })

    results_list = list(results_by_video.values())
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)

    engine.close()


if __name__ == "__main__":
    main()
