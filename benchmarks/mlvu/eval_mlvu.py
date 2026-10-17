import os
import json
import argparse
from tqdm import tqdm
import torch
import gc

from infer.inference_engine import TTSInferenceEngine, InferenceEngine
from infer.video_utils import load_video_and_audio_data, load_video, load_video_and_audio_data_batch, load_video_batch


def parse_args():
    parser = argparse.ArgumentParser(description="MLVU Evaluation Script (Tasks 1-7)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Root dir of MLVU containing 'MLVU' folder")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save result JSON")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_num_seqs", type=int, default=8, help="Max number of sequences per iteration (batch size)")
    parser.add_argument("--max_model_len", type=int, default=65536, help="Maximum context length for the model")

    parser.add_argument("--num_clips", type=int, default=6, help="Number of clips for Jigsaw stage")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--disable_audio", action="store_true", help="Whether to disable audio during video inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for data loading")

    return parser.parse_args()


def main():
    args = parse_args()
    
    TASKS = [
        "1_plotQA",
        "2_needle",
        "3_ego",
        "4_count",
        "5_order",
        "6_anomaly_reco",
        "7_topic_reasoning",
    ]

    json_root = os.path.join(args.benchmark_dir, "MLVU/json")
    video_root = os.path.join(args.benchmark_dir, "MLVU/video")

    if not os.path.exists(json_root) or not os.path.exists(video_root):
        raise FileNotFoundError(f"MLVU structure seems incorrect. Expected 'json' and 'video' dirs under {args.benchmark_dir}")

    all_items = []
    print(f">>> Loading MLVU Data (Tasks 1-7) from {args.benchmark_dir}")
    
    for task in TASKS:
        json_file = os.path.join(json_root, f"{task}.json")
        if not os.path.exists(json_file):
            print(f"[Warning] Task file not found: {json_file}, skipping.")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                item['task_folder'] = task 
                all_items.append(item)

    print(f">>> Total samples loaded: {len(all_items)}")

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
    batch_size = args.batch_size
    
    for batch_start in tqdm(range(0, len(all_items), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(all_items))
        batch_items = all_items[batch_start:batch_end]

        batch_questions = []
        batch_candidates = []
        batch_loaded_data = []
        batch_gt_letters = []
        batch_meta = []

        video_paths = []
        valid_items = []
        for item in batch_items:
            video_filename = item['video']
            task_folder = item['task_folder']
            video_path = os.path.join(video_root, task_folder, video_filename)

            if not os.path.exists(video_path):
                print(f"[Warning] Video missing: {video_path}")
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
                uid = f"{item['task_folder']}/{item['video']}"
                print(f"[Error] Failed to load {uid}: {load_errors[i]}")
                continue

            loaded_data = load_results[i]

            question_text = item['question']
            raw_candidates = item.get('candidates', [])
            gt_content = item.get('answer', "")
            question_type = item.get('question_type', item['task_folder'])

            formatted_candidates = []
            gt_letter = None

            if raw_candidates and len(raw_candidates) > 0:
                for idx, cand in enumerate(raw_candidates):
                    letter = chr(65 + idx)  # Label choices alphabetically from A.
                    formatted_candidates.append(f"{letter}. {cand}")

                    if isinstance(cand, str) and isinstance(gt_content, str):
                        if cand.strip() == gt_content.strip():
                            gt_letter = letter
                    else:
                        if cand == gt_content:
                            gt_letter = letter

            if gt_letter is None:
                gt_letter = gt_content

            batch_questions.append(question_text)
            batch_candidates.append(formatted_candidates)
            batch_loaded_data.append(loaded_data)
            batch_gt_letters.append(gt_letter)
            batch_meta.append({
                "task": item['task_folder'],
                "video": item['video'],
                "question": question_text,
                "question_type": question_type,
                "candidates": raw_candidates,
                "answer_content": gt_content,
                "answer_letter": gt_letter
            })

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
            os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    print(f">>> Evaluation finished. Saving results to {args.output_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    engine.close()


if __name__ == "__main__":
    main()
