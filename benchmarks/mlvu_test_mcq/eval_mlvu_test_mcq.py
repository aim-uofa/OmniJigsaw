import os
import json
import argparse
from tqdm import tqdm
import torch
import gc

from infer.inference_engine import TTSInferenceEngine, InferenceEngine
from infer.video_utils import load_video_and_audio_data, load_video, load_video_and_audio_data_batch, load_video_batch


def parse_args():
    parser = argparse.ArgumentParser(description="MLVU Test (MCQ) Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Root dir containing 'MLVU_Test', 'test_multi_choice_tasks.json'")
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

    question_file = os.path.join(args.benchmark_dir, "test_multi_choice_tasks.json")
    gt_file = os.path.join(args.benchmark_dir, "test-ground-truth", "test_mcq_gt.json")

    video_root = os.path.join(args.benchmark_dir, "MLVU_Test/video")

    if not os.path.exists(question_file):
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"GT file not found: {gt_file}")

    print(f">>> Loading MLVU Question Data from {question_file}")
    with open(question_file, 'r') as f:
        questions_data = json.load(f)

    print(f">>> Loading MLVU GT Data from {gt_file}")
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    gt_map = {item['question_id']: item['answer'] for item in gt_data}

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
    print(f">>> Starting Evaluation on {len(questions_data)} samples...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(questions_data), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(questions_data))
        batch_items = questions_data[batch_start:batch_end]

        batch_questions = []
        batch_candidates = []
        batch_loaded_data = []
        batch_gt_letters = []
        batch_meta = []

        video_paths = []
        valid_items = []
        for item in batch_items:
            qid = item['question_id']
            video_filename = item['video']
            video_path = os.path.join(video_root, video_filename)

            if not os.path.exists(video_path):
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
                video_filename = item['video']
                print(f"[Error] Failed to load {video_filename}: {load_errors[i]}")
                continue

            loaded_data = load_results[i]

            qid = item['question_id']
            question_text = item['question']
            raw_candidates = item['candidates']
            gt_content = gt_map.get(qid, "")

            formatted_candidates = []
            gt_letter = None

            for idx, cand in enumerate(raw_candidates):
                letter = chr(65 + idx) # Label choices alphabetically from A.
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
                "question_id": qid,
                "video": item['video'],
                "question_type": item['question_type'],
                "question": question_text,
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
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    print(f">>> Evaluation finished. Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    engine.close()


if __name__ == "__main__":
    main()
