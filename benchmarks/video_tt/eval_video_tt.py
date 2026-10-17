import os
import json
import argparse
import re
import pandas as pd
from tqdm import tqdm
import torch
import gc

from infer.inference_engine import TTSInferenceEngine, InferenceEngine
from infer.video_utils import load_video_and_audio_data, load_video, load_video_and_audio_data_batch, load_video_batch


def extract_pure_question_and_candidates(raw_question_text):
    pattern = r'(?:\n|^)([A-Z])\.\s+(.*?)(?=\s*\n[A-Z]\.|\nAnswer with|$)'
    matches = re.findall(pattern, raw_question_text, re.DOTALL)
    extracted_options = {label: content.strip() for label, content in matches}
    
    formatted_candidates = []
    labels = sorted(extracted_options.keys())
    for label in labels:
        content = extracted_options[label]
        clean_content = re.sub(r'^[A-Z]\.\s+', '', content)
        formatted_candidates.append(f"{label}. {clean_content}")

    if labels:
        first_label = labels[0]
        first_match = re.search(rf'(?:\n|^){first_label}\.\s+', raw_question_text)
        if first_match:
            pure_question = raw_question_text[:first_match.start()].strip()
        else:
            pure_question = raw_question_text.split('\n')[0].strip()
    else:
        pure_question = raw_question_text

    return pure_question, formatted_candidates


def parse_args():
    parser = argparse.ArgumentParser(description="Video-TT Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Root dir containing 'Benchmark-AllVideos-HQ-encoded' and 'data/test-00000-of-00001.parquet'")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save result JSON")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_num_seqs", type=int, default=8, help="Max number of sequences per iteration (batch size)")
    parser.add_argument("--max_model_len", type=int, default=65536, help="Maximum context length for the model")

    parser.add_argument("--num_clips", type=int, default=6, help="Number of clips for Jigsaw stage")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--disable_audio", action="store_true", help="Whether to disable audio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for data loading")

    return parser.parse_args()


def main():
    args = parse_args()

    parquet_file = os.path.join(args.benchmark_dir, "data/test-00000-of-00001.parquet")
    video_root_hq = os.path.join(args.benchmark_dir, "Benchmark-AllVideos-HQ-encoded")
    video_root_lq = os.path.join(args.benchmark_dir, "Benchmark-AllVideos-LQ")

    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    print(f">>> Loading Video-TT Data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    questions_data_raw = df.to_dict(orient='records')
        
    bad_video_keywords = set()
    
    questions_data = []
    filtered_count = 0
    for item in questions_data_raw:
        video_id = str(item.get('video_id', ''))
        if any(bad_name in video_id for bad_name in bad_video_keywords):
            filtered_count += 1
            continue
        questions_data.append(item)
        
    print(f">>> Data Filtering Complete:")
    print(f"    - Original samples: {len(questions_data_raw)}")
    print(f"    - Bad video samples removed: {filtered_count}")
    print(f"    - Final samples to evaluate: {len(questions_data)}")

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
            video_filename = f"{item['video_id']}.mp4"
            video_path = os.path.join(video_root_hq, video_filename)
            
            if not os.path.exists(video_path):
                video_path = os.path.join(video_root_lq, video_filename)

            if not os.path.exists(video_path):
                print(f"[Warning] Video missing in both HQ and LQ: {video_filename}, skipping qid {item['qid']}")
                continue

            video_paths.append(video_path)
            valid_items.append(item)

        if len(video_paths) == 0:
            continue

        if not args.disable_audio:
            load_results, load_errors = load_video_and_audio_data_batch(video_paths, num_workers=args.num_workers)
        else:
            load_results, load_errors = load_video_batch(video_paths, num_workers=args.num_workers)

        for i, item in enumerate(valid_items):
            if load_errors[i] is not None:
                print(f"[Error] Failed to load {item['video_id']}: {load_errors[i]}")
                continue

            loaded_data = load_results[i]
            
            raw_q = item['question']
            pure_q, formatted_cands = extract_pure_question_and_candidates(raw_q)
            formatted_cands.append("E. None of the above")
            
            gt_letter = item['answer']

            batch_questions.append(pure_q)
            batch_candidates.append(formatted_cands)
            batch_loaded_data.append(loaded_data)
            batch_gt_letters.append(gt_letter)
            batch_meta.append({
                "idx": item['qid'],
                "video_id": item['video_id'],
                "question_type": item['capability'],
                "question": pure_q,
                "candidates": formatted_cands,
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
