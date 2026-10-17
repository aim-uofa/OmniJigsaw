import os
import argparse
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import gc

from infer.inference_engine import InferenceEngine
from infer.video_utils import load_audio_with_av, load_audio_batch

def parse_args():
    parser = argparse.ArgumentParser(description="MMAU-Pro Audio Evaluation Script (Closed-Ended Only)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Root dir containing 'test.parquet' and 'data/' folder")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save result Parquet file")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_num_seqs", type=int, default=8, help="Max number of sequences per iteration")
    parser.add_argument("--max_model_len", type=int, default=65536, help="Maximum context length")

    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for data loading")

    return parser.parse_args()

def main():
    args = parse_args()

    parquet_file = os.path.join(args.benchmark_dir, "test.parquet")
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

    print(f">>> Loading MMAU-Pro Data from {parquet_file}")
    df = pd.read_parquet(parquet_file)
    original_len = len(df)

    print(">>> Filtering for Closed-Ended (Multiple Choice) questions only...")

    excluded_categories = ['open', 'instruction following']
    df = df[~df['category'].isin(excluded_categories)].copy()

    df = df[df['choices'].notna()]
    df = df[df['choices'].apply(lambda x: len(x) > 0 if isinstance(x, (list, np.ndarray)) else False)]

    print(f">>> Filtered data: {len(df)} samples (Dropped {original_len - len(df)} non-choice samples)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    engine = InferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 1},
        seed=args.seed
    )

    print(f">>> Starting Evaluation on {len(df)} samples...")

    success_indices = []
    model_outputs_list = []
    raw_responses_list = []

    batch_size = args.batch_size
    df_items = list(df.iterrows())

    for batch_start in tqdm(range(0, len(df_items), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(df_items))
        batch_items = df_items[batch_start:batch_end]

        batch_questions = []
        batch_candidates = []
        batch_loaded_data = []
        batch_gt_letters = []
        batch_meta = []

        audio_paths = []
        valid_items = []
        for index, row in batch_items:
            audio_rel_path = row['audio_path']
            if isinstance(audio_rel_path, (list, np.ndarray)):
                audio_rel_path = audio_rel_path[0]

            audio_full_path = os.path.join(args.benchmark_dir, audio_rel_path)

            if not os.path.exists(audio_full_path):
                print(f"[Warning] Audio missing: {audio_full_path}")
                continue

            audio_paths.append(audio_full_path)
            valid_items.append((index, row))

        if len(audio_paths) > 0:
            load_results, load_errors = load_audio_batch(
                audio_paths, target_sr=16000, num_workers=args.num_workers
            )
        else:
            load_results, load_errors = [], []

        for i, (index, row) in enumerate(valid_items):
            if load_errors[i] is not None:
                print(f"[Error] Failed to load row {index}: {load_errors[i]}")
                continue

            audio_wav = load_results[i]

            question_text = row['question']

            choices = row['choices']
            if isinstance(choices, np.ndarray):
                choices = choices.tolist()

            gt_content = row['answer']

            formatted_candidates = []
            gt_letter = None

            for idx, choice in enumerate(choices):
                letter = chr(65 + idx) # Label choices alphabetically from A.
                formatted_candidates.append(f"{letter}. {choice}")
                if isinstance(choice, str) and isinstance(gt_content, str):
                    if choice.strip() == gt_content.strip():
                        gt_letter = letter
                else:
                    if choice == gt_content:
                        gt_letter = letter

            batch_questions.append(question_text)
            batch_candidates.append(formatted_candidates)
            batch_loaded_data.append(audio_wav)
            batch_gt_letters.append(gt_letter)
            batch_meta.append({"choices": choices, "df_index": index})

        if len(batch_questions) > 0:
            try:
                batch_responses = engine.run_audio_inference_batch(
                    questions=batch_questions,
                    candidates_list=batch_candidates,
                    loaded_data_list=batch_loaded_data,
                    answer_idx_list=batch_gt_letters
                )
            except Exception as e:
                print(f"[Error] Batch inference failed: {e}")
                batch_responses = ["Error during generation"] * len(batch_questions)
        else:
            batch_responses = []

        for i, response_text in enumerate(batch_responses):
            final_output_content = response_text

            match = re.search(r"([A-Z])", response_text.strip())
            if match:
                pred_letter = match.group(1)
                idx = ord(pred_letter) - ord('A')
                choices = batch_meta[i]["choices"]
                if 0 <= idx < len(choices):
                    final_output_content = choices[idx]

            success_indices.append(batch_meta[i]["df_index"])
            model_outputs_list.append(final_output_content)
            raw_responses_list.append(response_text)

        del batch_loaded_data
        gc.collect()
        torch.cuda.empty_cache()

        if len(success_indices) % 20 == 0 and len(success_indices) > 0:
            current_df = df.loc[success_indices].copy()
            current_df['model_output'] = model_outputs_list
            current_df['model_raw_response'] = raw_responses_list
            current_df.to_parquet(args.output_file)

    result_df = df.loc[success_indices].copy()
    result_df['model_output'] = model_outputs_list
    result_df['model_raw_response'] = raw_responses_list

    print(f">>> Saving results to {args.output_file}")
    result_df.to_parquet(args.output_file)

    engine.close()
    print(">>> Done.")

if __name__ == "__main__":
    main()
