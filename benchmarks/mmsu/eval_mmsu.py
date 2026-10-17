import os
import json
import argparse
from tqdm import tqdm
import torch
import gc
import re

from infer.inference_engine import InferenceEngine
from infer.video_utils import load_audio_with_av, load_audio_batch


def parse_args():
    parser = argparse.ArgumentParser(description="MMSU (Audio) Evaluation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to HF model")
    parser.add_argument("--benchmark_dir", type=str, required=True, help="Root dir containing 'audio' and 'question/mmsu.jsonl'")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save result JSONL")

    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="vLLM GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_num_seqs", type=int, default=8, help="Max number of sequences per iteration (batch size)")
    parser.add_argument("--max_model_len", type=int, default=65536, help="Maximum context length for the model")

    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers for data loading")

    return parser.parse_args()


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def main():
    args = parse_args()

    jsonl_file = os.path.join(args.benchmark_dir, "question", "mmsu.jsonl")
    audio_root = args.benchmark_dir

    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"Annotation file not found: {jsonl_file}")

    print(f">>> Loading MMSU Data from {jsonl_file}")
    data_list = load_jsonl(jsonl_file)

    engine = InferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 1},
        seed=args.seed
    )

    results = []
    print(f">>> Starting Evaluation on {len(data_list)} samples...")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(data_list), batch_size), desc="Evaluating batches"):
        batch_end = min(batch_start + batch_size, len(data_list))
        batch_items = data_list[batch_start:batch_end]

        batch_questions = []
        batch_candidates = []
        batch_loaded_data = []
        batch_meta = []

        audio_paths = []
        valid_items = []
        for idx_in_batch, item in enumerate(batch_items):
            rel_path = item['audio_path']
            if rel_path.startswith("/"):
                rel_path = rel_path[1:]

            audio_path = os.path.join(audio_root, rel_path)

            if not os.path.exists(audio_path):
                idx = batch_start + idx_in_batch
                print(f"[Warning] Audio missing: {audio_path}, skipping sample {idx}")
                continue

            audio_paths.append(audio_path)
            valid_items.append(item)

        if len(audio_paths) == 0:
            continue

        load_results, load_errors = load_audio_batch(
            audio_paths, target_sr=16000, num_workers=args.num_workers
        )

        for i, item in enumerate(valid_items):
            if load_errors[i] is not None:
                print(f"[Error] Failed to load {audio_paths[i]}: {load_errors[i]}")
                continue

            audio_wav = load_results[i]

            qid = item['id']
            question_text = item['question']
            choices = [
                item.get('choice_a', ''),
                item.get('choice_b', ''),
                item.get('choice_c', ''),
                item.get('choice_d', '')
            ]
            valid_choices = [c for c in choices if c]

            formatted_candidates = []

            for j, choice in enumerate(valid_choices):
                letter = chr(65 + j)
                formatted_candidates.append(f"{letter}. {choice}")

            batch_questions.append(question_text)
            batch_candidates.append(formatted_candidates)
            batch_loaded_data.append(audio_wav)
            batch_meta.append(item.copy())

        if len(batch_questions) == 0:
            continue

        try:
            batch_responses = engine.run_audio_inference_batch(
                questions=batch_questions,
                candidates_list=batch_candidates,
                loaded_data_list=batch_loaded_data
            )
        except Exception as e:
            print(f"[Error] Batch inference failed: {e}")
            batch_responses = ["Error during generation"] * len(batch_questions)

        for i, response_text in enumerate(batch_responses):
            res_item = batch_meta[i].copy()
            res_item['response'] = response_text
            results.append(res_item)

        del batch_loaded_data
        gc.collect()
        torch.cuda.empty_cache()

        if len(results) % 20 == 0:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")

    print(f">>> Evaluation finished. Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    engine.close()


if __name__ == "__main__":
    main()
