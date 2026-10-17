import json
import argparse
import os
from collections import defaultdict

def load_jsonl_data(jsonl_path):
    """Load data from the provided JSONL file."""
    records = []
    if not os.path.exists(jsonl_path):
        print(f"Error: File {jsonl_path} does not exist.")
        return records

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_number}: {e}")
    return records

def calculate_accuracy_per_task_and_category(data):
    """Calculate accuracy for each unique task and category."""
    exclude_ids = {
        "stress_based_reasoning_ad6bdb44-58c4-43bf-bad3-a63d7d17028a",
        "stress_based_reasoning_b81cd3d3-00c4-47c4-a946-e30aec355015",
        "stress_based_reasoning_b64d5976-8dcd-4f7d-8411-9bbe90a51667",
        "dialogue_reasoning_1855b758-b438-43e4-8581-6d53580c4c7b"
    }

    task_category_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    
    total_correct = 0
    total_count = 0

    for record in data:
        record_id = record.get('id', '')
        if record_id in exclude_ids:
            continue

        task = record.get('category', 'Unknown Task')
        category = record.get('sub-category', 'Unknown Sub-category')

        answer_gt = record.get('answer_gt', '')
        if answer_gt == "middle aged adult":
            answer_gt = "Middle-aged adult"
        elif answer_gt == "elderly adult":
            answer_gt = "Elderly adult"

        response = record.get('response', '')
        model_predict = None
        
        try:
            predict = str(response).strip().replace('\n', '')
            if predict:
                if predict[0] in ['A', 'B', 'C', 'D']:
                    model_predict = predict[0]
                elif len(predict) > 1 and predict[-2] in ['A', 'B', 'C', 'D']:
                    model_predict = predict[-2]
                elif len(predict) == 1 and predict[0] in ['A', 'B', 'C', 'D']:
                    model_predict = predict[0]
        except:
            continue

        if not model_predict:
            task_category_stats[task][category]["total"] += 1
            total_count += 1
            continue

        choices = {
            'A': str(record.get('choice_a', '')),
            'B': str(record.get('choice_b', '')),
            'C': str(record.get('choice_c', '')),
            'D': str(record.get('choice_d', ''))
        }
        
        if choices.get(model_predict) == answer_gt:
            task_category_stats[task][category]["correct"] += 1
            total_correct += 1
                
        task_category_stats[task][category]["total"] += 1
        total_count += 1

    final_task_stats = {}
    for task, categories in task_category_stats.items():
        task_correct = 0
        task_total = 0
        cat_accuracies = {}
        for cat, counts in categories.items():
            acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            cat_accuracies[cat] = {"accuracy": acc, "correct": counts["correct"], "total": counts["total"]}
            task_correct += counts["correct"]
            task_total += counts["total"]
        
        final_task_stats[task] = {
            "sub_categories": cat_accuracies,
            "task_avg": task_correct / task_total if task_total > 0 else 0,
            "task_correct": task_correct,
            "task_total": task_total
        }

    overall_accuracy = total_correct / total_count if total_count > 0 else 0

    return final_task_stats, overall_accuracy, total_count

def main():
    parser = argparse.ArgumentParser(description="Process a JSONL file and calculate MMSU accuracy.")
    parser.add_argument('jsonl_path', type=str, help="Path to the input JSONL file")
    parser.add_argument('--output_txt', type=str, default=None, help="Path to save the evaluation report")
    args = parser.parse_args()

    output_buffer = []

    def log_print(msg=""):
        print(msg)
        output_buffer.append(str(msg))

    data = load_jsonl_data(args.jsonl_path)
    task_stats, overall_accuracy, total_count = calculate_accuracy_per_task_and_category(data)

    log_print("="*80)
    log_print(f"{'MMSU Evaluation Report':^80}")
    log_print("="*80)
    log_print(f"Input File: {args.jsonl_path}")
    log_print(f"Total Samples (after exclusion): {total_count}")
    log_print("-" * 80)

    for task in sorted(task_stats.keys()):
        log_print(f"\n[Category: {task}]")
        sub_cats = task_stats[task]["sub_categories"]
        for sub_cat in sorted(sub_cats.keys()):
            info = sub_cats[sub_cat]
            log_print(f"  - {sub_cat.ljust(35)}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")
        
        avg = task_stats[task]["task_avg"]
        log_print(f"  > {task} Average Accuracy: {avg:.4f}")

    log_print("\n" + "="*80)
    log_print(f"OVERALL ACCURACY: {overall_accuracy:.4f}")
    log_print(f"TOTAL SAMPLES   : {total_count}")
    log_print("="*80)

    if args.output_txt:
        try:
            with open(args.output_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_buffer))
            print(f"\n[Success] Report saved to: {args.output_txt}")
        except Exception as e:
            print(f"\n[Error] Failed to save TXT: {e}")

if __name__ == "__main__":
    main()
