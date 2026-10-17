import json
import argparse
import os
import re
from collections import defaultdict

def extract_option_from_response(response_text: str):
    """
    Extract the first A-Z option letter from a model response.
    The parser accepts both prefixed and standalone option outputs.
    """
    if not isinstance(response_text, str):
        return ""

    match = re.search(r"^\s*([A-Z])[\.\):]?", response_text.strip())
    if match:
        return match.group(1)

    match = re.search(r"[A-Z]", response_text)
    if match:
        return match.group(0)
    
    return ""

def calculate_mlvu_accuracy(results_path, txt_output_path):
    if not os.path.exists(results_path):
        print(f"Error: File not found -> '{results_path}'")
        return

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return

    stats_by_type = defaultdict(lambda: {"correct": 0, "total": 0})
    for item in results_data:
        q_type = item.get('question_type', item.get('task', 'unknown_type'))
        gt_letter = item.get('answer_letter', '').strip()
        model_response = item.get('response', '')
        if not gt_letter: 
            continue
        extracted_letter = extract_option_from_response(model_response)
        stats_by_type[q_type]["total"] += 1
        if extracted_letter == gt_letter:
            stats_by_type[q_type]["correct"] += 1

    lines = []
    lines.append("\n" + "="*60)
    lines.append(f"{'MLVU Task Type':<25} | {'Accuracy':<12} | {'Correct/Total'}")
    lines.append("-" * 60)

    cat_accs = []
    total_correct = 0
    total_questions = 0

    for q_type in sorted(stats_by_type.keys()):
        stat = stats_by_type[q_type]
        acc = (stat['correct'] / stat['total'] * 100) if stat['total'] > 0 else 0
        cat_accs.append(acc)
        total_correct += stat['correct']
        total_questions += stat['total']
        lines.append(f"{q_type:<25} | {acc:>10.2f}% | {stat['correct']:>4}/{stat['total']:<4}")

    micro_avg = (total_correct / total_questions * 100) if total_questions > 0 else 0
    macro_avg = sum(cat_accs) / len(cat_accs) if cat_accs else 0

    lines.append("-" * 60)
    lines.append(f"{'Overall (Micro Avg)':<25} | {micro_avg:>10.2f}% | {total_correct:>4}/{total_questions:<4}")
    lines.append(f"{'Overall (Macro Avg)':<25} | {macro_avg:>10.2f}% | (Avg of categories)")
    lines.append("="*60 + "\n")

    output_content = "\n".join(lines)
    print(output_content)

    if txt_output_path:
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"Full report saved to: {txt_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MLVU MCQ accuracy.")
    parser.add_argument("--input", type=str, required=True, help="Path to the prediction json file.")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to save the summary text file.")
    args = parser.parse_args()
    
    calculate_mlvu_accuracy(args.input, args.output_txt)
