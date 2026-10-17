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


def calculate_mlvu_accuracy(results_path: str, output_txt: str = None):
    """
    Compute MLVU accuracy grouped by question type.
    """
    
    output_buffer = []

    def log_print(msg=""):
        print(msg)
        output_buffer.append(str(msg))
    
    if not os.path.exists(results_path):
        log_print(f"Error: results file not found: '{results_path}'")
        return

    log_print(f"Loading evaluation results: {results_path}")

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)

        if not isinstance(results_data, list):
            log_print("Error: the top-level JSON value must be a list.")
            return

    except json.JSONDecodeError:
        log_print(f"Error: invalid JSON file: '{results_path}'")
        return
    except Exception as e:
        log_print(f"Error while reading the results file: {e}")
        return
        
    total_questions = len(results_data)
    if total_questions == 0:
        log_print("Warning: the JSON file contains no evaluation entries.")
        return

    # Track correct and answered counts by question type.
    stats_by_type = defaultdict(lambda: {"correct": 0, "answered": 0})

    for item in results_data:
        if not isinstance(item, dict) or 'answer_letter' not in item or 'response' not in item:
            continue
            
        question_type = item.get('question_type', 'unknown_type')

        ground_truth_letter = item['answer_letter']
        model_response = item['response']

        stats_by_type[question_type]["answered"] += 1

        extracted_letter = extract_option_from_response(model_response)

        if extracted_letter and extracted_letter == ground_truth_letter:
            stats_by_type[question_type]["correct"] += 1

    log_print("\n--- Evaluation Complete ---")
    log_print(f"Total questions: {total_questions}")
    
    total_correct = 0
    total_answered = 0
    
    log_print("\n-------------------------------------")
    log_print("Accuracy by Question Type:")
    log_print("-------------------------------------")
    
    for q_type, stats in sorted(stats_by_type.items()):
        answered = stats['answered']
        correct = stats['correct']
        
        total_answered += answered
        total_correct += correct
        
        accuracy = (correct / answered * 100) if answered > 0 else 0.0
        log_print(f"{q_type.ljust(25)}: {accuracy:.2f}%  ({correct}/{answered})")

    log_print("\n-------------------------------------")
    log_print("Overall Performance")
    log_print("-------------------------------------")
    
    overall_accuracy = (total_correct / total_answered * 100) if total_answered > 0 else 0.0
    log_print(f"Answered questions: {total_answered}")
    log_print(f"Correct answers: {total_correct}")
    log_print(f"Overall accuracy (correct / answered): {overall_accuracy:.2f}%")
    
    if output_txt:
        try:
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_buffer))
            print(f"\n[Success] Evaluation report saved to: {output_txt}")
        except Exception as e:
            print(f"\n[Error] Failed to save the report: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate accuracy for MLVU benchmark results, categorized by question_type."
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the JSON results file (e.g., 'mlvu_test_results.json')."
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Path to save the summary output text file."
    )
    

    args = parser.parse_args()
    
    calculate_mlvu_accuracy(args.results_file, args.output_txt)
