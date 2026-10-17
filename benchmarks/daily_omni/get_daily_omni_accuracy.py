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


def calculate_daily_omni_accuracy(results_path: str, output_txt: str = None):
    """
    Compute Daily-Omni accuracy across its reporting dimensions.
    """
    output_buffer = []

    def log_print(msg=""):
        print(msg)
        output_buffer.append(str(msg))

    if not os.path.exists(results_path):
        log_print(f"Error: results file not found: '{results_path}'")
        return 1

    log_print(f"Loading evaluation results: {results_path}")

    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)

        if not isinstance(results_data, list):
            log_print("Error: the top-level JSON value must be a list.")
            return 1

    except json.JSONDecodeError:
        log_print(f"Error: invalid JSON file: '{results_path}'")
        return 1
    except Exception as e:
        log_print(f"Error while reading the results file: {e}")
        return 1

    total_questions = len(results_data)
    if total_questions == 0:
        log_print("Warning: the JSON file contains no evaluation entries.")
        return 1

    stats_by_type = defaultdict(lambda: {"correct": 0, "answered": 0})
    stats_by_video_category = defaultdict(lambda: {"correct": 0, "answered": 0})
    stats_by_duration = defaultdict(lambda: {"correct": 0, "answered": 0})

    total_answered = 0
    total_correct = 0

    for item in results_data:
        if not isinstance(item, dict) or 'Answer' not in item or 'response' not in item:
            continue

        ground_truth_letter = item['Answer']
        model_response = item['response']

        extracted_letter = extract_option_from_response(model_response)
        is_correct = extracted_letter == ground_truth_letter if extracted_letter else False

        qa_type = item.get('Type', 'unknown_type')
        video_category = item.get('video_category', 'unknown_category')
        video_duration = item.get('video_duration', 'unknown_duration')

        stats_by_type[qa_type]["answered"] += 1
        stats_by_video_category[video_category]["answered"] += 1
        stats_by_duration[video_duration]["answered"] += 1

        total_answered += 1
        if is_correct:
            stats_by_type[qa_type]["correct"] += 1
            stats_by_video_category[video_category]["correct"] += 1
            stats_by_duration[video_duration]["correct"] += 1
            total_correct += 1

    log_print("\n--- Evaluation Complete ---")
    log_print(f"Total questions: {total_questions}")

    log_print("\n-------------------------------------")
    log_print("Accuracy by QA Type:")
    log_print("-------------------------------------")

    for qa_type, stats in sorted(stats_by_type.items()):
        answered = stats['answered']
        correct = stats['correct']
        accuracy = (correct / answered * 100) if answered > 0 else 0.0
        log_print(f"{str(qa_type).ljust(25)}: {accuracy:.2f}%  ({correct}/{answered})")

    log_print("\n-------------------------------------")
    log_print("Accuracy by Video Category:")
    log_print("-------------------------------------")

    for video_category, stats in sorted(stats_by_video_category.items()):
        answered = stats['answered']
        correct = stats['correct']
        accuracy = (correct / answered * 100) if answered > 0 else 0.0
        log_print(f"{str(video_category).ljust(25)}: {accuracy:.2f}%  ({correct}/{answered})")

    log_print("\n-------------------------------------")
    log_print("Accuracy by Video Duration:")
    log_print("-------------------------------------")

    for duration, stats in sorted(stats_by_duration.items()):
        answered = stats['answered']
        correct = stats['correct']
        accuracy = (correct / answered * 100) if answered > 0 else 0.0
        log_print(f"{str(duration).ljust(10)}: {accuracy:.2f}%  ({correct}/{answered})")

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
            return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate accuracy for Daily-Omni results, categorized by QA type, video category, and duration."
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the JSON results file (e.g., 'daily_omni_results.json')."
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Path to save the summary output text file."
    )

    args = parser.parse_args()

    raise SystemExit(calculate_daily_omni_accuracy(args.results_file, args.output_txt))
