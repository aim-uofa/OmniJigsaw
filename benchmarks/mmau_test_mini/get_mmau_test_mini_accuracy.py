import argparse
import json
import pickle
from tqdm import tqdm
from pathlib import Path
import re

def string_match(answer, prediction, choices):
    # Normalize text to case-insensitive word sets.
    def tokenize(text):
        return set(re.findall(r'\b\w+\b', text.lower()))
    
    prediction_tokens = tokenize(prediction)
    answer_tokens = tokenize(answer)
    
    if not prediction_tokens:
        return False
    
    # Exclude words shared with the correct answer from competing choices.
    incorrect_tokens = set()
    for choice in choices:
        choice_tokens = tokenize(choice)
        if choice_tokens != answer_tokens:
            incorrect_tokens.update(choice_tokens - answer_tokens)
    
    # Require every correct-answer token in the prediction.
    cond1 = answer_tokens.issubset(prediction_tokens)
    
    # Reject tokens specific to incorrect choices.
    cond2 = prediction_tokens.isdisjoint(incorrect_tokens)
    
    return cond1 and cond2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process benchmark JSON and calculate accuracy.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file to be evaluated')
    parser.add_argument('--output_txt', type=str, default=None, help='Path to save the summary report')
    
    args = parser.parse_args()  
    
    output_buffer = []

    def log_print(msg=""):
        print(msg)
        output_buffer.append(str(msg))
        
    with open(args.input, 'r') as f:
        input_data = json.load(f)

    corr, total = 0, 0

    # Track correct and total counts by task and difficulty.
    task_metrics = {'sound': [0, 0], 'music': [0, 0], 'speech': [0, 0]}
    diff_metrics = {'easy': [0, 0], 'hard': [0, 0], 'medium': [0, 0]}
    
    # Track correct and total counts for each subcategory.
    subcat_metrics = {}

    output_key = 'model_output'
    no_pred_count = 0
    matched_outputs = []
    new_data = []

    for idx, sample in enumerate(tqdm(input_data)):
        
        # Skip samples without model predictions.
        if output_key not in sample:
            continue
        
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        task = sample['task']
        difficulty = sample['difficulty']
        choices = sample['choices']
        
        subcat = sample.get('sub-category', None)
        if subcat is not None:
            # Initialize counts for an unseen subcategory.
            if subcat not in subcat_metrics:
                subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            task_metrics[task][0] += 1
            diff_metrics[difficulty][0] += 1
            if subcat is not None:
                subcat_metrics[subcat][0] += 1
            matched_outputs.append([_answer, _prediction])
            corr += 1
            sample['match'] = 1
        else:
            sample['match'] = 0

        total += 1
        new_data.append(sample)
        task_metrics[task][1] += 1
        diff_metrics[difficulty][1] += 1
        if subcat is not None:
            subcat_metrics[subcat][1] += 1


    # Report accuracy by task, difficulty, and subcategory.
    log_print("*"*30)
    log_print("Task-wise Accuracy:")
    for task in task_metrics:
        n_correct, n_total = task_metrics[task]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        log_print(f"{task} : {acc:.2f}% over {n_total} samples")
    
    log_print("*"*30)
    log_print("Difficulty-wise Accuracy:")
    for diff in diff_metrics:
        n_correct, n_total = diff_metrics[diff]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        log_print(f"{diff} : {acc:.2f}% over {n_total} samples")
    
    log_print("*"*30)
    log_print("Sub-category-wise Accuracy:")
    for subcat in subcat_metrics:
        n_correct, n_total = subcat_metrics[subcat]
        acc = (n_correct / n_total) * 100 if n_total > 0 else 0
        log_print(f"{subcat} : {acc:.2f}% over {n_total} samples")

    log_print("*"*30)
    log_print(f"Total Accuracy: {(corr/total) * 100:.2f}% over {total} samples")
    log_print("*"*30)
    log_print(f"No prediction count: {no_pred_count}")
    
    if args.output_txt:
        try:
            with open(args.output_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_buffer))
            print(f"\n[Success] Report saved to: {args.output_txt}")
        except Exception as e:
            print(f"\n[Error] Failed to save TXT: {e}")
