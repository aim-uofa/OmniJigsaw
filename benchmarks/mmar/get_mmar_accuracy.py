import argparse
import json
import re
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Calculate accuracy for MMAR benchmark results.")
    parser.add_argument('--input', type=str, required=True, help='Path to the evaluated results JSON file')
    parser.add_argument('--output_txt', type=str, default=None, help='Path to save the summary report')
    
    args = parser.parse_args()  
    
    output_buffer = []

    def log_print(msg=""):
        print(msg)
        output_buffer.append(str(msg))

    with open(args.input, 'r') as f:
        input_data = json.load(f)

    corr, total = 0, 0

    modality_metrics = {
        'sound': [0, 0], 
        'music': [0, 0], 
        'speech': [0, 0], 
        'mix-sound-music': [0, 0], 
        'mix-sound-speech': [0, 0], 
        'mix-music-speech': [0, 0], 
        'mix-sound-music-speech': [0, 0]
    }
    
    category_metrics = {
        'Signal Layer': [0, 0], 
        'Perception Layer': [0, 0], 
        'Semantic Layer': [0, 0], 
        'Cultural Layer': [0, 0]
    }
    
    subcat_metrics = {}

    output_key = 'model_output'
    no_pred_count = 0

    log_print(f">>> Processing MMAR Results: {args.input}")

    for idx, sample in enumerate(tqdm(input_data, desc="Analyzing")):

        if output_key not in sample:
            continue
        
        if output_key not in sample:
            _prediction = ''
            no_pred_count += 1
        else:
            _prediction = sample[output_key]

        _answer = sample['answer']
        modality = sample.get('modality', 'unknown')
        category = sample.get('category', 'unknown')
        choices = sample['choices']
        
        subcat = sample.get('sub-category', None)
        if subcat and subcat not in subcat_metrics:
            subcat_metrics[subcat] = [0, 0]

        match_result = string_match(_answer, _prediction, choices)

        if match_result:
            if modality in modality_metrics:
                modality_metrics[modality][0] += 1
            if category in category_metrics:
                category_metrics[category][0] += 1
            if subcat:
                subcat_metrics[subcat][0] += 1
            corr += 1

        total += 1
        if modality in modality_metrics:
            modality_metrics[modality][1] += 1
        if category in category_metrics:
            category_metrics[category][1] += 1
        if subcat:
            subcat_metrics[subcat][1] += 1

    
    log_print("\n" + "*"*40)
    log_print(f"{'MMAR Modality-wise Accuracy':^40}")
    log_print("*"*40)
    for m in sorted(modality_metrics.keys()):
        n_corr, n_tot = modality_metrics[m]
        acc = (n_corr / n_tot) * 100 if n_tot > 0 else 0
        if n_tot > 0:
            log_print(f"{m.ljust(25)} : {acc:>6.2f}% ({n_corr}/{n_tot})")

    log_print("\n" + "*"*40)
    log_print(f"{'MMAR Category-wise Accuracy':^40}")
    log_print("*"*40)
    for c in sorted(category_metrics.keys()):
        n_corr, n_tot = category_metrics[c]
        acc = (n_corr / n_tot) * 100 if n_tot > 0 else 0
        if n_tot > 0:
            log_print(f"{c.ljust(25)} : {acc:>6.2f}% ({n_corr}/{n_tot})")

    log_print("\n" + "*"*40)
    log_print(f"{'MMAR Sub-category-wise Accuracy':^40}")
    log_print("*"*40)
    for sc in sorted(subcat_metrics.keys()):
        n_corr, n_tot = subcat_metrics[sc]
        acc = (n_corr / n_tot) * 100 if n_tot > 0 else 0
        log_print(f"{sc.ljust(25)} : {acc:>6.2f}% ({n_corr}/{n_tot})")

    log_print("\n" + "="*40)
    overall_acc = (corr / total) * 100 if total > 0 else 0
    log_print(f"OVERALL ACCURACY : {overall_acc:.2f}%")
    log_print(f"TOTAL SAMPLES    : {total}")
    log_print(f"INVALID SAMPLES  : {no_pred_count}")
    log_print("="*40)
    
    if args.output_txt:
        try:
            with open(args.output_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_buffer))
            print(f"\n[Success] Summary report saved to: {args.output_txt}")
        except Exception as e:
            print(f"\n[Error] Failed to write TXT: {e}")
