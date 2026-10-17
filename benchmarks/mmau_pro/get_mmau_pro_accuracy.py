import pandas as pd
import argparse
import os
from collections import defaultdict

def calculate_mmau_pro_mcq(input_parquet, output_txt):
    output_buffer = []

    def log_print(msg=""):
        print(msg)
        output_buffer.append(str(msg))

    if not os.path.exists(input_parquet):
        print(f"Error: File not found -> {input_parquet}")
        return

    log_print(f"Loading results from: {input_parquet}")
    df = pd.read_parquet(input_parquet)

    mcq_df = df[df['choices'].notna()].copy()
    
    mcq_df = mcq_df[~mcq_df['category'].isin(['open', 'instruction following'])]

    total_samples = len(mcq_df)
    if total_samples == 0:
        log_print("No MCQ samples found in the dataset.")
        return

    log_print(f"Total MCQ samples to evaluate: {total_samples}")

    def check_correct(row):
        gt = str(row['answer']).strip().lower()
        pred = str(row['model_output']).strip().lower()
        return gt == pred

    mcq_df['is_correct'] = mcq_df.apply(check_correct, axis=1)

    category_stats = mcq_df.groupby('category')['is_correct'].agg(['sum', 'count']).to_dict('index')

    log_print("\n" + "="*60)
    log_print(f"{'MMAU-Pro Category':<20} | {'Accuracy':<12} | {'Correct/Total'}")
    log_print("-" * 60)

    total_correct = 0
    for cat, stat in sorted(category_stats.items()):
        acc = (stat['sum'] / stat['count'] * 100)
        total_correct += stat['sum']
        log_print(f"{cat:<20} | {acc:>10.2f}% | {int(stat['sum']):>4}/{int(stat['count']):<4}")

    overall_acc = (total_correct / total_samples * 100)
    log_print("-" * 60)
    log_print(f"{'Overall MCQ':<20} | {overall_acc:>10.2f}% | {int(total_correct):>4}/{total_samples:<4}")
    log_print("="*60 + "\n")

    if output_txt:
        try:
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write("\n".join(output_buffer))
            print(f"Report successfully saved to: {output_txt}")
        except Exception as e:
            print(f"Error saving TXT file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified MCQ Evaluation for MMAU-Pro")
    parser.add_argument("--input", type=str, required=True, help="Path to mmau_pro_results.parquet")
    parser.add_argument("--output_txt", type=str, required=True, help="Path to save the summary report")
    args = parser.parse_args()

    calculate_mmau_pro_mcq(args.input, args.output_txt)
