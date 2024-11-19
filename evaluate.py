import os
import pandas as pd
import matplotlib.pyplot as plt
from ranx import Qrels, Run, evaluate
import numpy as np

# Define file mappings for basic and fine-tuned models, with only test files
FILE_MAP = {
    "basic": {
        "bi_encoder": "results/result_bi_1.tsv",
        "cross_encoder": "results/result_ce_1.tsv",
    },
    "fine_tuned": {
        "bi_encoder": "results/result_bi_ft_1.tsv",
        "cross_encoder": "results/result_ce_ft_1.tsv",
    },
}

# Metrics to evaluate
METRICS = ["precision@1", "precision@5", "ndcg@5", "mrr", "map"]

# Function to evaluate a model and return metrics
def evaluate_model(qrels, run_file, model_name, output_file):
    run = Run.from_file(run_file, kind="trec")
    print(f"Evaluating {model_name} model with file {run_file}...")

    results = evaluate(qrels, run, metrics=METRICS, make_comparable=True)
    print(f"Metrics for {model_name}:\n", results)

    df = pd.DataFrame([results])
    df.to_csv(output_file, index=False)

    return df, results

# Function to generate the ski-jump plot based on P@5
def plot_ski_jump(qrels, run_file, model_name, mode, output_dir):
    run = Run.from_file(run_file, kind="trec")
    p_at_5_results = evaluate(qrels, run, metrics=["precision@5"], return_mean=False)

    # Prepare data for plotting
    query_ids = list(qrels.qrels.keys())
    per_query_p_at_5 = p_at_5_results

    sorted_data = sorted(zip(query_ids, per_query_p_at_5), key=lambda x: x[1])
    sorted_query_ids, sorted_p_at_5 = zip(*sorted_data)

    # Add jitter for clarity in the plot
    jitter = np.random.uniform(-0.01, 0.01, size=len(sorted_p_at_5))
    sorted_p_at_5 = np.array(sorted_p_at_5) + jitter

    # Plot
    plt.figure(figsize=(14, 6))  # Increase the width of the figure
    plt.plot(sorted_query_ids, sorted_p_at_5, 'o')
    plt.title(f'Ski-Jump Plot for P@5 ({model_name.capitalize()} {mode.capitalize()} Model)')
    plt.xlabel('Query IDs')
    plt.ylabel('P@5')

    # Reduce the number of x-ticks for clarity (display every 10th query)
    step = max(1, len(sorted_query_ids) // 10)
    plt.xticks(ticks=range(0, len(sorted_query_ids), step), labels=sorted_query_ids[::step], rotation=70)

    plt.grid(True)
    plt.tight_layout()
    
    # Save plot with descriptive filename
    plot_file = os.path.join(output_dir, f"{model_name}_{mode}_ski_jump_plot.png")
    plt.savefig(plot_file)
    print(f"Ski-jump plot saved as {plot_file}")
    plt.show()

# Main function to evaluate models for the test set
def main(qrel_file, output_dir, mode):
    qrels = Qrels.from_file(qrel_file, kind="trec")

    # Filter qrels for relevance scores of 1 and 2
    filtered_qrels_dict = {
        query_id: {doc_id: score for doc_id, score in doc_scores.items() if score in [1, 2]}
        for query_id, doc_scores in qrels.qrels.items()
    }
    filtered_qrels = Qrels(filtered_qrels_dict)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Retrieve the correct file mappings based on the mode (basic or fine-tuned)
    model_files = FILE_MAP[mode]

    # Evaluate and plot for each model in the test set only
    for model_name, run_file in model_files.items():
        output_file = os.path.join(output_dir, f"{model_name}_metrics_{mode}.csv")

        # Evaluate the model and save metrics
        df, results = evaluate_model(filtered_qrels, run_file, model_name, output_file)

        # Generate ski-jump plot for P@5
        plot_ski_jump(filtered_qrels, run_file, model_name, mode, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate basic and fine-tuned retrieval models.")
    parser.add_argument('--qrel_file', type=str, required=True, help="Path to the Qrel file (e.g., test_qrel).")
    parser.add_argument('--output_dir', type=str, default="evaluation_results", help="Directory to save evaluation results.")
    parser.add_argument('--mode', type=str, choices=['basic', 'fine_tuned'], required=True, help="Mode of evaluation: 'basic' or 'fine_tuned'.")
    args = parser.parse_args()

    main(args.qrel_file, args.output_dir, args.mode)
