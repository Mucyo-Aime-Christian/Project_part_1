import pandas as pd
import ranx as rx
import argparse
import matplotlib.pyplot as plt

def load_qrels(qrel_file):
    """Load Qrels file."""
    return rx.Qrels.from_file(qrel_file)

def evaluate_model(qrels, run_file, output_file, model_name):
    """
    Evaluate the model using different metrics and save the results.

    Args:
        qrels (Qrels): Relevance judgments.
        run_file (str): Path to the run file containing ranked results.
        output_file (str): Path to the output CSV file for saving evaluation metrics.
        model_name (str): Name of the model being evaluated ('tf-idf' or 'bm25').
    """
    # Load the run results explicitly specifying the kind as 'trec'
    run = rx.Run.from_file(run_file, kind="trec")

    # Define the metrics to use for evaluation
    metrics = ["precision@1", "precision@5", "recall@5", "map", "ndcg@5", "mrr"]
    
    # Perform evaluation and make the Qrels and Run comparable
    results = rx.evaluate(qrels, run, metrics, make_comparable=True)

    # Save the evaluation metrics to a CSV file
    df_results = pd.DataFrame([results])
    df_results.to_csv(output_file, index=False)

    # Display evaluation results
    print(f"\n=== Evaluation Metrics for {model_name} ===")
    print(df_results)

def plot_precision_at_k(qrels, run_file, model_name):
    """
    Plot Precision@5 for each query to visualize the results in a ski-jump style.
    
    Args:
        qrels (Qrels): Relevance judgments.
        run_file (str): Path to the run file.
        model_name (str): Name of the model being evaluated.
    """
    import numpy as np

    # Load the run
    run = rx.Run.from_file(run_file, kind="trec")

    # Compute Precision@5 for each query using `evaluate()` with `make_comparable=True`
    results_per_query = rx.evaluate(qrels, run, metrics=["precision@5"], return_mean=False, make_comparable=True)

    # Get the list of query IDs
    query_ids = list(qrels.qrels.keys())

    # Sort the query IDs and corresponding Precision@5 values for a ski-jump effect
    sorted_data = sorted(zip(query_ids, results_per_query), key=lambda x: x[1])
    sorted_query_ids, sorted_precision = zip(*sorted_data)

    # Adding jitter to the precision values to avoid overlapping
    jitter = np.random.uniform(-0.005, 0.005, size=len(sorted_precision))
    sorted_precision = np.array(sorted_precision) + jitter

    # Create a DataFrame to store the results for easier manipulation
    df_p_at_5 = pd.DataFrame({'Query_ID': sorted_query_ids, 'P@5': sorted_precision})

    # Create the scatter plot
    plt.figure(figsize=(12, 7))
    plt.scatter(df_p_at_5['Query_ID'], df_p_at_5['P@5'], s=60, c='blue', alpha=0.8, edgecolors='black', linewidth=0.6, label=model_name)
    plt.title(f'Ski-Jump Plot for P@5 ({model_name} Model)')
    plt.xlabel('Topic (Query IDs)')
    plt.ylabel('P@5')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend()

    # Limit the number of x-ticks for better readability
    step = max(1, len(df_p_at_5['Query_ID']) // 15)
    plt.xticks(ticks=range(0, len(df_p_at_5['Query_ID']), step), labels=df_p_at_5['Query_ID'][::step], rotation=45)

    # Tight layout and saving the plot
    plt.tight_layout()
    plot_file = f"{model_name.lower()}_ski_jump_plot_sorted_by_precision.png"
    plt.savefig(plot_file)
    print(f"Ski-jump plot saved as {plot_file}")
    plt.show()

def main(qrel_file, run_file_tfidf, run_file_bm25, output_file_tfidf, output_file_bm25):
    """
    Main function to evaluate both TF-IDF and BM25 models.
    
    Args:
        qrel_file (str): Path to the Qrels file.
        run_file_tfidf (str): Path to the run file for TF-IDF model.
        run_file_bm25 (str): Path to the run file for BM25 model.
        output_file_tfidf (str): Path to save evaluation metrics for TF-IDF model.
        output_file_bm25 (str): Path to save evaluation metrics for BM25 model.
    """
    # Load the relevance judgments (Qrels)
    print("Loading Qrels...")
    qrels = load_qrels(qrel_file)

    # Evaluate and save results for TF-IDF
    print("\nEvaluating TF-IDF model...")
    evaluate_model(qrels, run_file_tfidf, output_file_tfidf, "tf-idf")

    # Evaluate and save results for BM25
    print("\nEvaluating BM25 model...")
    evaluate_model(qrels, run_file_bm25, output_file_bm25, "bm25")

    # Optional: plot Precision@5 for visual comparison
    plot_precision_at_k(qrels, run_file_tfidf, "TF-IDF")
    plot_precision_at_k(qrels, run_file_bm25, "BM25")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF and BM25 models.")
    parser.add_argument('--qrel_file', type=str, required=True, help="Path to the Qrels file (TSV format).")
    parser.add_argument('--run_file_tfidf', type=str, required=True, help="Path to the run file for TF-IDF (TSV format).")
    parser.add_argument('--run_file_bm25', type=str, required=True, help="Path to the run file for BM25 (TSV format).")
    parser.add_argument('--output_file_tfidf', type=str, required=True, help="Path to save evaluation metrics for TF-IDF (CSV format).")
    parser.add_argument('--output_file_bm25', type=str, required=True, help="Path to save evaluation metrics for BM25 (CSV format).")

    args = parser.parse_args()
    main(args.qrel_file, args.run_file_tfidf, args.run_file_bm25, args.output_file_tfidf, args.output_file_bm25)
