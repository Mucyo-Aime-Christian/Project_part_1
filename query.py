import pyterrier as pt
import pandas as pd
import argparse
import os
from load_data import load_answers, clean_query

# Ensure PyTerrier is started
if not pt.started():
    pt.init()

def save_results_in_trec_format(results, output_file, model_name):
    """Save the retrieval results in TREC format without headers."""
    with open(output_file, 'w') as f:
        for index, row in results.iterrows():
            f.write(f"{row['qid']} Q0 {row['docno']} {row['rank']} {row['score']} {model_name}\n")

def print_top_answers(results, answers_dict, model_name, top_n=5):
    """Print the top N most relevant answers."""
    print(f"\nTop {top_n} results from {model_name}:")
    for i in range(min(top_n, len(results))):
        doc_id = results.iloc[i]['docno']
        rank = results.iloc[i]['rank']
        score = results.iloc[i]['score']
        answer_text = answers_dict.get(str(doc_id), "Answer not found")
        print(f"Rank {rank}, Score: {score}")
        print(f"Answer: {answer_text}\n")

def retrieve_single_query(index_path, query_text, answers_file, results_dir):
    """Retrieve documents for a single query and print the top results."""
    # Load the existing index
    index_ref = pt.IndexRef.of(os.path.join(index_path, "data.properties"))
    index = pt.IndexFactory.of(index_ref)

    # Clean the query and prepare it
    cleaned_query = clean_query(query_text)
    query_id = "user_query_1"  # Unique ID for this user query
    queries_df = pd.DataFrame([[query_id, cleaned_query]], columns=["qid", "query"])

    # Load the answers for reference
    answers = load_answers(answers_file)
    answers_dict = {str(answer['Id']): answer['Text'] for answer in answers}

    # Perform retrieval using BM25 and TF-IDF
    print("Running retrieval for user query...")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25", num_results=20)
    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF", num_results=20)

    results_bm25 = bm25.transform(queries_df)
    results_tfidf = tfidf.transform(queries_df)

    # Print top 5 results for BM25 and TF-IDF
    print_top_answers(results_bm25, answers_dict, "BM25", top_n=5)
    print_top_answers(results_tfidf, answers_dict, "TF-IDF", top_n=5)

    # Save results for evaluation
    os.makedirs(results_dir, exist_ok=True)
    bm25_results_file = os.path.join(results_dir, 'bm25_user_results.trec')
    tfidf_results_file = os.path.join(results_dir, 'tfidf_user_results.trec')
    save_results_in_trec_format(results_bm25, bm25_results_file, 'bm25')
    save_results_in_trec_format(results_tfidf, tfidf_results_file, 'tf-idf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve documents using BM25 and TF-IDF models.")
    parser.add_argument('--index_path', type=str, required=True, help="Path to the directory containing the index.")
    parser.add_argument('--query', type=str, required=True, help="The user's query text.")
    parser.add_argument('--answers_file', type=str, required=True, help="Path to the answers.json file.")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory to store the results.")

    args = parser.parse_args()

    retrieve_single_query(args.index_path, args.query, args.answers_file, args.results_dir)
