import os
import pyterrier as pt
import pandas as pd
from load_data import load_answers, load_topics, build_index, clean_query

def save_results(results, output_file, model_name):
    """
    Save the results in TREC format with a .tsv extension and use the model name
    as the system name in the output file.
    """
    # Update the output file extension to .tsv
    output_file = output_file.replace('.txt', '.tsv')
    # Save the results with the specified system name
    pt.io.write_results(results, output_file, format='trec', run_name=model_name)

def main(answers_file, topics_1_file, topics_2_file, output_file_tfidf_1, output_file_tfidf_2, output_file_bm25_1, output_file_bm25_2):
    print("Loading answers...")
    answers = load_answers(answers_file)

    # Build the index using the function from load_data.py
    print("Building inverted index...")
    index_ref = build_index(answers)

    # Load and clean topics
    print("Loading and cleaning topics...")
    topics_1 = load_topics(topics_1_file)
    topics_2 = load_topics(topics_2_file)

    # Prepare DataFrames for the queries
    queries_1 = pd.DataFrame(
        [[topic['Id'], clean_query(topic['Title'])] for topic in topics_1],
        columns=["qid", "query"]
    )
    queries_2 = pd.DataFrame(
        [[topic['Id'], clean_query(topic['Title'])] for topic in topics_2],
        columns=["qid", "query"]
    )

    # Running retrieval using the tuned parameters with pt.terrier.Retriever
    print("Running retrieval...")
    bm25 = pt.terrier.Retriever(
        index_ref,
        wmodel="BM25",
        num_results=10,
        controls={"bm25.b": 0.75, "bm25.k_1": 1.5}
    )
    tfidf = pt.terrier.Retriever(index_ref, wmodel="TF_IDF", num_results=10)

    # Retrieve results for topics_1 and save
    results_tfidf_1 = tfidf.transform(queries_1)
    results_bm25_1 = bm25.transform(queries_1)
    save_results(results_tfidf_1, output_file_tfidf_1, "tf-idf")
    save_results(results_bm25_1, output_file_bm25_1, "bm25")

    # Retrieve results for topics_2 and save
    results_tfidf_2 = tfidf.transform(queries_2)
    results_bm25_2 = bm25.transform(queries_2)
    save_results(results_tfidf_2, output_file_tfidf_2, "tf-idf")
    save_results(results_bm25_2, output_file_bm25_2, "bm25")

    print("Retrieval and saving results completed.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process topic files and generate results for both TF-IDF and BM25 models.")
    parser.add_argument('--answers_file', type=str, required=True, help="Path to the answers.json file")
    parser.add_argument('--topics_1_file', type=str, required=True, help="Path to the topics_1.json file")
    parser.add_argument('--topics_2_file', type=str, required=True, help="Path to the topics_2.json file")
    parser.add_argument('--output_file_tfidf_1', type=str, required=True, help="Path to output the first result file for TF-IDF")
    parser.add_argument('--output_file_tfidf_2', type=str, required=True, help="Path to output the second result file for TF-IDF")
    parser.add_argument('--output_file_bm25_1', type=str, required=True, help="Path to output the first result file for BM25")
    parser.add_argument('--output_file_bm25_2', type=str, required=True, help="Path to output the second result file for BM25")

    args = parser.parse_args()
    main(args.answers_file, args.topics_1_file, args.topics_2_file, args.output_file_tfidf_1, args.output_file_tfidf_2, args.output_file_bm25_1, args.output_file_bm25_2)
