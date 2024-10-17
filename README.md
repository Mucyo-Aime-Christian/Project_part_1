# 🌟 Ubuntu Information Retrieval System 🌟
`A project that implements an information retrieval system to query technical questions related to the Ubuntu operating system. Using PyTerrier, we explore and compare the traditional retrieval models BM25 and TF-IDF, evaluating their ability to retrieve the most relevant documents.`

# 🚀 Features
📄 Document Indexing: `Efficiently builds an inverted index of Ubuntu-related technical documents.`

🔍 Query Processing: `Supports both user-provided single queries and batch processing using predefined topics.`

⚖️ Model Comparison: `Implements and compares BM25 and TF-IDF retrieval models.`

📊 Evaluation: `Measures model performance using standard metrics like Precision, Recall, Mean Average Precision (MAP), and nDCG, evaluated against ground truth Qrels.`

# 🛠️ Technologies Used
🐍 Python: `The backbone of the project.`
🧰 PyTerrier: `For retrieval and indexing.`
📊 Ranx: `For model evaluation.`

# 📋 How to Run the Project
Clone the Repository: `git clone https://github.com/yourusername/ubuntu-information-retrieval-system.git`

Install Dependencies: `Install the necessary libraries using pip`
- pip install pyterrier tqdm ranx

# Run the Retrieval System: `Build the index, perform retrieval, and run evaluation:`

- You must have a results file already set under your project folder. If not, run `mkdir -p results`
# Runs the project to load files and retrieve data
python3 check.py \
 --answers_file files/Answers.json \
 --topics_1_file files/topics_1.json \
 --topics_2_file files/topics_2.json \
 --output_file_tfidf_1 results/output_tfidf_1.tsv \
 --output_file_tfidf_2 results/output_tfidf_2.tsv \
 --output_file_bm25_1 results/output_bm25_1.tsv \
 --output_file_bm25_2 results/output_bm25_2.tsv

 # evaluates the data retrieved compared to qrels
 python3 src/evaluate.py \
    --qrel_file "files/qrel_1.trec" \
    --run_file_tfidf "results/output_tfidf_1.tsv" \
    --run_file_bm25 "results/output_bm25_1.tsv" \
    --output_file_tfidf "results/tfidf_evaluation.csv" \
    --output_file_bm25 "results/bm25_evaluation.csv"
# to run a single query(example)

python3 src/query.py \
    --index_path "/Users/christianmucyo/Documents/Fall 2024 Semester/COS 470/Project/Part1/part1/src/index" \
    --query "How do I install packages on Ubuntu?" \
    --answers_file "files/Answers.json" \
    --results_dir "results/"



```# 📂 Project Structure
PART1/
├── files/
│   ├── Answers.json            # Dataset containing Ubuntu-related technical documents
│   ├── qrel_1.trec             # Ground truth relevance judgments (Qrels)
│   ├── topics_1.json           # Predefined queries (topics)
│   └── topics_2.json           # Additional predefined queries
│
│
├── results/
│   ├── bm25_evaluation.csv      # Evaluation metrics for BM25 model
│   ├── bm25_results_1.tsv       # Retrieval results from BM25 for topics_1
│   ├── bm25_results_2.tsv       # Retrieval results from BM25 for topics_2
│   ├── bm25_user_results.trec   # BM25 results for user-provided query
│   ├── tfidf_evaluation.csv     # Evaluation metrics for TF-IDF model
│   ├── tfidf_results_1.tsv      # Retrieval results from TF-IDF for topics_1
│   ├── tfidf_results_2.tsv      # Retrieval results from TF-IDF for topics_2
│   └── tfidf_user_results.trec  # TF-IDF results for user-provided query
│
├── src/
|   |__index/                    # Folder for storing the inverted index
│   ├── check.py                 # Script for testing or preprocessing
│   ├── evaluate.py              # Script to evaluate retrieval results
│   ├── load_data.py             # Utility functions to load and process data
│   └── query.py                 # Main script for querying and retrieval
│
├── README.md                    # Project documentation```




