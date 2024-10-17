import pyterrier as pt
import pandas as pd
import re
import json

# Initialize PyTerrier
if not pt.started():
    pt.init()

def load_answers(file_path):
    """Load answers from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        answers = json.load(file)
    return answers

def load_topics(file_path):
    """Load topics from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        topics = json.load(file)
    return topics

def filter_empty_documents(answers):
    """Filter out documents that are empty."""
    return [
        {'docno': str(answer['Id']), 'text': answer['Text']}
        for answer in answers if answer['Text'].strip() != ''
    ]

# defines stop words in a set
STOP_WORDS = set(["the", "is", "in", "and", "to", "a", "of", "that", "it"])
# Function to remove stop words from the text
def remove_stop_words(text):
    tokens = text.split()
    return " ".join([token for token in tokens if token.lower() not in STOP_WORDS])

def clean_text(text):
    """Remove HTML tags, URLs, and stop words from the text."""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = remove_stop_words(text)  # Remove stop words
    return text.strip()

def build_index(answers, index_path="./index"):
    """Build an index using PyTerrier's DFIndexer."""
    # Filter out empty documents
    filtered_answers = filter_empty_documents(answers)

    # Clean the text in the documents
    for answer in filtered_answers:
        answer['text'] = clean_text(answer['text'])

    # Convert to DataFrame
    docs_df = pd.DataFrame(filtered_answers)

    # Create the index using DFIndexer
    indexer = pt.DFIndexer(
        index_path,
        overwrite=True,
        stopwords="english",
        stemmer="porter",
        tokeniser="UTFTokeniser",
        meta={'docno': 'text', 'text': 'text'},
        meta_tags=['docno']
    )
    index_ref = indexer.index(docs_df["text"], docs_df["docno"])
    
    return index_ref

def clean_query(query):
    """Clean the query text."""
    query = re.sub(r'http\S+|www\S+|<[^>]+>', '', query)  # Remove URLs and HTML tags
    query = re.sub(r'[^\w\s]', '', query)  # Remove non-alphanumeric characters
    query = query.strip()
    return query.lower()
