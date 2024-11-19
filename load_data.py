import json
import re
from bs4 import BeautifulSoup
import random
from sentence_transformers import InputExample
from sentence_transformers import SentenceTransformer, CrossEncoder

### Text Cleaning Function ###
def clean_text(text):
    """Clean text by removing HTML tags, URLs, special characters, and converting to lowercase."""
    if text:
        # Remove HTML tags
        if "<" in text and ">" in text:
            text = BeautifulSoup(text, "lxml").get_text()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
    else:
        text = ""
    
    return text

### Data Loading Functions ###
def load_topics(file_path):
    """Load and preprocess topics using 'query_id' as ID and 'query' as text."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    topics = {}
    for item in data:
        topic_id = item['query_id']  # Using 'query_id' based on the structure you provided
        query_text = clean_text(item['query'])
        
        topics[topic_id] = query_text
    return topics

def load_answers(file_path):
    """Load answers using 'doc_id' and clean text in 'doc' field."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    answers = {item['doc_id']: clean_text(item['doc']) for item in data if 'doc' in item}
    return answers

def parse_qrel(qrel_file):
    """Parse qrel file to map query IDs to relevant document IDs and relevance scores."""
    qrel_data = {}
    with open(qrel_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrel_data:
                qrel_data[query_id] = {}
            qrel_data[query_id][doc_id] = int(relevance)
    return qrel_data

### Prepare Training and Validation Samples ###
def prepare_train_val_samples(topics, answers, qrels, split_ratio=0.9):
    """Prepare training and validation samples from qrels, topics, and answers."""
    train_samples, val_samples = [], []
    query_ids = list(qrels.keys())
    random.shuffle(query_ids)
    
    train_size = int(len(query_ids) * split_ratio)
    train_ids = query_ids[:train_size]
    val_ids = query_ids[train_size:]
    
    # Generate InputExamples for each set
    for qid in train_ids:
        topic_text = topics.get(qid, "")
        for doc_id, relevance in qrels[qid].items():
            label = 1.0 if relevance > 0 else 0.0
            if doc_id in answers:
                train_samples.append(InputExample(texts=[topic_text, answers[doc_id]], label=label))
    
    for qid in val_ids:
        topic_text = topics.get(qid, "")
        for doc_id, relevance in qrels[qid].items():
            label = 1.0 if relevance > 0 else 0.0
            if doc_id in answers:
                val_samples.append(InputExample(texts=[topic_text, answers[doc_id]], label=label))

    return train_samples, val_samples

### Bi-encoder and Cross-encoder Wrapper Classes ###
class BiEncoderModel:
    def __init__(self, model_name='distilbert-base-nli-stsb-mean-tokens'):
        self.model = SentenceTransformer(model_name)
        print(f"Initialized Bi-Encoder Model: {model_name}")

    def encode(self, texts, batch_size=32):
        """Encode texts using bi-encoder in batches."""
        return self.model.encode(texts, batch_size=batch_size, convert_to_tensor=True)

class CrossEncoderModel:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name)
        print(f"Initialized Cross-Encoder Model: {model_name}")

    def re_rank(self, query, documents):
        """Re-rank documents given a query using the Cross-encoder."""
        return self.model.predict([[query, doc] for doc in documents])
