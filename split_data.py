import json
import random
import sys
import os
from bs4 import BeautifulSoup

def parse_qrel(qrel_file):
    """Parse the qrel file to get unique query-answer relevance judgments."""
    qrel_data = {}
    with open(qrel_file, 'r', encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if query_id not in qrel_data:
                qrel_data[query_id] = set()  # Using a set to store unique pairs
            
            # Add only unique document IDs for each query ID
            qrel_data[query_id].add((doc_id, int(relevance)))

    # Convert sets back to list of dictionaries for easier downstream processing
    for query_id in qrel_data:
        qrel_data[query_id] = [{"doc_id": doc_id, "relevance": relevance} for doc_id, relevance in qrel_data[query_id]]

    return qrel_data

def save_results(results, output_file):
    """Save results to file, ensuring consistency."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + "\n")
    print(f"Results saved to {output_file}")

def clean_body_text(html_content):
    """Remove HTML tags from the Body field using BeautifulSoup."""
    return BeautifulSoup(html_content, "html.parser").get_text()

def save_qrel(test_data, qrel_data, qrel_output_file):
    """Save unique test data queries and their relevant answers in TREC qrel format."""
    results = []
    seen_pairs = set()  # Track unique (query_id, doc_id) pairs

    for entry in test_data:
        query_id = entry['query_id']
        if query_id in qrel_data:
            for qrel_entry in qrel_data[query_id]:
                pair = (query_id, qrel_entry['doc_id'])
                if pair not in seen_pairs:  # Only add if the pair is new
                    results.append(f"{query_id}\t0\t{qrel_entry['doc_id']}\t{qrel_entry['relevance']}")
                    seen_pairs.add(pair)

    save_results(results, qrel_output_file)

def split_data(topics_file, qrel_file, answers_file, train_output="train_data.json", val_output="val_data.json", test_output="test_data.json", qrel_output="test_qrel.tsv", train_ratio=0.8, val_ratio=0.1):
    """Splits the topics data into training, validation, and test sets, creating query-document pairs with labels."""

    # Load topics and answers with UTF-8 encoding
    with open(topics_file, 'r', encoding='utf-8') as f:
        topics = json.load(f)
    
    with open(answers_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)

    # Parse qrel to map query IDs to relevant document IDs and relevance scores
    qrel_data = parse_qrel(qrel_file)

    # Shuffle the topics to ensure randomness
    random.shuffle(topics)

    # Get answer texts as a dictionary for fast lookup by answer ID
    answer_dict = {ans['Id']: ans['Text'] for ans in answers}

    query_document_groups = []

    # Group all answers for each query to avoid splitting answers across sets
    for idx, topic in enumerate(topics):
        query_id = topic.get('Id', f'topic_{idx}')
        
        # Combine Title and Body for the query, cleaning the Body text
        title = f"[TITLE] {topic['Title']}"
        body = f"[BODY] {clean_body_text(topic.get('Body', ''))}"  # Optional body content if available
        query_text = f"{title} {body}".strip()

        if query_id in qrel_data:
            query_group = []
            for qrel in qrel_data[query_id]:
                doc_id = qrel['doc_id']
                relevance = qrel['relevance']
                if doc_id in answer_dict:
                    query_group.append({
                        "query": query_text,
                        "query_id": query_id,
                        "doc": answer_dict[doc_id],
                        "doc_id": doc_id,
                        "score": relevance
                    })
            # Only add if the query has answers
            if query_group:
                query_document_groups.append(query_group)

    # Determine the sizes of each split
    train_size = int(len(query_document_groups) * train_ratio)
    val_size = int(len(query_document_groups) * val_ratio)

    # Split into train, validation, and test sets
    train_groups = query_document_groups[:train_size]
    val_groups = query_document_groups[train_size:train_size + val_size]
    test_groups = query_document_groups[train_size + val_size:]

    # Flatten the grouped queries into single lists of train, validation, and test data
    train_data = [item for group in train_groups for item in group]
    val_data = [item for group in val_groups for item in group]
    test_data = [item for group in test_groups for item in group]

    # Save train, validation, and test data to respective output files
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    
    with open(val_output, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)

    with open(test_output, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)

    # Save qrel file for the test data, only including relevant queries and answers from the test set
    save_qrel(test_data, qrel_data, qrel_output)

    print(f"Training data saved to {train_output}")
    print(f"Validation data saved to {val_output}")
    print(f"Test data saved to {test_output}")

### Command-line Interface ###
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python split_data.py <topics_file> <qrel_file> <answers_file>")
        sys.exit(1)

    topics_file = sys.argv[1]
    qrel_file = sys.argv[2]  
    answers_file = sys.argv[3]

    # Call the split function with the provided arguments and standard output file names
    split_data(
    topics_file, 
    qrel_file, 
    answers_file, 
    train_output="data/train_data.json", 
    val_output="data/val_data.json", 
    test_output="data/test_data.json", 
    qrel_output="data/test_qrel.tsv"
)
