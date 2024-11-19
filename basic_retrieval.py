import os
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import json
import torch
from load_data import clean_text

# Ensure results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)  # Create results folder if it doesn't exist

def run_basic_retrieval(answers_file, topics_1_file, topics_2_file, encoded_answers_file='encoded_answers.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and clean answers
    print("Loading answers...")
    answers = json.load(open(answers_file, 'r', encoding='utf-8'))
    for ans in answers:
        ans['Text'] = clean_text(ans['Text'])
    
    # Initialize models
    bi_encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens').to(device)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Load or encode answers
    if os.path.exists(encoded_answers_file):
        print("Loading encoded answers...")
        encoded_answers = torch.load(encoded_answers_file, map_location=device)
    else:
        print("Encoding answers...")
        encoded_answers = bi_encoder.encode([ans['Text'] for ans in answers], convert_to_tensor=True).to(device)
        torch.save(encoded_answers, encoded_answers_file)

    # Process topics 1 for retrieval on the test set
    print("Processing topics 1...")
    topics_1 = json.load(open(topics_1_file, 'r', encoding='utf-8'))
    all_bi_results_1, all_ce_results_1 = retrieve_topics(topics_1, answers, bi_encoder, cross_encoder, encoded_answers, device, topic_type=1)

    # Save results for topics 1
    save_results(all_bi_results_1, os.path.join(results_dir, "result_bi_1.tsv"))
    save_results(all_ce_results_1, os.path.join(results_dir, "result_ce_1.tsv"))

    # Process topics 2 for general retrieval
    print("Processing topics 2...")
    topics_2 = json.load(open(topics_2_file, 'r', encoding='utf-8'))
    all_bi_results_2, all_ce_results_2 = retrieve_topics(topics_2, answers, bi_encoder, cross_encoder, encoded_answers, device, topic_type=2)

    # Save results for topics 2
    save_results(all_bi_results_2, os.path.join(results_dir, "result_bi_2.tsv"))
    save_results(all_ce_results_2, os.path.join(results_dir, "result_ce_2.tsv"))

def retrieve_topics(topics, answers, bi_encoder, cross_encoder, encoded_answers, device, topic_type=1):
    all_bi_results, all_ce_results = [], []

    for idx, topic in enumerate(topics):
        if topic_type == 1:
            query = topic.get('query')
            query_id = topic.get('query_id')
        else:  # For topics 2
            query = topic.get('Title')
            query_id = topic.get('Id')

        if not query or not query_id:
            print(f"Skipping topic at index {idx} due to missing 'query' or 'query_id'")
            continue

        # Encode query and calculate similarity for bi-encoder retrieval
        query_embedding = bi_encoder.encode(query, convert_to_tensor=True).to(device)
        cosine_scores = util.cos_sim(query_embedding, encoded_answers)[0]
        
        # Top 100 bi-encoder results
        top_100_indices = torch.topk(cosine_scores, k=100).indices.tolist()
        top_100_texts = [answers[i]['Text'] for i in top_100_indices]
        top_100_ids = [answers[i]['Id'] for i in top_100_indices]
        
        # Record bi-encoder results
        for rank, doc_id in enumerate(top_100_ids):
            all_bi_results.append(f"{query_id}\tQ0\t{doc_id}\t{rank + 1}\t{cosine_scores[top_100_indices[rank]].item()}\tbi-encoder")
        
        # Re-rank with Cross-encoder
        re_ranked_scores = cross_encoder.predict([[query, doc] for doc in top_100_texts])
        re_ranked_pairs = sorted(zip(top_100_ids, re_ranked_scores), key=lambda x: x[1], reverse=True)
        
        for rank, (doc_id, score) in enumerate(re_ranked_pairs):
            all_ce_results.append(f"{query_id}\tQ0\t{doc_id}\t{rank + 1}\t{score}\tcross-encoder")

    return all_bi_results, all_ce_results

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + "\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python basic_retrieval.py <answers_file> <test_topics_file> <topics_2_file>")
        sys.exit(1)

    answers_file = sys.argv[1]
    test_topics_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    run_basic_retrieval(answers_file, test_topics_file, topics_2_file)
