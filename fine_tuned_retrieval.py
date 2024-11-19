import sys
import os
import json
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from load_data import clean_text

# Ensure results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def run_fine_tuned_retrieval(answers_file, topics_file, output_bi_file, output_cross_file, bi_model_path, cross_model_path, encoded_answers_file='encoded_answers.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and clean answers
    answers = json.load(open(answers_file, 'r', encoding='utf-8'))
    for ans in answers:
        ans['Text'] = clean_text(ans['Text'])

    # Load and clean topics
    topics = json.load(open(topics_file, 'r', encoding='utf-8'))
    for topic in topics:
        if 'query' in topic:
            topic['query'] = clean_text(topic['query'])
        elif 'Title' in topic:
            topic['Title'] = clean_text(topic['Title'])

    # Load fine-tuned models
    bi_encoder = SentenceTransformer(bi_model_path).to(device)
    cross_encoder = CrossEncoder(cross_model_path)

    # Load or encode answers
    if os.path.exists(encoded_answers_file):
        print(f"Loading encoded answers from {encoded_answers_file}...")
        encoded_answers = torch.load(encoded_answers_file, map_location=device)
    else:
        print("Encoding answers and saving embeddings...")
        encoded_answers = bi_encoder.encode([ans['Text'] for ans in answers], batch_size=32, convert_to_tensor=True).to(device)
        torch.save(encoded_answers, encoded_answers_file)

    all_bi_results, all_cross_results = [], []

    for idx, topic in enumerate(topics):
        query = topic.get('query') or topic.get('Title')
        query_id = topic.get('query_id') or topic.get('Id')

        if not query or not query_id:
            continue

        # Encode query and compute cosine similarity
        query_embedding = bi_encoder.encode(query, convert_to_tensor=True).to(device)
        cosine_scores = util.cos_sim(query_embedding, encoded_answers)[0]

        # Top 100 bi-encoder results
        top_100_indices = torch.topk(cosine_scores, k=100).indices.tolist()
        top_100_texts = [answers[i]['Text'] for i in top_100_indices]
        top_100_ids = [answers[i]['Id'] for i in top_100_indices]
        top_100_scores = cosine_scores[top_100_indices].tolist()

        # Record Bi-encoder results
        for rank, (doc_id, score) in enumerate(zip(top_100_ids, top_100_scores)):
            all_bi_results.append(f"{query_id}\tQ0\t{doc_id}\t{rank + 1}\t{score}\tbi-encoder")

        # Cross-encoder re-ranking
        re_ranked_scores = cross_encoder.predict([[query, doc] for doc in top_100_texts])
        re_ranked_pairs = sorted(zip(top_100_ids, re_ranked_scores), key=lambda x: x[1], reverse=True)

        for rank, (doc_id, score) in enumerate(re_ranked_pairs):
            all_cross_results.append(f"{query_id}\tQ0\t{doc_id}\t{rank + 1}\t{score}\tcross-encoder")

    # Save results
    save_results(all_bi_results, output_bi_file)
    save_results(all_cross_results, output_cross_file)

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + "\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python fine_tuned_retrieval.py <answers_file> <test_topics_file> <topics_2_file> <output_file_prefix> <bi_model_path> <cross_model_path> <encoded_answers_file>")
        sys.exit(1)

    answers_file = sys.argv[1]
    test_topics_file = sys.argv[2]
    topics_2_file = sys.argv[3]
    output_file_prefix = sys.argv[4]
    bi_model_path = sys.argv[5]
    cross_model_path = sys.argv[6]
    encoded_answers_file = sys.argv[7]

    # Run fine-tuned retrieval for test set (topics 1)
    run_fine_tuned_retrieval(
        answers_file, 
        test_topics_file, 
        os.path.join(results_dir, f"result_bi_ft_1.tsv"), 
        os.path.join(results_dir, f"result_ce_ft_1.tsv"), 
        bi_model_path, 
        cross_model_path, 
        encoded_answers_file
    )

    # Run fine-tuned retrieval for topics_2 (general retrieval)
    run_fine_tuned_retrieval(
        answers_file, 
        topics_2_file, 
        os.path.join(results_dir, f"result_bi_ft_2.tsv"), 
        os.path.join(results_dir, f"result_ce_ft_2.tsv"), 
        bi_model_path, 
        cross_model_path, 
        encoded_answers_file
    )
