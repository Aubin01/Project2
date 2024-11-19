from sentence_transformers import CrossEncoder, InputExample, losses
from torch.utils.data import DataLoader
import json
import math
import sys
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator


def fine_tune_cross_encoder(train_file, val_file, model_save_path, num_epochs=10):
    """
    Fine-tunes a CrossEncoder model using training and validation data.
    """
    # Load training and validation data
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    print(f"Loaded {len(train_data)} training samples and {len(val_data)} validation samples.")

    # Prepare training samples
    train_samples = [
        InputExample(texts=[item["query"], item["doc"]], label=float(item["score"]))
        for item in train_data
    ]

    # Prepare evaluator data for validation
    evaluator_data = {}
    for item in val_data:
        qid = item["query_id"]
        if qid not in evaluator_data:
            evaluator_data[qid] = {
                "query": item["query"],
                "positive": [],
                "negative": []
            }

        # Add to positive or negative based on the score
        if item["score"] >= 1:
            evaluator_data[qid]["positive"].append(item["doc"])
        elif item["score"] == 0:
            evaluator_data[qid]["negative"].append(item["doc"])

    # Ensure at least one positive and one negative sample for each query
    evaluator_data = {
        qid: data for qid, data in evaluator_data.items() if data["positive"] and data["negative"]
    }

    if not evaluator_data:
        raise ValueError("No valid evaluator data. Ensure val_data contains queries with both positives and negatives.")

    print(f"Evaluator initialized with {len(evaluator_data)} queries.")

    # Initialize CrossEncoder model
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("Initialized CrossEncoder model for fine-tuning.")

    # Create DataLoader for training data
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=16)

    # Create CERerankingEvaluator
    evaluator = CERerankingEvaluator(evaluator_data, name="train-eval")

    # Define training parameters
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% warm-up steps
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Start training with evaluation after each epoch
    print(f"Training CrossEncoder model for {num_epochs} epochs...")
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        save_best_model=True,
        show_progress_bar=True
    )
    print(f"Fine-tuned CrossEncoder model saved at {model_save_path}")


# Command-line interface
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_cross_encoder.py <train_file> <val_file> <model_save_path>")
        sys.exit(1)

    train_file = sys.argv[1]
    val_file = sys.argv[2]
    model_save_path = sys.argv[3]

    fine_tune_cross_encoder(train_file, val_file, model_save_path)
