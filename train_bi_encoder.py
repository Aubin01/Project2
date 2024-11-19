from sentence_transformers import SentenceTransformer, InputExample
import torch
from torch.utils.data import DataLoader
import json
import sys
from tqdm import tqdm  # Importing tqdm for progress bar
from load_data import clean_text  # Import clean_text from load_data

def fine_tune_bi_encoder(train_file, val_file, model_save_path):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the train and val data
    train_data = json.load(open(train_file))
    val_data = json.load(open(val_file))

    # Prepare training data for bi-encoder fine-tuning with cleaned text
    train_examples = [
        InputExample(texts=[clean_text(data['query']), clean_text(data['doc'])], label=float(data['score']))
        for data in train_data
    ]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Initialize the model and move it to the correct device
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Training loop
    model.train()
    for epoch in range(10):  # Set the number of epochs here
        total_loss = 0.0
        print(f"Epoch {epoch + 1} in progress...")
        with tqdm(total=len(train_examples), desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
            for example in train_examples:
                # Encode the texts and move embeddings to the device
                text1, text2 = example.texts
                label = torch.tensor([example.label], device=device)  # Wrap label in a tensor with batch dimension

                # Use the forward method with `requires_grad=True` for embeddings
                embedding1 = model.encode([text1], convert_to_tensor=True).requires_grad_().to(device)
                embedding2 = model.encode([text2], convert_to_tensor=True).requires_grad_().to(device)

                # Calculate cosine similarity and compute loss
                scores = torch.nn.functional.cosine_similarity(embedding1, embedding2)
                loss = torch.nn.functional.mse_loss(scores, label)
                total_loss += loss.item()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)  # Update the progress bar

        print(f"Epoch {epoch + 1} - Loss: {total_loss / len(train_examples)}")

    # Save the fine-tuned model
    model.save(model_save_path)
    print(f"Fine-tuned Bi-encoder saved at {model_save_path}")

if __name__ == "__main__":
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    model_save_path = sys.argv[3]
    fine_tune_bi_encoder(train_file, val_file, model_save_path)
