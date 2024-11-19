#### Bi-Encoder and Cross-Encoder Retrieval Models

## Project Description
This project implements two advanced Information Retrieval (IR) models: Bi-encoder and Cross-encoder. The primary objective is to process travel-related queries and retrieve relevant documents by employing both basic and fine-tuned variants of these retrieval models. The Bi-encoder searches the entire document collection for initial retrieval, while the Cross-encoder re-ranks the top-100 results from the Bi-encoder for improved precision.

The project evaluates retrieval performance using metrics such as precision@1, precision@5, recall@5, MAP, nDCG@5, and MRR, generating ski-jump plots for each model based on precision@5 across queries.

## Table of Contents
1. Files
2. Installation
3. How to Run the Code
4. Outputs
5. Evaluation
6. Performance Notes

## Files
1. load_data.py: Loads and preprocesses documents and queries by applying text cleaning. Supports embedding generation with a Bi-encoder, re-ranking with a Cross-encoder, and provides functions for model fine-tuning to improve retrieval.

2. split_data.py: Splits the dataset into training, validation, and test sets, ensuring unique query-document pairs in each set for effective fine-tuning and evaluation.

3. basic_retrieval.py: Performs initial retrieval using the Bi-encoder model and re-ranks top results with the Cross-encoder, without fine-tuning.

4. train_bi_encoder.py: Fine-tunes and saves the Bi-encoder model on labeled training data to improve its ability to generate embeddings for retrieval.

5. fine_tune_cross_encoder.py: Fine-tunes and saves the Cross-encoder model, optimizing it to re-rank top-100 results from the Bi-encoder based on relevance.

6. fine_tuned_retrieval.py: Runs retrieval using fine-tuned Bi-encoder and Cross-encoder models to generate more accurate, relevance-ranked results.

7. evaluate.py: Calculates and outputs performance metrics (precision, recall, MAP, nDCG, MRR) for retrieval results of both fine-tuned and non-fine-tuned models, generating relevant visualizations.

## Installation
Prerequisites:

Python 3.x
 Required Python packages: pandas, argparse, matplotlib, ranx, sentence_transformers, toch 

## How to Run the Code
Step 1: to Split Data for Training and Testing, run the following command: 
```bash
ex: python src/split_data.py data/topics_1.json data/qrel_1.tsv data/Answers.json
```
step 2: basic retrieval, run the following command: 
```bash
 ex: python src/basic_retrieval.py data/Answers.json test_data.json topics2.json
```
step3: Fine-Tuning the Models:

 Run the Bi-encoder fine-tuning script:
 ```bash
 ex: python src/train_bi_encoder.py data/train_data.json data/val_data.json models/bi_encoder_finetuned
```
 Run the Cross-encoder fine-tuning and re-ranking script: 
  ```bash
 ex: python src/train_cross_encoder.py data/train_data.json data/val_data.json models/cross_encoder_finetuned
```

step 4: Generating Retrieval Results:
After fine-tuning, run the fine_tuned_retrieval.py script:
```bash
ex: python src/fine_tuned_retrieval.py data/Answers.json data/test_data.json data/topics_2.json results models/bi_encoder_finetuned models/cross_encoder_finetuned encoded_answers.pt
```

## Outputs
Upon running the retrieval scripts, eight output files will be generated:

result_bi_1.tsv: Bi-encoder without fine-tuning on the test set.
result_bi_2.tsv: Bi-encoder without fine-tuning on topic_2 file.
result_bi_ft_1.tsv: Fine-tuned Bi-encoder on the test set.
result_bi_ft_2.tsv: Fine-tuned Bi-encoder on topic_2 file.
result_ce_1.tsv: Cross-encoder without fine-tuning on the test set.
result_ce_2.tsv: Cross-encoder without fine-tuning on topic_2 file.
result_ce_ft_1.tsv: Fine-tuned Cross-encoder on the test set.
result_ce_ft_2.tsv: Fine-tuned Cross-encoder on topic_2 file.

## Evaluation
Evaluate Retrieval Performance:

Run the evaluation script to calculate precision, recall, MAP, nDCG, and other metrics:

For fine_tuned model:
```bash
ex: python src/evaluate.py --qrel_file data/test_qrel.tsv --output_dir evaluation_results --mode fine_tune
```
For basic model:
```bash
ex: python src/evaluate.py --qrel_file data/test_qrel.tsv --output_dir evaluation_results --mode basic
```

## Performance Notes
Running the retrieval code can take a significant amount of time, particularly when encoding the answers, unless executed on high-performance servers, such as the CS department's servers. The computational load stems from the simultaneous processing of Bi-encoder and Cross-encoder results and the calculation of various evaluation metrics.

Interestingly, fine-tuning did not yield improved performance in this instance, and the results were below expectations. This may be due to potential issues during model training, and further investigation is planned to identify and address any underlying causes.
