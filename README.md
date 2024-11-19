# Description
This project applies various information retrieval models, including bi-encoders and cross-encoders, to retrieve relevant answers for user queries. The models are evaluated in fine-tuned and non-fine-tuned configurations across two datasets (topics_1 and topics_2). Key components include dataset preparation, model training, evaluation, and analysis of results. Insights from the project aim to improve precision and efficiency in neural retrieval systems.

# python Files description
Here’s an overview of the Python files in the project:

# Core Files
`split_data.py`

Splits topics_1.json into training, validation, and test datasets.
# Outputs:
- train_topics.json
- validation_topics.json
- test_set.json
# basic_retrieval.py

Implements non-fine-tuned bi-encoder and cross-encoder models.
Encodes queries and documents using FAISS for similarity search.
Outputs results in TSV format`
# fine_tuned_retrieval.py

# Implements fine-tuned versions of bi-encoder and cross-encoder models.
Trains models on qrel_1.tsv or qrel_2.tsv and evaluates their retrieval performance.
Outputs fine-tuned results.
# train_bi_encoder.py

Script for training and fine-tuning the bi-encoder model using qrel_1.tsv or qrel_2.tsv.
# train_cross_encoder.py

Script for training and fine-tuning the cross-encoder model on datasets.
# evaluate.py

Evaluates model outputs using metrics such as:
Precision@k, nDCG@k,MRR,Mean Average Precision (mAP)
Outputs evaluation scores for analysis.

# query.py

Parses and preprocesses queries from JSON files for consistency during retrieval.
# load_data.py

Contains utility functions for loading, parsing, and preprocessing datasets, QRELs, and model outputs.
Ensures consistent data handling across scripts.

## Prerequisites

To run this project, ensure you have the following:

1. **Python 3.x** installed on your system.
2. Required Python libraries, which can be installed via:
   ```bash
   pip install -r "requirements"
   ```
`requirements`: sentence-transformers, faiss-cpu, torch numpy

## Project structure
- models/
    - bi_encoder_finetuned
    - cross_encoder_finetuned
- data/
    - Answers.json
    - qrel_1.tsv
    - test_data.json
    - test_qrel.tsv
    - topics_1.json
    - topics_2.json
    - train_data.json
    - val_data.json
- src/
    - basic_retrieval.py
    - fine_tuned_retrieval.py
    - train_bi_encoder.py
    - train_cross_encoder.py
    - evaluate.py
    - query.py
    - load_data.py
    - split_data.py
    
- results_file/
    - result_bi_1.tsv
    - result_bi_2.tsv
    - result_bi_ft_1.tsv
    - result_bi_ft_2.tsv
    - result_ce_1.tsv
    - result_ce_2.tsv
    - result_ce_ft_1.tsv
    - result_ce_ft_2.tsv
evaluation_metrics.csv
- README.md

# `How to run the project`

Step 1: to Split Data for Training and Testing, run the following command: 
```bash
ex: python src/split_data.py data/topics_1.json data/qrel_1.tsv data/Answers.json
```
step 2: basic retrieval, run the following command: 
```bash
 python src/basic_retrieval.py data/Answers.json test_data.json topics2.json
```
step3: Fine-Tuning the Models:

 Run the Bi-encoder fine-tuning script:
 ```bash
 python src/train_bi_encoder.py data/train_data.json data/val_data.json models/bi_encoder_finetuned
```
 Run the Cross-encoder fine-tuning and re-ranking script: 
  ```bash
 python src/train_cross_encoder.py data/train_data.json data/val_data.json models/cross_encoder_finetuned
```

step 4: Generating Retrieval Results:
After fine-tuning, run the fine_tuned_retrieval.py script:
```bash
ex: python src/fine_tuned_retrieval.py data/Answers.json data/test_data.json data/topics_2.json results models/bi_encoder_finetuned models/cross_encoder_finetuned encoded_answers.pt
```

step 5: Evaluate Retrieval Performance:
Run the evaluation script to calculate precision, recall, MAP, nDCG, and other metrics:
For fine_tuned model:
```bash
ex: python src/evaluate.py --qrel_file data/test_qrel.tsv --output_dir evaluation_results --mode fine_tune
```
For basic model:
```bash
ex: python src/evaluate.py --qrel_file data/test_qrel.tsv --output_dir evaluation_results --mode basic
```
