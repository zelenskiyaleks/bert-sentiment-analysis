![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-supported-blue)

# BERT Sentiment Analysis

This project is my end-to-end NLP pipeline for sentiment analysis of movie reviews.

The goal is simple: given a movie review, predict whether it is:

- **0** → Negative
- **1** → Positive

Example:

Input:

```text
This movie was absolutely amazing. Great acting and story.
```

Prediction:

```text
Positive
```

---

## Quick Start

Clone repository:

```bash
git clone https://github.com/zelenskiyaleks/bert-sentiment-analysis.git
cd bert-sentiment-analysis
```

Build Docker image:

```bash
docker build -t bert-sentiment-api .
```

Run API:

```bash
docker run -p 8000:8000 bert-sentiment-api
```

Open Swagger UI:

```text
http://localhost:8000/docs
```
---

## Why I built this project

I wanted to practice not only training transformer models, but also building a clean ML project structure:

- data loading
- exploratory data analysis
- text preprocessing
- tokenization
- model training
- evaluation

The goal was to make it look closer to a production ML pipeline, not just a notebook experiment.

---

## Dataset

I use the IMDb movie review dataset from Hugging Face.

Dataset contains:

- 50,000 labeled reviews
- binary sentiment labels
- balanced classes

For benchmark experiments I used:

- Train subset: 5,000 samples
- Validation subset: 1,000 samples

A reduced subset was used because of Apple Silicon MPS memory limitations during multi-model benchmarking.

---

## Project Structure

```bash
bert-sentiment-analysis/
│
├── configs/
├── notebooks/
├── reports/
│   ├── experiments.csv
│   ├── hyperparameter_tuning.csv
│   └── best_config.json
│
├── models/
├── src/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── inference/
│   └── utils/
│
├── Dockerfile
├── environment.yml
├── README.md
├── .dockerignore
└── .gitignore
```

---

## Data Exploration

During EDA I found:

- Classes are perfectly balanced
- Review lengths are highly right-skewed
- Some reviews contain HTML tags like `<br />`

This helped define the preprocessing pipeline.

---

## Text Preprocessing

Current preprocessing includes:

- removing HTML tags
- removing extra spaces
- tokenization with truncation

Example:

Before:

```text
Great movie! <br /><br /> Must watch.
```

After:

```text
Great movie! Must watch.
```

---

## Model

Baseline model:

bert-base-uncased

Training setup:

- Epochs: 1
- Batch size: 8
- Max sequence length: 512
- Learning rate: 2e-5

---

## Results

Baseline validation accuracy:

**93.04%**

Training hardware:

Apple Silicon (MPS)

---

## Models Compared

The benchmark currently includes:

- BERT (bert-base-uncased)
- RoBERTa (roberta-base)
- DistilBERT (distilbert-base-uncased)
  
## Benchmark Results

| Model | Accuracy | Train Time (min) |
| --- | --- | --- |
| BERT | 0.9060 | 12.28 |
| RoBERTa | 0.9140 | 10.51 |
| DistilBERT | 0.9090 | 6.32 |

### Observations

- RoBERTa achieved the best validation accuracy
- DistilBERT was the fastest model
- DistilBERT provides a strong speed vs quality tradeoff
- Benchmark results are automatically logged into `reports/experiments.csv`

## Training

Train model:

```bash
python -m src.training.train
```

Run multi-model benchmark:

```bash
python -m src.training.benchmark
```

---

## Run Inference

The project supports interactive command-line inference.
Run sentiment prediction from the command line:

```bash
python -m src.inference.predict
```

Example:

Input:

```text
This movie was absolutely amazing. Great acting and story.
```

Output:

```text
Prediction Result
Sentiment: Positive
Confidence: 0.9970
```

## API Service

The project also includes a FastAPI inference service for sentiment prediction.

Run API server:

```bash
uvicorn src.api.app:app --reload
```

API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Example request:

```json
{
  "text": "This movie was amazing!"
}
```

Example response:

```json
{
  "sentiment": "Positive",
  "confidence": 0.9981
}
```

---

## Docker Support

Build Docker image:

```bash
docker build -t bert-sentiment-api .
```

Run Docker container:

```bash
docker run -p 8000:8000 bert-sentiment-api
```

Open API documentation:

```text
http://localhost:8000/docs
```

---

## Hyperparameter Tuning

The project includes a hyperparameter tuning pipeline for experimenting with different training configurations.

Run tuning:

```bash
python -m src.training.tune
```

Current tuning parameters:

- learning rate
- batch size

Example tuning results:

| Learning Rate | Batch Size | Accuracy |
| --- | --- | --- |
| 2e-5 | 4 | 0.896 |
| 2e-5 | 8 | 0.892 |
| 3e-5 | 4 | 0.890 |
| 3e-5 | 8 | 0.888 |

Results are automatically saved to:

```text
reports/hyperparameter_tuning.csv
```

Best configuration is saved to:

```text
reports/best_config.json
```

The tuning pipeline also tracks:

- validation accuracy
- training time
- experiment configurations

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- FastAPI
- Docker
- scikit-learn
- Jupyter Notebook