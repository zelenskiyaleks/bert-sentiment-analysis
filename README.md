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

Train subset: 5,000 samples
Validation subset: 1,000 samples

A reduced subset was used because of Apple Silicon MPS memory limitations during multi-model benchmarking.

---

## Project Structure

```bash
bert-sentiment-analysis/
│
├── configs/
├── notebooks/
├── reports/
├── models/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── inference/
│   └── utils/
│
├── environment.yml
├── README.md
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

The benchmark currently includes (for comparison I reduced subset because of Apple Silicon MPS memory limitations during multi-model benchmarking):

- BERT (bert-base-uncased)
- RoBERTa (roberta-base)
- DistilBERT (distilbert-base-uncased)
  
## Benchmark Results

Model	     Accuracy	Train Time (min)
BERT	     0.9060	    12.28
RoBERTa	     0.9140	    10.51
DistilBERT	 0.9090	    6.32

## Observations

RoBERTa achieved the best validation accuracy
DistilBERT was the fastest model
DistilBERT provides a strong speed vs quality tradeoff
Benchmark results are automatically logged into reports/experiments.csv

## How to run

Train model:

```bash
python -m src.training.train
```

Run multi-model benchmark:

```bash
python -m src.training.benchmark
```

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Jupyter Notebook