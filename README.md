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

For baseline experiments I used:

- Train: 20,000 samples
- Validation: 5,000 samples

---

## Project Structure

```bash
bert-sentiment-analysis/
│
├── configs/
├── notebooks/
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

## How to run

Train model:

```bash
python -m src.training.train
```

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Jupyter Notebook