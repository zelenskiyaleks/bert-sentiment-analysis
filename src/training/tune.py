import gc
import time
import torch
import pandas as pd
import json
from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from src.training.metrics import compute_metrics
from src.data.data_loader import create_train_validation_split
from src.data.preprocessing import clean_text
from src.features.tokenization import (
    load_tokenizer,
    tokenize_batch
)
from src.utils.config import load_config


LEARNING_RATES = [
    2e-5,
    3e-5
]

BATCH_SIZES = [
    4,
    8
]


def train_and_evaluate(
    model_name,
    tokenizer,
    tokenized_train,
    tokenized_valid,
    data_collator,
    learning_rate,
    batch_size,
    num_train_epochs,
    max_length
):

    print("=" * 50)

    print(
        f"Training with lr={learning_rate}, "
        f"batch_size={batch_size}"
    )

    print("Loading fresh model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    args = TrainingArguments(
        output_dir=(
            f"./models/"
            f"lr_{learning_rate}_bs_{batch_size}"
        ),

        num_train_epochs=num_train_epochs,

        per_device_train_batch_size=batch_size,

        per_device_eval_batch_size=batch_size,

        eval_strategy="epoch",

        save_strategy="no",

        learning_rate=learning_rate,

        report_to="none"
    )

    trainer = Trainer(
        model=model,

        args=args,

        train_dataset=tokenized_train,

        eval_dataset=tokenized_valid,

        processing_class=tokenizer,

        data_collator=data_collator,

        compute_metrics=compute_metrics
    )

    print("Starting training...")

    start_time = time.time()

    trainer.train()

    end_time = time.time()

    train_time_minutes = (
        end_time - start_time
    ) / 60

    print("Running evaluation...")

    metrics = trainer.evaluate()

    accuracy = metrics["eval_accuracy"]

    print(f"Accuracy: {accuracy:.4f}")

    print(
        f"Train time: "
        f"{train_time_minutes:.2f} minutes"
    )

    print("Cleaning GPU memory...")

    del trainer
    del model

    gc.collect()

    torch.mps.empty_cache()

    return {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_length": max_length,
        "accuracy": round(accuracy, 4),
        "train_time_minutes": round(
            train_time_minutes,
            2
        )
    }


def main():

    print("Loading configuration...")

    config = load_config(
        "configs/bert_base.yaml"
    )

    model_name = config["model_name"]

    max_length = config["tokenization"]["max_length"]

    num_train_epochs = (
        config["training"]["num_train_epochs"]
    )

    print("Loading dataset...")

    train_part, valid_part = (
        create_train_validation_split()
    )

    print("Reducing dataset size for tuning...")

    train_part = train_part.sample(
        2000,
        random_state=42
    )
    
    valid_part = valid_part.sample(
        500,
        random_state=42
    )

    print("Cleaning text...")

    train_part["text"] = train_part["text"].apply(
        clean_text
    )

    valid_part["text"] = valid_part["text"].apply(
        clean_text
    )

    print("Loading tokenizer...")

    tokenizer = load_tokenizer(
        model_name
    )

    print(
        "Converting to Hugging Face dataset..."
    )

    train_ds = Dataset.from_pandas(
        train_part
    )

    valid_ds = Dataset.from_pandas(
        valid_part
    )

    print("Tokenizing dataset...")

    tokenized_train = train_ds.map(
        lambda x: tokenize_batch(
            x,
            tokenizer,
            max_length
        ),
        batched=True
    )

    tokenized_valid = valid_ds.map(
        lambda x: tokenize_batch(
            x,
            tokenizer,
            max_length
        ),
        batched=True
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    results = []

    for learning_rate in LEARNING_RATES:

        for batch_size in BATCH_SIZES:

            result = train_and_evaluate(
                model_name=model_name,

                tokenizer=tokenizer,

                tokenized_train=tokenized_train,

                tokenized_valid=tokenized_valid,

                data_collator=data_collator,

                learning_rate=learning_rate,

                batch_size=batch_size,

                num_train_epochs=num_train_epochs,

                max_length=max_length
            )

            results.append(result)

    print("Saving tuning results...")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "reports/hyperparameter_tuning.csv",
        index=False
    )

    best_result = results_df.sort_values(
        by="accuracy",
        ascending=False
    ).iloc[0]
    
    print("\nBest configuration:")
    
    print(best_result)

    best_config = {
        "model_name": best_result["model_name"],
        "learning_rate": float(
            best_result["learning_rate"]
        ),
        "batch_size": int(
            best_result["batch_size"]
        ),
        "max_length": int(
            best_result["max_length"]
        ),
        "accuracy": float(
            best_result["accuracy"]
        ),
        "train_time_minutes": float(
            best_result["train_time_minutes"]
        )
    }
    
    with open(
        "reports/best_config.json",
        "w"
    ) as f:
    
        json.dump(
            best_config,
            f,
            indent=4
        )
    
    print(
        "\nSaved best config to "
        "reports/best_config.json"
    )


if __name__ == "__main__":
    main()