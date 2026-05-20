import gc
import time

import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from src.training.metrics import compute_metrics

from src.data.data_loader import (
    create_train_validation_split
)

from src.data.preprocessing import clean_text

from src.features.tokenization import (
    load_tokenizer,
    tokenize_batch
)

from src.utils.config import load_config


def main():

    models_config = [
        "bert_base.yaml",
        "roberta_base.yaml",
        "distilbert_base.yaml"
    ]

    print("Loading dataset...")

    train_part, valid_part = create_train_validation_split()

    print("Creating benchmark subset...")

    train_part = train_part.sample(
        5000,
        random_state=42
    )
    
    valid_part = valid_part.sample(
        1000,
        random_state=42
    )
    
    print(f"Train subset shape: {train_part.shape}")
    print(f"Validation subset shape: {valid_part.shape}")

    print("Cleaning text...")

    train_part["text"] = train_part["text"].apply(
        clean_text
    )

    valid_part["text"] = valid_part["text"].apply(
        clean_text
    )

    for config_i in models_config:

        print("=" * 50)

        print(f"Loading configuration: {config_i}")

        config = load_config(
            f"configs/{config_i}"
        )

        model_name = config["model_name"]

        max_length = config["tokenization"]["max_length"]

        batch_size = config["training"]["batch_size"]

        learning_rate = float(
            config["training"]["learning_rate"]
        )

        num_train_ep = config["training"]["num_train_epochs"]

        print(f"Start training: {model_name}")

        print("Loading tokenizer...")

        tokenizer = load_tokenizer(
            model_name
        )

        print("Converting to Hugging Face dataset...")

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

        print("Loading model...")

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )

        print("Preparing training arguments...")

        args = TrainingArguments(
            output_dir=f"./models/{model_name}",

            num_train_epochs=num_train_ep,

            per_device_train_batch_size=batch_size,

            per_device_eval_batch_size=batch_size,

            eval_strategy="epoch",

            save_strategy="no",

            learning_rate=learning_rate,

            fp16=False,

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

        train_time_minutes = (
            time.time() - start_time
        ) / 60

        print("Running evaluation...")

        metrics = trainer.evaluate()

        accuracy = metrics["eval_accuracy"]

        print(f"Accuracy: {accuracy:.4f}")

        print(f"Train time: {train_time_minutes:.2f} min")

        result = {
            "model_name": model_name,
            "accuracy": round(accuracy, 4),
            "train_time_minutes": round(
                train_time_minutes,
                2
            ),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "notes": "baseline benchmark"
        }

        print("Saving benchmark result...")

        pd.DataFrame([result]).to_csv(
            "reports/experiments.csv",
            mode="a",
            header=False,
            index=False
        )

        print(result)

        print("Cleaning memory...")

        del model
        del trainer

        gc.collect()

        torch.mps.empty_cache()

    print("=" * 50)

    print("Benchmark completed.")


if __name__ == "__main__":
    main()