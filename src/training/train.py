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

def main():

    print("Loading configuration...")

    config = load_config(
        "configs/bert_base.yaml"
    )

    model_name = config["model_name"]

    max_length = config["tokenization"]["max_length"]

    batch_size = config["training"]["batch_size"]
    
    learning_rate = float(config["training"]["learning_rate"])

    num_train_ep = config["training"]["num_train_epochs"]

    print("Loading dataset...")

    train_part, valid_part = create_train_validation_split()

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
        output_dir="./models",

        num_train_epochs=num_train_ep,

        per_device_train_batch_size=batch_size,

        per_device_eval_batch_size=batch_size,

        eval_strategy="epoch",

        save_strategy="epoch",

        learning_rate= learning_rate,

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

    trainer.train()

    print("Running evaluation...")

    metrics = trainer.evaluate()

    print(metrics)


if __name__ == "__main__":
    main()