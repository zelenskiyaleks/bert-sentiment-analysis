import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from src.data.preprocessing import clean_text
from src.utils.config import load_config


LABELS = {
    0: "Negative",
    1: "Positive"
}

def predict_sentiment(text):

    print("Loading inference config...")

    config = load_config(
        "configs/inference.yaml"
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_name"]
    )

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_path"]
    )

    print("Cleaning text...")

    cleaned_text = clean_text(text)

    print("Tokenizing text...")

    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt"
    )

    print("Running inference...")

    model.eval()
    
    with torch.no_grad():
    
        outputs = model(**inputs)
    
        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )
    
        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()
    
        confidence = probabilities[
            0,
            prediction
        ].item()
    
    predicted_label = LABELS[prediction]
    
    return predicted_label, confidence

def main():

    text = input(
        "Enter movie review: "
    )

    prediction, confidence = predict_sentiment(
        text
    )

    print("\nPrediction Result")

    print(
        f"Sentiment: {prediction}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )


if __name__ == "__main__":
    main()