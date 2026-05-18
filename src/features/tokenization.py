from transformers import AutoTokenizer

def load_tokenizer(model_name):
    """
    Load tokenizer for a transformer model.
    """
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_batch(
    examples,
    tokenizer,
    max_length
    ):
    """
    Tokenize batch of texts.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length
    )