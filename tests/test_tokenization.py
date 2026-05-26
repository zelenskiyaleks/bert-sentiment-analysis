from src.features.tokenization import (
    load_tokenizer,
    tokenize_batch
)


def test_tokenizer_loads():

    tokenizer = load_tokenizer(
        "bert-base-uncased"
    )

    assert tokenizer is not None


def test_tokenize_batch_returns_input_ids():

    tokenizer = load_tokenizer(
        "bert-base-uncased"
    )

    batch = {
        "text": ["Amazing movie!"]
    }

    tokenized = tokenize_batch(
        batch,
        tokenizer,
        max_length=128
    )

    assert "input_ids" in tokenized