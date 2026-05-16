from datasets import load_dataset
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def load_imdb_data():
    """
    Load IMDb dataset from Hugging Face.
    """
    dataset = load_dataset("imdb")
    return dataset


def create_train_validation_split(test_size=0.2, random_state=RANDOM_SEED):
    """
    Load IMDb training split and create train/validation split.
    """
    dataset = load_imdb_data()

    train_df = dataset["train"].to_pandas()

    train_part, valid_part = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=train_df["label"]
    )

    return train_part, valid_part


if __name__ == "__main__":
    train_part, valid_part = create_train_validation_split()

    print(f"Train shape: {train_part.shape}")
    print(f"Validation shape: {valid_part.shape}")