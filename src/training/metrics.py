import numpy as np

from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    """
    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }