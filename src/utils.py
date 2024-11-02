import numpy as np
import re
from sklearn.metrics import f1_score


def clean_text(text: str) -> str:
    # remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    # remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7f]", "", text)

    return text


def calculate_f1_score(y_true: np.array, y_pred: np.array, type: str) -> float:

    if type == "micro":
        return f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    elif type == "macro":
        return f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    elif type == "weighted":
        return f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    else:
        raise ValueError(
            "The type parameter must be the following: ['micro', 'macro', 'weighted]"
        )
