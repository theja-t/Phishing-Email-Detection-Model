"""Dataset loading, model training, and email prediction."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypedDict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = {"phishing", "legitimate"}
REQUIRED_COLUMNS = {"subject", "body", "label"}


class EmailPrediction(TypedDict):
    label: str
    phishing_probability: float


def load_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    """Read and validate a CSV dataset, returning email text and labels."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    texts: list[str] = []
    labels: list[str] = []
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                "Dataset is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(reader, start=2):
            subject = (row.get("subject") or "").strip()
            body = (row.get("body") or "").strip()
            label = (row.get("label") or "").strip().lower()
            if label not in LABELS:
                raise ValueError(
                    f"Invalid label on CSV row {row_number}: {label!r}. "
                    "Use 'phishing' or 'legitimate'."
                )
            if not subject and not body:
                raise ValueError(
                    f"CSV row {row_number} has an empty subject and body."
                )
            texts.append(f"{subject} {body}".strip())
            labels.append(label)

    counts = {label: labels.count(label) for label in LABELS}
    too_few = [label for label, count in counts.items() if count < 2]
    if too_few:
        raise ValueError(
            "The dataset must contain at least two examples of each class; "
            f"not enough examples for: {', '.join(sorted(too_few))}."
        )
    return texts, labels


def create_pipeline() -> Pipeline:
    """Build the text vectorizer and classifier pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    max_features=20_000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_model(
    data_path: str | Path, model_path: str | Path
) -> dict[str, float]:
    """Evaluate on a held-out split, then fit and save a model on all rows."""
    texts, labels = load_dataset(data_path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=max(0.25, 2 / len(labels)),
        random_state=42,
        stratify=labels,
    )

    evaluation_model = create_pipeline()
    evaluation_model.fit(train_texts, train_labels)
    predictions = evaluation_model.predict(test_texts)
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(
            precision_score(
                test_labels, predictions, pos_label="phishing", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                test_labels, predictions, pos_label="phishing", zero_division=0
            )
        ),
        "f1": float(
            f1_score(test_labels, predictions, pos_label="phishing", zero_division=0)
        ),
    }

    final_model = create_pipeline()
    final_model.fit(texts, labels)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, destination)
    return metrics


def predict_email(
    model_path: str | Path, subject: str, body: str
) -> EmailPrediction:
    """Predict an email label and its phishing probability."""
    if not subject.strip() and not body.strip():
        raise ValueError("Provide a non-empty email subject or body.")

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {path}. Run 'phishing-detector train' first."
        )

    model: Pipeline = joblib.load(path)
    text = f"{subject.strip()} {body.strip()}".strip()
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    phishing_probability = float(probabilities[classes.index("phishing")])
    prediction = str(model.predict([text])[0])
    return EmailPrediction(
        label=prediction,
        phishing_probability=phishing_probability,
    )
