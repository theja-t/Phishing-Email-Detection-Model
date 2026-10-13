import csv

import pytest

from phishing_detector.model import load_dataset, predict_email, train_model


def test_load_dataset_combines_email_fields(tmp_path):
    dataset = tmp_path / "emails.csv"
    dataset.write_text(
        "subject,body,label\n"
        "Account alert,Verify your account,phishing\n"
        "Team update,Meeting at noon,legitimate\n"
        "Gift offer,Claim it now,phishing\n"
        "Project notes,Notes attached,legitimate\n",
        encoding="utf-8",
    )

    texts, labels = load_dataset(dataset)

    assert texts[0] == "Account alert Verify your account"
    assert labels == ["phishing", "legitimate", "phishing", "legitimate"]


def test_load_dataset_rejects_unknown_labels(tmp_path):
    dataset = tmp_path / "emails.csv"
    dataset.write_text(
        "subject,body,label\n"
        "Account alert,Verify your account,spam\n"
        "Team update,Meeting at noon,legitimate\n"
        "Gift offer,Claim it now,phishing\n"
        "Project notes,Notes attached,legitimate\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid label"):
        load_dataset(dataset)


def test_train_and_predict(tmp_path):
    dataset = tmp_path / "emails.csv"
    rows = [
        ("Weekly update", "The team meeting is Thursday", "legitimate"),
        ("Order shipped", "Track your order in your account", "legitimate"),
        ("Lunch plans", "Let me know which day works", "legitimate"),
        ("Project notes", "The schedule is on the intranet", "legitimate"),
        ("Verify account now", "Send your password to keep access", "phishing"),
        ("Claim free prize", "Enter your bank details at this link", "phishing"),
        ("Urgent mailbox alert", "Click here and enter your login", "phishing"),
        ("Payment overdue", "Open the file and enable macros", "phishing"),
    ]
    with dataset.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["subject", "body", "label"])
        writer.writerows(rows)
    model_path = tmp_path / "artifacts" / "model.joblib"

    metrics = train_model(dataset, model_path)
    result = predict_email(
        model_path,
        "Verify account",
        "Click here to enter your password now",
    )

    assert model_path.is_file()
    assert set(metrics) == {"accuracy", "precision", "recall", "f1"}
    assert result["label"] in {"phishing", "legitimate"}
    assert 0.0 <= result["phishing_probability"] <= 1.0


def test_predict_rejects_empty_email():
    with pytest.raises(ValueError, match="non-empty"):
        predict_email("missing.joblib", " ", "")
