# Phishing Email Detection Model

A small, local machine-learning project that classifies an email as `phishing`
or `legitimate` from its subject and body. It uses TF-IDF text features and a
logistic-regression classifier.

> **Demo only:** `data/emails.csv` contains a small hand-written sample dataset
> so you can try the workflow immediately. Its metrics are not representative
> of real-world performance. Use a large, diverse, carefully labeled dataset
> and validate it for your intended use before relying on predictions. This
> model is a decision-support example, not a security guarantee.

## Set up

From this directory, create and activate a virtual environment, then install
the project:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Train

Train on the included demo data and save a model under `artifacts/`:

```bash
phishing-detector train
```

Use a different CSV or model destination:

```bash
phishing-detector train --data path/to/emails.csv --model artifacts/custom.joblib
```

The training command reports held-out accuracy, precision, recall, and F1
before fitting the saved model on the complete dataset. Keep both classes in
the dataset, with at least two examples of each class.

## Predict

After training, provide an email subject and body:

```bash
phishing-detector predict \
  --subject "Action required: verify your account" \
  --body "Use this link to restore access to your account."
```

The command prints the predicted label and phishing probability. Model files
are loaded with `joblib`; only load model files you created or trust.

## Dataset format

Provide a UTF-8 CSV with these exact headers and one email per row:

```csv
subject,body,label
"Meeting notes","Here are the notes from today's meeting.","legitimate"
"Account alert","Verify your account immediately using this link.","phishing"
```

Labels must be `phishing` or `legitimate` (case-insensitive). The model joins
the subject and body as text; it does not open links or inspect attachments.

## Run tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```
