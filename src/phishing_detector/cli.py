"""Command-line interface for training and using the email classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

from phishing_detector.model import predict_email, train_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "emails.csv"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "phishing_model.joblib"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishing-detector",
        description="Train and use a phishing email text classifier.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and save a model")
    train_parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    train_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)

    predict_parser = commands.add_parser("predict", help="classify an email")
    predict_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    predict_parser.add_argument("--subject", required=True)
    predict_parser.add_argument("--body", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        metrics = train_model(args.data, args.model)
        print(f"Saved model to: {args.model}")
        print("Held-out evaluation (demo dataset; metrics may be unstable):")
        for name, value in metrics.items():
            print(f"  {name.capitalize():<10} {value:.3f}")
    elif args.command == "predict":
        result = predict_email(args.model, args.subject, args.body)
        print(f"Prediction: {result['label']}")
        print(
            "Phishing probability: "
            f"{result['phishing_probability']:.1%}"
        )


if __name__ == "__main__":
    main()
