import argparse
import os
import sys
from pathlib import Path


def _resolve_model_path(version: str) -> str:
    base = Path(__file__).resolve().parent.parent / "artifacts"
    return str(base / version)


def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("src.api.app:create_app", host=host, port=port, factory=True)


def run_train(args: list[str]) -> None:
    from src.training.trainer import main as train_main
    sys.argv = ["train"] + args
    train_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Osiris Fraud Detection ML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("serve", help="Start the FastAPI prediction server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--model-version", type=str, default=None, help="Model version to load (e.g. v1)")

    train_parser = subparsers.add_parser("train", help="Train the fraud detection model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=512)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--model-version", type=str, default="v1", help="Model version name (e.g. v1, v2)")

    args = parser.parse_args()

    if args.command == "serve":
        if args.model_version:
            os.environ["MODEL_PATH"] = _resolve_model_path(args.model_version)
        run_api(host=args.host, port=args.port)

    elif args.command == "train":
        os.environ["MODEL_PATH"] = _resolve_model_path(args.model_version)
        train_args = []
        train_args.extend(["--data", args.data])
        train_args.extend(["--epochs", str(args.epochs)])
        train_args.extend(["--batch-size", str(args.batch_size)])
        train_args.extend(["--lr", str(args.lr)])
        run_train(train_args)


if __name__ == "__main__":
    main()
