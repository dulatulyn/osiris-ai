import argparse
import os
from pathlib import Path


def _resolve_model_path(version: str) -> str:
    base = Path(__file__).resolve().parent.parent / "artifacts"
    return str(base / version)


def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("src.api.app:create_app", host=host, port=port, factory=True)


def run_train(args) -> None:
    from src.training.trainer import train
    train(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        lr=args.lr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Osiris Fraud Detection ML")
    subparsers = parser.add_subparsers(dest="command", required=True)

    api_parser = subparsers.add_parser("serve", help="Start the FastAPI prediction server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=8000)
    api_parser.add_argument("--model-version", type=str, default=None, help="Model version to load (e.g. v1)")

    train_parser = subparsers.add_parser("train", help="Train the fraud detection model")
    train_parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV")
    train_parser.add_argument("--n-estimators", type=int, default=500)
    train_parser.add_argument("--max-depth", type=int, default=7)
    train_parser.add_argument("--lr", type=float, default=0.05)
    train_parser.add_argument("--model-version", type=str, default="v1", help="Model version name (e.g. v1, v2)")
    # Backwards compat flags, ignored
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch-size", type=int, default=None)

    args = parser.parse_args()

    if args.command == "serve":
        if args.model_version:
            os.environ["MODEL_PATH"] = _resolve_model_path(args.model_version)
        run_api(host=args.host, port=args.port)

    elif args.command == "train":
        os.environ["MODEL_PATH"] = _resolve_model_path(args.model_version)
        run_train(args)


if __name__ == "__main__":
    main()
