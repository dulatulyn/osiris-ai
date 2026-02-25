"""Submit a training job to Azure ML, wait for completion, then download artifacts."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from azure.ai.ml import MLClient, Output, command
from azure.identity import EnvironmentCredential


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit training job to Azure ML")
    parser.add_argument("--rows", type=int, default=50_000, help="Dataset rows to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.10, help="Fraud rate (0-1)")
    parser.add_argument("--n-estimators", type=int, default=500, help="XGBoost n_estimators")
    parser.add_argument("--max-depth", type=int, default=7, help="XGBoost max_depth")
    parser.add_argument("--lr", type=float, default=0.05, help="XGBoost learning rate")
    parser.add_argument("--model-version", type=str, default="v1", help="Artifact version tag (e.g. v1, v2)")
    parser.add_argument("--compute", type=str, default="serverless", help="Azure ML compute name or 'serverless'")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job completion and download artifacts locally",
    )
    args = parser.parse_args()

    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_ML_RESOURCE_GROUP")
    workspace = os.environ.get("AZURE_ML_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace]):
        print("ERROR: Missing Azure ML environment variables.")
        print("Required: AZURE_SUBSCRIPTION_ID, AZURE_ML_RESOURCE_GROUP, AZURE_ML_WORKSPACE_NAME")
        sys.exit(1)

    print("Authenticating to Azure ML...")
    credential = EnvironmentCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )
    print(f"Connected to workspace: {ml_client.workspace_name}")

    # The job generates a dataset and trains XGBoost, then copies artifacts to ./outputs/
    # so Azure ML captures them as a named output.
    cli_command = (
        "pip install -r requirements.txt && "
        f"python scripts/generate_dataset.py "
        f"  --rows {args.rows} "
        f"  --fraud-rate {args.fraud_rate} "
        f"  --output data/dataset.csv && "
        f"python -m src.main train "
        f"  --data data/dataset.csv "
        f"  --n-estimators {args.n_estimators} "
        f"  --max-depth {args.max_depth} "
        f"  --lr {args.lr} "
        f"  --model-version {args.model_version} && "
        f"mkdir -p outputs/{args.model_version} && "
        f"cp -r artifacts/{args.model_version}/. outputs/{args.model_version}/"
    )

    job = command(
        code="./",
        command=cli_command,
        environment="AzureML-sklearn-1.5-ubuntu22.04-py39-cpu@latest",
        compute=args.compute if args.compute != "serverless" else None,
        display_name=f"osiris-fraud-train-{args.model_version}",
        experiment_name="osiris-fraud-ml",
        outputs={
            "model_artifacts": Output(type="uri_folder", path=f"outputs/{args.model_version}"),
        },
    )

    print("Submitting training job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)

    print("=" * 60)
    print(f"Job submitted successfully!")
    print(f"  Job Name:   {returned_job.name}")
    print(f"  Studio URL: {returned_job.studio_url}")
    print("=" * 60)

    if not args.wait:
        print("Run with --wait to stream logs and download artifacts after completion.")
        return

    print("Streaming job logs (this may take several minutes)...")
    ml_client.jobs.stream(returned_job.name)

    print("Downloading job artifacts...")
    download_dir = Path("./downloads")
    download_dir.mkdir(exist_ok=True)
    ml_client.jobs.download(
        returned_job.name,
        download_path=str(download_dir),
        output_name="model_artifacts",
    )

    # Move downloaded artifacts to the expected local path
    src = download_dir / "named-outputs" / "model_artifacts"
    dst = Path(f"artifacts/{args.model_version}")

    if src.exists():
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        print(f"Artifacts saved to {dst}/")
    else:
        print(f"WARNING: Expected downloaded artifacts at {src} but directory not found.")
        print("Check the Azure ML Studio for job outputs.")

    print("Done.")


if __name__ == "__main__":
    main()
