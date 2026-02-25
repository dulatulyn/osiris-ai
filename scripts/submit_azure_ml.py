import argparse
import os
import sys

from azure.ai.ml import MLClient, command, Input
from azure.identity import EnvironmentCredential


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--compute", type=str, default="serverless")
    parser.add_argument("--environment-name", type=str, default="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest")
    args = parser.parse_args()

    print("Authenticating to Azure ML...")
    credential = EnvironmentCredential()
    
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
    resource_group = os.environ.get("AZURE_ML_RESOURCE_GROUP")
    workspace = os.environ.get("AZURE_ML_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace]):
        print("Error: Missing Azure ML environment variables.")
        print("Required: AZURE_SUBSCRIPTION_ID, AZURE_ML_RESOURCE_GROUP, AZURE_ML_WORKSPACE_NAME")
        sys.exit(1)

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

    print(f"Connected to Workspace: {ml_client.workspace_name}")

    # Here we submit the command job to run `src.main train` on Azure ML compute
    # It assumes data/dataset.csv is created or pulled from an Azure ML Datastore/Asset in a real scenario.
    # We upload the local code folder to the cluster.
    cli_command = (
        "pip install -r requirements.txt && "
        "python scripts/generate_dataset.py --rows 50000 --fraud-rate 0.10 --output data/dataset.csv && "
        f"python -m src.main train --data data/dataset.csv --epochs {args.epochs} --batch-size 512"
    )

    job = command(
        code="./",  # upload the whole project repo to Azure
        command=cli_command,
        environment=args.environment_name,
        compute=args.compute if args.compute != "serverless" else None,
        display_name="osiris-fraud-training",
        experiment_name="osiris-fraud-ml"
    )

    print("Submitting training job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    
    print("====================================")
    print("Job submitted successfully!")
    print(f"Job Name: {returned_job.name}")
    print(f"Studio URL: {returned_job.studio_url}")
    print("====================================")


if __name__ == "__main__":
    main()
