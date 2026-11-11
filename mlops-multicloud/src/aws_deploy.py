"""Upload artifact to S3 and deploy a SageMaker endpoint."""
import os, tarfile, boto3, sagemaker
from sagemaker.sklearn.model import SKLearnModel

BUCKET = os.environ["S3_BUCKET"]; ROLE = os.environ["SAGEMAKER_ROLE_ARN"]

with tarfile.open("model.tar.gz", "w:gz") as t:
    t.add("model/model.joblib", arcname="model.joblib")
key = "models/sklearn/model.tar.gz"
boto3.client("s3").upload_file("model.tar.gz", BUCKET, key)

model = SKLearnModel(model_data=f"s3://{BUCKET}/{key}", role=ROLE,
                     entry_point="src/inference.py", framework_version="1.2-1")
predictor = model.deploy(initial_instance_count=1, instance_type="ml.m5.large", endpoint_name="sklearn-mlops")
print("Endpoint:", predictor.endpoint_name)
