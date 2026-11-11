"""Deploy to Google Vertex AI."""
import os
from google.cloud import aiplatform
aiplatform.init(project=os.environ["GCP_PROJECT"], location="europe-west2",
                staging_bucket=os.environ["GCS_BUCKET"])
model = aiplatform.Model.upload(
    display_name="sklearn-mlops", artifact_uri=os.environ["GCS_MODEL_DIR"],  # dir containing model.joblib
    serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest")
endpoint = model.deploy(machine_type="n1-standard-2")
print("Endpoint:", endpoint.resource_name)
