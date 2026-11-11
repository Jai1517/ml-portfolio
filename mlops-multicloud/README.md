# End-to-End MLOps Pipeline – Multi-Cloud
Scikit-learn pipeline, DVC for data/model versioning, MLflow tracking, DevSecOps CI (bandit + pip-audit),
deployment to AWS (EC2/S3/SageMaker), Google Vertex AI, Azure ML.

```bash
pip install -r requirements.txt
dvc init && dvc repro        # prepare -> train
mlflow ui
python src/aws_deploy.py     # needs S3_BUCKET, SAGEMAKER_ROLE_ARN
python src/gcp_deploy.py     # needs GCP_PROJECT, GCS_BUCKET, GCS_MODEL_DIR
python src/azure_deploy.py   # needs AZ_SUB, AZ_RG, AZ_WS
uvicorn src.serve:app        # EC2 / container serving
```
