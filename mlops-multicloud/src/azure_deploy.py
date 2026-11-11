"""Deploy to Azure ML managed online endpoint."""
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
from azure.identity import DefaultAzureCredential

ml = MLClient(DefaultAzureCredential(), os.environ["AZ_SUB"], os.environ["AZ_RG"], os.environ["AZ_WS"])
model = ml.models.create_or_update(Model(path="model/model.joblib", name="sklearn-mlops", type="custom_model"))
ep = ManagedOnlineEndpoint(name="sklearn-mlops-ep", auth_mode="key")
ml.online_endpoints.begin_create_or_update(ep).result()
dep = ManagedOnlineDeployment(name="blue", endpoint_name=ep.name, model=model,
                              instance_type="Standard_DS3_v2", instance_count=1)
ml.online_deployments.begin_create_or_update(dep).result()
