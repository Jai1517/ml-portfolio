"""Scikit-learn pipeline with MLflow tracking and DVC-tracked outputs."""
import json, os
import joblib, mlflow, mlflow.sklearn, pandas as pd, yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

p = yaml.safe_load(open("params.yaml"))
df = pd.read_csv(p["data"]["path"])
X, y = df.drop(columns="target"), df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=p["data"]["test_size"], random_state=p["data"]["seed"])

pipe = Pipeline([("scale", StandardScaler()),
                 ("clf", RandomForestClassifier(**p["model"], random_state=p["data"]["seed"]))])

mlflow.set_experiment("multicloud-mlops")
with mlflow.start_run():
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    metrics = {"accuracy": accuracy_score(yte, pred), "f1": f1_score(yte, pred)}
    mlflow.log_params(p["model"]); mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipe, "model")

os.makedirs("model", exist_ok=True)
joblib.dump(pipe, "model/model.joblib")
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(metrics)
