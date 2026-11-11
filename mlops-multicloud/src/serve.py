"""Inference server (run on AWS EC2 / any container host)."""
import joblib, pandas as pd
from fastapi import FastAPI
app = FastAPI()
model = joblib.load("model/model.joblib")

@app.post("/predict")
def predict(rows: list[dict]):
    return {"predictions": model.predict(pd.DataFrame(rows)).tolist()}
