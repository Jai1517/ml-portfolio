"""SageMaker inference handler."""
import joblib, os, json, pandas as pd
def model_fn(d): return joblib.load(os.path.join(d, "model.joblib"))
def input_fn(body, ct): return pd.DataFrame(json.loads(body))
def predict_fn(data, model): return model.predict(data)
def output_fn(pred, accept): return json.dumps(pred.tolist())
