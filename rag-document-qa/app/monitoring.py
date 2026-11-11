"""MLflow logging + simple embedding drift detection."""
import time
import mlflow
import numpy as np
from scipy.stats import ks_2samp

mlflow.set_experiment("rag-document-qa")
_baseline: np.ndarray | None = None   # reference embedding norms
_recent: list[float] = []


def set_baseline(embeddings: list[list[float]]):
    global _baseline
    _baseline = np.linalg.norm(np.array(embeddings), axis=1)


def log_query(question: str, answer: str, latency: float, query_embedding: list[float], top_distance: float):
    _recent.append(float(np.linalg.norm(query_embedding)))
    drift_p = None
    if _baseline is not None and len(_recent) >= 30:
        drift_p = float(ks_2samp(_baseline, _recent[-100:]).pvalue)
    with mlflow.start_run(run_name=f"query-{int(time.time())}"):
        mlflow.log_param("question_len", len(question))
        mlflow.log_metrics({"latency_s": latency, "answer_len": len(answer), "top_distance": top_distance})
        if drift_p is not None:
            mlflow.log_metric("drift_pvalue", drift_p)
            mlflow.set_tag("data_drift", "detected" if drift_p < 0.05 else "none")
    return drift_p
