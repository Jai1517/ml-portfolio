"""Generate a demo dataset (swap for your real data source)."""
import os
from sklearn.datasets import load_breast_cancer
os.makedirs("data", exist_ok=True)
df = load_breast_cancer(as_frame=True).frame
df.to_csv("data/raw.csv", index=False)
print("saved", df.shape)
