"""Feature-store-ready layer: entity key + event timestamp + typed schema (Feast-compatible layout)."""
import datetime as dt
import pandas as pd

SCHEMA = {"review_id": "int64", "event_timestamp": "datetime64[ns]", "clean_text": "string",
          "token_count": "int64", "char_count": "int64", "label": "int64"}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "review_id": range(len(df)),
        "event_timestamp": pd.Timestamp(dt.datetime.utcnow()),
        "clean_text": df["clean_text"],
        "token_count": df["clean_text"].str.split().str.len(),
        "char_count": df["clean_text"].str.len(),
        "label": df["label"]})
    return out.astype(SCHEMA)


def write_offline_store(feats: pd.DataFrame, path="features/sentiment_features.parquet"):
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    feats.to_parquet(path, index=False)
    return path
