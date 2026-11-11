"""Sentiment analysis: NLTK + spaCy preprocessing, TF-IDF features, Logistic Regression."""
import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess import clean
from feature_store import build_features, write_offline_store

# Demo data; replace with e.g. IMDB / Amazon reviews CSV (columns: text, label)
DATA = [("I loved this movie, fantastic acting", 1), ("Absolutely terrible, a waste of time", 0),
        ("Great product and fast delivery", 1), ("Worst purchase ever, it broke instantly", 0),
        ("Really enjoyable and well written", 1), ("Not good, very disappointing", 0),
        ("Brilliant service, highly recommend", 1), ("Awful experience, never again", 0),
        ("Superb quality, exceeded expectations", 1), ("Boring plot and bad ending", 0),
        ("Happy with the result, works perfectly", 1), ("Poor quality and rude staff", 0)] * 10

df = pd.DataFrame(DATA, columns=["text", "label"])
df["clean_text"] = df["text"].map(clean)
print("Offline feature store:", write_offline_store(build_features(df)))

Xtr, Xte, ytr, yte = train_test_split(df["clean_text"], df["label"], test_size=0.25,
                                      stratify=df["label"], random_state=42)
pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                 ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
pred = pipe.predict(Xte)
p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="binary")
print(f"precision={p:.3f} recall={r:.3f} f1={f1:.3f}")
print(classification_report(yte, pred, target_names=["negative", "positive"]))
joblib.dump(pipe, "sentiment_model.joblib")
print("Prediction:", pipe.predict([clean("This was an amazing experience")]))
