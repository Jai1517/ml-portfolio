"""Text cleaning with NLTK (stopwords) + spaCy (lemmatisation)."""
import re
import nltk
import spacy
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    raise SystemExit("Run: python -m spacy download en_core_web_sm")
STOP = set(stopwords.words("english")) - {"not", "no", "never"}  # keep negations for sentiment


def clean(text: str) -> str:
    text = re.sub(r"http\S+|[^a-zA-Z\s]", " ", text.lower())
    return " ".join(t.lemma_ for t in nlp(text) if t.text not in STOP and len(t.text) > 1)
