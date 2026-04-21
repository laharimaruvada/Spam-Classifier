import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
_STOP_WORDS = set(stopwords.words("english"))
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in _STOP_WORDS]
    return " ".join(tokens)
