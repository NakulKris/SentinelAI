"""
preprocess.py — SentinelAI
Shared NLP preprocessing pipeline.
Must be identical to the pipeline used during training.
"""
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

_STOP_WORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()
_URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_RE    = re.compile(r"<[^>]+>")
_EMAIL_RE   = re.compile(r"\S+@\S+")
_NUMBER_RE  = re.compile(r"\b\d+\b")
_WHITESPACE = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """
    Apply the documented SentinelAI NLP preprocessing pipeline:
    1. Lowercase  2. URL removal  3. HTML removal  4. Email removal
    5. Number removal  6. Punctuation removal  7. Whitespace normalization
    8. Tokenization  9. Stopword removal  10. Lemmatization  11. Reconstruct
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NUMBER_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WHITESPACE.sub(" ", text).strip()
    tokens = word_tokenize(text)
    tokens = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in _STOP_WORDS
    ]
    return " ".join(tokens)
