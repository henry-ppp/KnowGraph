"""Example usage of NLP and ML tools in this environment."""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK - download required data on first run
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# spaCy - requires: uv run python -m spacy download en_core_web_sm
# import spacy
# nlp = spacy.load("en_core_web_sm")

# Example: NLTK tokenization
text = "Natural language processing enables machines to understand human language."
tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words("english"))
filtered = [t for t in tokens if t.isalnum() and t not in stop_words]
print("NLTK tokens:", filtered)

# Example: TF-IDF with scikit-learn
docs = [
    "machine learning and natural language",
    "natural language processing tools",
    "machine learning algorithms",
]
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(docs)
print("TF-IDF shape:", tfidf.shape)

# Optional: sentence-transformers (install with: uv sync --extra embeddings)
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(["Hello world", "NLP is fun"])
# print("Embeddings shape:", embeddings.shape)
