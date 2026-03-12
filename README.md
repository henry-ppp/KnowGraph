# KnowGraph

NLP and light ML environment powered by [uv](https://docs.astral.sh/uv/).

## Setup

```powershell
# Sync dependencies (creates .venv if needed)
uv sync

# Optional: include embeddings (sentence-transformers, ~500MB)
uv sync --extra embeddings
```

## Usage

```powershell
# Run scripts (uv auto-activates .venv)
uv run python your_script.py

# Add new dependencies
uv add some-package

# Install spaCy language model (for English)
uv run python -m spacy download en_core_web_sm
```

## Installed Packages

| Package | Purpose |
|---------|---------|
| **spacy** | Industrial NLP (NER, parsing, tokenization) |
| **nltk** | Classic NLP (tokenization, stemming, corpora) |
| **scikit-learn** | ML (classification, clustering, TF-IDF) |
| **numpy** / **pandas** | Data handling |
| **sentence-transformers** *(optional)* | Semantic embeddings |

## Quick Test

```powershell
uv run python example_usage.py
```
