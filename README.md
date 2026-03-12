# KnowGraph

NLP and light ML environment powered by [uv](https://docs.astral.sh/uv/), with keyphrase extraction from PDF documents.

## Setup

```powershell
# Sync dependencies (creates .venv if needed)
uv sync

# Optional: KeyBERT (requires sentence-transformers, ~500MB)
uv sync --extra embeddings

# spaCy model for TextRank (run once)
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## Keyphrase Extraction

Extract key phrases from PDFs using YAKE, RAKE, TextRank, or KeyBERT:

```powershell
# YAKE (default, fast, no extra deps)
uv run python extract_keyphrases.py document.pdf -t 15 -v

# RAKE
uv run python extract_keyphrases.py document.pdf -m rake -t 15 -v

# TextRank (requires en_core_web_sm)
uv run python extract_keyphrases.py document.pdf -m textrank -t 15 -v

# KeyBERT (requires: uv sync --extra embeddings)
uv run python extract_keyphrases.py document.pdf -m keybert -t 15 -v
```

### CLI Options

| Option | Description |
|--------|-------------|
| `-m`, `--method` | `yake`, `rake`, `textrank`, `keybert` (default: yake) |
| `-t`, `--top` | Number of key phrases to extract |
| `-l`, `--language` | Language code for YAKE (e.g. en, pt) |
| `-n`, `--ngram` | Max n-gram size for YAKE/KeyBERT (1-3) |
| `-v`, `--verbose` | Show processing steps and timing |

With `-v`, timing is shown for PDF extraction and keyphrase extraction stages.

### Score Interpretation

| Method | Score meaning |
|--------|---------------|
| YAKE | Lower = more relevant |
| RAKE | Higher = more relevant |
| TextRank | Higher = more relevant |
| KeyBERT | Higher = more relevant |

### Python API

```python
from extract_keyphrases import extract_keyphrases_from_pdf

keywords = extract_keyphrases_from_pdf("document.pdf", method="yake", top=15, verbose=True)
for phrase, score in keywords:
    print(f"{phrase}: {score:.4f}")
```

## Installed Packages

| Package | Purpose |
|---------|---------|
| **yake** | Unsupervised keyword extraction |
| **rake-nltk** | RAKE keyword extraction |
| **pytextrank** | TextRank (graph-based) extraction |
| **pdfplumber** | PDF text extraction |
| **spacy** | NLP pipeline (TextRank) |
| **nltk** | Tokenization, stopwords |
| **scikit-learn** | ML utilities |
| **keybert** *(optional)* | BERT-based extraction |

## Notes

- **Text-based PDFs only**: pdfplumber extracts text from digital PDFs. Scanned/image PDFs require OCR (e.g. Tesseract).
- **Charts and tables**: PDFs with many figures/tables may produce noisy keyphrases; YAKE or KeyBERT often handle these better than RAKE.

## Quick Test

```powershell
uv run python example_usage.py
```

## Method Comparison on a pdf

Benchmark on a 149-page business major book (~214k chars, ~33k words), extracting top 15 key phrases:

| Method | PDF extraction | Keyphrase extraction | Total | Result quality |
|--------|----------------|----------------------|-------|----------------|
| **YAKE** | 9.0s | 1.9s | 10.9s | Domain-relevant terms (entrepreneurship, entrepreneurs, business, economic). Clean, focused output. |
| **RAKE** | 8.7s | 0.4s | 9.1s | Fastest, but heavily biased toward long name lists from acknowledgments. Less useful for topic extraction. |
| **TextRank** | 9.4s | 13.9s | 23.3s | Good domain phrases (business entrepreneurs, new businesses, entrepreneurial firms). Coherent, readable. |
| **KeyBERT** | 8.3s | 149.7s | 158.0s | Semantic, domain-aware (entrepreneurship, economics). Some PDF artifacts (e.g. hyphenation splits). Slowest due to model loading and embedding. |

**Summary:** YAKE and TextRank offer the best balance of quality and speed for this text-heavy PDF. RAKE is fastest but prone to noisy results on documents with acknowledgments or lists. KeyBERT is most semantically accurate but ~2.5 min for keyphrase extraction on 33k words.
