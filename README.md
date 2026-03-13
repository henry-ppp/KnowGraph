# KnowGraph

Experiments on different approaches of extracting key topics from a long pdf, with NLP and light ML environment powered by [uv](https://docs.astral.sh/uv/).

## Setup

```powershell
# Sync dependencies (creates .venv if needed)
uv sync

# Optional: KeyBERT (requires sentence-transformers, ~500MB)
uv sync --extra embeddings

# Optional: Topic extraction (OpenAI embeddings, UMAP, HDBSCAN)
uv sync --extra topics

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

## Topic Extraction

Extract clustered topics from PDFs using OpenAI embeddings, UMAP, HDBSCAN, and LLM labeling. Requires `OPENAI_API_KEY` and `uv sync --extra topics`.

Create a `.env` file (copy from `.env.example`) with your OpenAI API key:
```powershell
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

```powershell
# LLM labeling (default, recommended)
uv run python extract_topics.py document.pdf -v

# YAKE labeling (offline, no API)
uv run python extract_topics.py document.pdf --label-method yake -v
```

### Pipeline

1. **Text extraction** — PyMuPDF (fast, strips headers/footers)
2. **Semantic chunking** — Recursive splitter (paragraphs → sentences → words)
3. **Embedding** — OpenAI `text-embedding-3-small` (async, batched)
4. **Dimensionality reduction** — UMAP (default), PCA (faster), or none. *Skipped when using agglomerative.*
5. **Clustering** — HDBSCAN (auto topic count) or Agglomerative (fixed `n_clusters`, clusters on raw embeddings)
6. **Topic labeling** — LLM (default) or YAKE on representative chunks

### CLI Options

| Option | Default | Description |
|--------|---------|--------------|
| `--model` | text-embedding-3-small | Embedding model |
| `--dimensions` | 256 | Embedding dimensions |
| `--cluster-method` | hdbscan | `hdbscan` or `agglomerative`. Agglomerative skips reduction. |
| `--n-clusters` | 15 | Number of clusters (agglomerative only) |
| `--min-cluster-size` | 3 | HDBSCAN min cluster size (hdbscan only) |
| `--reduce-method` | umap | `umap`, `pca` (faster), or `none`. Ignored when `--cluster-method agglomerative`. |
| `--reduce-components` | 5 | Output dimensions for umap/pca |
| `--no-umap` | - | Skip reduction (same as `--reduce-method none`) |
| `--label-method` | llm | `llm` or `yake` |
| `--label-model` | gpt-4o-mini | LLM for topic labeling |
| `--cache-dir` | .cache/topics | Embedding cache directory |
| `--no-cache` | - | Disable embedding cache |
| `--output-dir` | output/topics | Save intermediate and final results |
| `--no-output` | - | Do not save results to disk |
| `-v` | - | Verbose (timing per stage, token usage) |

Results are saved to `{output-dir}/{pdf_stem}/`:
- `chunks.json` — text chunks
- `embeddings.npy` — embedding vectors
- `umap_reduced.npy` — Reduced vectors (when using umap or pca)
- `cluster_labels.json` — Cluster label per chunk (-1 = noise for HDBSCAN only; agglomerative assigns all)
- `topics.json` — final topics with labels, chunk indices, and `is_paratext` (true when content is boilerplate: copyright, TOC, acknowledgements, etc.)
- `metadata.json` — run parameters, per-step timing, token usage

### Python API

```python
from extract_topics import extract_topics_from_pdf

topics = extract_topics_from_pdf("document.pdf", verbose=True)
for t in topics:
    print(f'Topic: "{t.label}" ({len(t.chunk_indices)} chunks)')

# With token usage (also shown when verbose=True)
usage = {}
topics = extract_topics_from_pdf("document.pdf", usage=usage)
print(f"Tokens: embedding={usage['embedding_tokens']:,}, LLM={usage['llm_total_tokens']:,}")
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
| **openai** *(topics)* | Embedding and chat API |
| **pymupdf** *(topics)* | Fast PDF text extraction |
| **umap-learn** *(topics)* | Dimensionality reduction |
| **hdbscan** *(topics)* | Density-based clustering |
| **tiktoken** *(topics)* | Token counting for batching |

## Notes

- **Text-based PDFs only**: pdfplumber (keyphrases) and PyMuPDF (topics) extract text from digital PDFs. Scanned/image PDFs require OCR (e.g. Tesseract).
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
