"""Extract topics from PDF documents using OpenAI embeddings, UMAP, HDBSCAN, and LLM labeling."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, file=sys.stderr)


@dataclass
class Topic:
    """A topic extracted from a document cluster."""

    label: str
    cluster_id: int
    chunk_indices: list[int]
    representative_excerpts: list[str]


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    try:
        import tiktoken
    except ImportError:
        raise RuntimeError(
            "tiktoken not installed. Run: uv sync --extra topics"
        )
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def extract_text_pymupdf(path: str, verbose: bool = False) -> str:
    """Extract text from PDF using PyMuPDF. Strips headers, footers, page numbers."""
    try:
        import pymupdf
    except ImportError:
        raise RuntimeError(
            "pymupdf not installed. Run: uv sync --extra topics"
        )
    _log(f"[1/6] Extracting text with PyMuPDF: {path}", verbose)
    doc = pymupdf.open(path)
    chunks: list[str] = []
    page_num_re = re.compile(r"^\d+$")
    for i, page in enumerate(doc):
        text = page.get_text()
        if text:
            lines = text.splitlines()
            filtered: list[str] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if page_num_re.match(line) and len(line) <= 4:
                    continue
                filtered.append(line)
            if filtered:
                chunks.append("\n".join(filtered))
        if verbose and (i + 1) % 20 == 0:
            _log(f"      Processed {i + 1}/{len(doc)} pages...", verbose)
    doc.close()
    full_text = "\n\n".join(chunks)
    if verbose:
        words = len(full_text.split())
        _log(f"      Extracted {len(full_text):,} chars, ~{words:,} words", verbose)
    return full_text


def _split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Recursively split text by separators until chunks are small enough."""
    if _count_tokens(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in separators:
        if sep not in text:
            continue
        parts = text.split(sep)
        if len(parts) < 2:
            continue
        result: list[str] = []
        current = ""
        for i, part in enumerate(parts):
            candidate = (current + sep + part).strip() if current else part.strip()
            if not candidate:
                continue
            if _count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if _count_tokens(part) > chunk_size:
                    sub = _split_recursive(part, separators[1:], chunk_size, overlap)
                    result.extend(sub)
                    current = ""
                else:
                    current = part.strip()
        if current:
            result.append(current)
        if result:
            return result
    return [text]


def chunk_text_recursive(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_tokens: int = 20,
    verbose: bool = False,
) -> list[str]:
    """Split text into semantic chunks using recursive splitting."""
    _log("[2/6] Chunking text (recursive splitter)...", verbose)
    separators = ["\n\n", "\n", ". ", " "]
    raw = _split_recursive(text, separators, chunk_size, overlap)
    chunks = [c.strip() for c in raw if c.strip() and _count_tokens(c) >= min_chunk_tokens]
    if verbose:
        _log(f"      Created {len(chunks)} chunks", verbose)
    return chunks


def _get_embedding_encoding(model: str) -> str:
    """Return tiktoken encoding for embedding model."""
    if "3-small" in model or "3-large" in model:
        return "cl100k_base"
    return "cl100k_base"


async def _embed_batch(
    client: "OpenAI",
    chunks: list[str],
    model: str,
    dimensions: int | None,
) -> list[list[float]]:
    """Embed a single batch of chunks."""
    kwargs: dict = {"model": model, "input": chunks}
    if dimensions is not None and ("3-small" in model or "3-large" in model):
        kwargs["dimensions"] = dimensions
    response = await client.embeddings.create(**kwargs)
    return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


async def embed_chunks_async(
    chunks: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int | None = 256,
    batch_tokens: int = 250_000,
    max_concurrent: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """Embed chunks using OpenAI API with async batching."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError(
            "openai not installed. Run: uv sync --extra topics"
        )
    _log("[3/6] Generating embeddings (async, batched)...", verbose)
    client = AsyncOpenAI()
    encoding = _get_embedding_encoding(model)
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
    except ImportError:
        enc = None

    def token_count(t: str) -> int:
        if enc:
            return len(enc.encode(t))
        return len(t) // 4

    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0
    for chunk in chunks:
        tc = token_count(chunk)
        if tc > 8192:
            chunk = chunk[:32000]
            tc = token_count(chunk)
        if current_tokens + tc > batch_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [chunk]
            current_tokens = tc
        else:
            current_batch.append(chunk)
            current_tokens += tc
    if current_batch:
        batches.append(current_batch)

    if verbose:
        _log(f"      {len(batches)} batch(es), {len(chunks)} chunks", verbose)

    sem = asyncio.Semaphore(max_concurrent)

    async def limited_embed(b: list[str]) -> list[list[float]]:
        async with sem:
            return await _embed_batch(client, b, model, dimensions)

    results = await asyncio.gather(*[limited_embed(b) for b in batches])
    all_embeddings = [emb for batch in results for emb in batch]
    return np.array(all_embeddings, dtype=np.float32)


def embed_chunks(
    chunks: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int | None = 256,
    batch_tokens: int = 250_000,
    max_concurrent: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """Synchronous wrapper for embed_chunks_async."""
    return asyncio.run(
        embed_chunks_async(
            chunks, model, dimensions, batch_tokens, max_concurrent, verbose
        )
    )


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 15,
    min_dist: float = 0.0,
    random_state: int = 42,
    verbose: bool = False,
) -> np.ndarray:
    """Reduce embedding dimensions using UMAP."""
    try:
        from umap import UMAP
    except ImportError:
        raise RuntimeError(
            "umap-learn not installed. Run: uv sync --extra topics"
        )
    _log("[4/6] Reducing dimensions with UMAP...", verbose)
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=min_dist,
        random_state=random_state,
    )
    reduced = reducer.fit_transform(embeddings)
    if verbose:
        _log(f"      {embeddings.shape[1]}D -> {n_components}D", verbose)
    return reduced


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    metric: str = "euclidean",
    verbose: bool = False,
) -> np.ndarray:
    """Cluster embeddings using HDBSCAN."""
    try:
        import hdbscan
    except ImportError:
        raise RuntimeError(
            "hdbscan not installed. Run: uv sync --extra topics"
        )
    _log("[5/6] Clustering with HDBSCAN...", verbose)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels) - {-1})
    if verbose:
        _log(f"      Found {n_clusters} clusters (+ noise)", verbose)
    return labels


def _label_clusters_llm(
    chunks: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    label_model: str = "gpt-4o-mini",
    k_nearest: int = 5,
    excerpt_chars: int = 200,
    verbose: bool = False,
) -> dict[int, str]:
    """Label clusters using LLM on representative chunks."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "openai not installed. Run: uv sync --extra topics"
        )
    _log("[6/6] Labeling clusters with LLM...", verbose)
    client = OpenAI()
    unique_labels = sorted(set(labels) - {-1})
    if not unique_labels:
        return {}

    cluster_chunk_indices: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl >= 0:
            cluster_chunk_indices.setdefault(lbl, []).append(i)

    labels_to_label = [lbl for lbl in unique_labels if lbl in cluster_chunk_indices]
    result: dict[int, str] = {}

    for cluster_id in labels_to_label:
        indices = cluster_chunk_indices[cluster_id]
        centroid = embeddings[indices].mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        nearest_indices = np.argsort(distances)[: min(k_nearest, len(indices))]
        excerpts = []
        for j, idx in enumerate(nearest_indices):
            chunk = chunks[idx]
            excerpt = chunk[:excerpt_chars] + ("..." if len(chunk) > excerpt_chars else "")
            excerpts.append(f"[{j + 1}] {excerpt}")
        combined = "\n\n".join(excerpts)

        system = (
            "You are an expert at summarizing document themes. "
            "Given representative text excerpts from a cluster of similar passages, "
            "produce a short, concise topic label (3–6 words). "
            "Output only the label, no explanation or punctuation."
        )
        response = client.chat.completions.create(
            model=label_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": combined},
            ],
            temperature=0.3,
        )
        label_text = response.choices[0].message.content or "Unlabeled"
        label_text = label_text.strip().strip('"\'')
        result[cluster_id] = label_text

    if verbose:
        _log(f"      Labeled {len(result)} clusters", verbose)
    return result


def _label_clusters_yake(
    chunks: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    k_nearest: int = 5,
    verbose: bool = False,
) -> dict[int, str]:
    """Label clusters using YAKE on representative chunks. Runs YAKE on each selected
    chunk individually, then picks the most frequent top keyword (or from nearest chunk)."""
    try:
        import yake
    except ImportError:
        raise RuntimeError("yake not installed. Run: uv sync")
    _log("[6/6] Labeling clusters with YAKE (per selected chunk)...", verbose)
    unique_labels = sorted(set(labels) - {-1})
    result: dict[int, str] = {}
    extractor = yake.KeywordExtractor(lan="en", n=2, top=3)

    for cluster_id in unique_labels:
        indices = np.where(labels == cluster_id)[0]
        centroid = embeddings[indices].mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        nearest_indices = np.argsort(distances)[: min(k_nearest, len(indices))]
        selected_chunks = [chunks[i] for i in nearest_indices]

        keywords_from_chunks: list[str] = []
        for chunk in selected_chunks:
            if not chunk.strip():
                continue
            kw = extractor.extract_keywords(chunk)
            if kw:
                keywords_from_chunks.append(kw[0][0].lower())

        if not keywords_from_chunks:
            result[cluster_id] = "Unlabeled"
            continue
        counts: dict[str, int] = {}
        for kw in keywords_from_chunks:
            counts[kw] = counts.get(kw, 0) + 1
        best = max(counts, key=lambda k: (counts[k], -keywords_from_chunks.index(k)))
        result[cluster_id] = best

    if verbose:
        _log(f"      Labeled {len(result)} clusters", verbose)
    return result


def label_clusters(
    chunks: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    method: str = "llm",
    label_model: str = "gpt-4o-mini",
    k_nearest: int = 5,
    verbose: bool = False,
) -> dict[int, str]:
    """Label clusters using LLM or YAKE."""
    if method == "yake":
        return _label_clusters_yake(chunks, labels, embeddings, k_nearest, verbose)
    return _label_clusters_llm(
        chunks, labels, embeddings, label_model, k_nearest, verbose=verbose
    )


def _load_embedding_cache(cache_path: Path, chunk_hashes: list[str]) -> np.ndarray | None:
    """Load embeddings from cache if all chunk hashes match."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
        cached_hashes = data.get("hashes", [])
        if cached_hashes != chunk_hashes:
            return None
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return embeddings
    except (json.JSONDecodeError, KeyError):
        return None


def _save_embedding_cache(cache_path: Path, hashes: list[str], embeddings: np.ndarray) -> None:
    """Save embeddings to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {"hashes": hashes, "embeddings": embeddings.tolist()},
            f,
            indent=0,
        )


def extract_topics_from_pdf(
    path: str,
    *,
    model: str = "text-embedding-3-small",
    dimensions: int | None = 256,
    min_cluster_size: int = 3,
    use_umap: bool = True,
    umap_components: int = 5,
    label_method: str = "llm",
    label_model: str = "gpt-4o-mini",
    cache_dir: str | Path = ".cache/topics",
    use_cache: bool = True,
    verbose: bool = False,
) -> list[Topic]:
    """Extract topics from a PDF using the full embedding-based pipeline."""
    t0 = time.perf_counter()
    text = extract_text_pymupdf(path, verbose=verbose)
    t1 = time.perf_counter()
    if verbose:
        _log(f"      Extraction: {t1 - t0:.2f}s", verbose)
    if not text.strip():
        raise ValueError(f"No text extracted from {path}. Is it a scanned/image PDF?")

    chunks = chunk_text_recursive(text, verbose=verbose)
    if len(chunks) < 2:
        raise ValueError(f"Too few chunks ({len(chunks)}). Document may be too short.")

    chunk_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]
    cache_path = Path(cache_dir) / "embeddings.json" if use_cache else None

    if use_cache and cache_path:
        cached = _load_embedding_cache(cache_path, chunk_hashes)
        if cached is not None:
            embeddings = cached
            if verbose:
                _log("[3/6] Loaded embeddings from cache", verbose)
        else:
            embeddings = embed_chunks(
                chunks, model=model, dimensions=dimensions, verbose=verbose
            )
            _save_embedding_cache(cache_path, chunk_hashes, embeddings)
    else:
        embeddings = embed_chunks(
            chunks, model=model, dimensions=dimensions, verbose=verbose
        )

    if use_umap:
        to_cluster = reduce_umap(
            embeddings, n_components=umap_components, verbose=verbose
        )
    else:
        to_cluster = embeddings

    labels = cluster_hdbscan(
        to_cluster, min_cluster_size=min_cluster_size, verbose=verbose
    )

    cluster_labels = label_clusters(
        chunks, labels, embeddings, method=label_method, label_model=label_model, verbose=verbose
    )

    topics: list[Topic] = []
    for cluster_id in sorted(cluster_labels.keys()):
        indices = list(np.where(labels == cluster_id)[0])
        centroid = embeddings[indices].mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        nearest = np.argsort(distances)[:5]
        excerpts = [chunks[i][:200] + ("..." if len(chunks[i]) > 200 else "") for i in nearest]
        topics.append(
            Topic(
                label=cluster_labels[cluster_id],
                cluster_id=cluster_id,
                chunk_indices=indices,
                representative_excerpts=excerpts,
            )
        )

    if verbose:
        _log(f"      Total: {time.perf_counter() - t0:.2f}s", verbose)
    return topics


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Extract topics from a PDF using OpenAI embeddings, UMAP, HDBSCAN, and LLM labeling"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=256,
        help="Embedding dimensions (default: 256)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="HDBSCAN min cluster size (default: 3)",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP dimensionality reduction",
    )
    parser.add_argument(
        "--label-method",
        choices=["llm", "yake"],
        default="llm",
        help="Topic labeling method (default: llm)",
    )
    parser.add_argument(
        "--label-model",
        default="gpt-4o-mini",
        help="LLM for topic labeling (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/topics",
        help="Embedding cache directory (default: .cache/topics)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show processing steps")
    args = parser.parse_args()

    try:
        topics = extract_topics_from_pdf(
            args.pdf_path,
            model=args.model,
            dimensions=args.dimensions,
            min_cluster_size=args.min_cluster_size,
            use_umap=not args.no_umap,
            label_method=args.label_method,
            label_model=args.label_model,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            verbose=args.verbose,
        )
        for t in topics:
            print(f'Topic: "{t.label}" ({len(t.chunk_indices)} chunks)')
    except FileNotFoundError:
        print(f"Error: File not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
