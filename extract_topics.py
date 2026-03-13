"""Extract topics from PDF documents using OpenAI embeddings, UMAP, HDBSCAN, and LLM labeling."""

from __future__ import annotations

import argparse
import asyncio
import hashlib

from dotenv import load_dotenv

load_dotenv()

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
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
    is_paratext: bool = False


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
) -> tuple[list[list[float]], int]:
    """Embed a single batch of chunks. Returns (embeddings, token_count)."""
    kwargs: dict = {"model": model, "input": chunks}
    if dimensions is not None and ("3-small" in model or "3-large" in model):
        kwargs["dimensions"] = dimensions
    response = await client.embeddings.create(**kwargs)
    embeddings = [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
    tokens = response.usage.total_tokens if response.usage else 0
    return embeddings, tokens


async def embed_chunks_async(
    chunks: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int | None = 256,
    batch_tokens: int = 250_000,
    max_concurrent: int = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """Embed chunks using OpenAI API with async batching. Returns (embeddings, total_tokens)."""
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

    async def limited_embed(b: list[str]) -> tuple[list[list[float]], int]:
        async with sem:
            return await _embed_batch(client, b, model, dimensions)

    results = await asyncio.gather(*[limited_embed(b) for b in batches])
    all_embeddings = [emb for batch, _ in results for emb in batch]
    total_tokens = sum(tokens for _, tokens in results)
    if verbose:
        _log(f"      Embedding tokens: {total_tokens:,}", verbose)
    return np.array(all_embeddings, dtype=np.float32), total_tokens


def embed_chunks(
    chunks: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int | None = 256,
    batch_tokens: int = 250_000,
    max_concurrent: int = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """Synchronous wrapper for embed_chunks_async. Returns (embeddings, total_tokens)."""
    return asyncio.run(
        embed_chunks_async(
            chunks, model, dimensions, batch_tokens, max_concurrent, verbose
        )
    )


def reduce_pca(
    embeddings: np.ndarray,
    n_components: int = 5,
    verbose: bool = False,
) -> np.ndarray:
    """Reduce embedding dimensions using PCA. Much faster than UMAP."""
    from sklearn.decomposition import PCA

    _log("[4/6] Reducing dimensions with PCA...", verbose)
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    reducer = PCA(n_components=n_components, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    if verbose:
        _log(f"      {embeddings.shape[1]}D -> {n_components}D", verbose)
    return reduced


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int = 5,
    n_neighbors: int = 10,
    n_epochs: int = 100,
    min_dist: float = 0.0,
    random_state: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Reduce embedding dimensions using UMAP. Uses random_state=None for parallelism."""
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
        n_epochs=n_epochs,
        init="pca",
        min_dist=min_dist,
        low_memory=False,
        random_state=random_state,
        n_jobs=-1,
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


def cluster_agglomerative(
    embeddings: np.ndarray,
    n_clusters: int = 15,
    linkage: str = "ward",
    metric: str = "euclidean",
    verbose: bool = False,
) -> np.ndarray:
    """Cluster embeddings using Agglomerative Clustering. No dimensionality reduction needed."""
    from sklearn.cluster import AgglomerativeClustering

    _log("[5/6] Clustering with Agglomerative...", verbose)
    n_clusters = min(n_clusters, len(embeddings))
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=metric,
    )
    labels = clusterer.fit_predict(embeddings)
    if verbose:
        _log(f"      Found {n_clusters} clusters", verbose)
    return labels


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    min_cluster_size: int = 3,
    n_clusters: int = 15,
    verbose: bool = False,
) -> np.ndarray:
    """Cluster embeddings using HDBSCAN or Agglomerative."""
    if method == "agglomerative":
        return cluster_agglomerative(
            embeddings, n_clusters=n_clusters, verbose=verbose
        )
    return cluster_hdbscan(
        embeddings, min_cluster_size=min_cluster_size, verbose=verbose
    )


def _label_clusters_llm(
    chunks: list[str],
    labels: np.ndarray,
    embeddings: np.ndarray,
    label_model: str = "gpt-4o-mini",
    k_nearest: int = 5,
    excerpt_chars: int = 200,
    verbose: bool = False,
) -> tuple[dict[int, str], dict[str, int]]:
    """Label clusters using LLM on representative chunks. Returns (labels, usage)."""
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
        return {}, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    cluster_chunk_indices: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        if lbl >= 0:
            cluster_chunk_indices.setdefault(lbl, []).append(i)

    labels_to_label = [lbl for lbl in unique_labels if lbl in cluster_chunk_indices]
    result: dict[int, str] = {}
    total_prompt = 0
    total_completion = 0

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
            "Output only the label, no explanation or punctuation.\n\n"
            "If the content is paratext (boilerplate that is not main body content), "
            "prefix your label with 'Paratext: ' followed by a short type. "
            "Paratext includes: title page, copyright notice, table of contents, "
            "acknowledgements, publisher/author info, publication lists, ordering info, "
            "and similar front/back matter. Examples: 'Paratext: Table of contents', "
            "'Paratext: Copyright and publisher info', 'Paratext: IEA publications list'. "
            "For normal topical content, output just the topic label."
        )
        response = client.chat.completions.create(
            model=label_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": combined},
            ],
            temperature=0.3,
        )
        if response.usage:
            total_prompt += response.usage.prompt_tokens
            total_completion += response.usage.completion_tokens
        label_text = response.choices[0].message.content or "Unlabeled"
        label_text = label_text.strip().strip('"\'')
        result[cluster_id] = label_text

    usage = {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
    }
    if verbose:
        _log(f"      Labeled {len(result)} clusters", verbose)
        _log(f"      LLM tokens: {usage['total_tokens']:,} (prompt: {usage['prompt_tokens']:,}, completion: {usage['completion_tokens']:,})", verbose)
    return result, usage


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
) -> tuple[dict[int, str], dict[str, int] | None]:
    """Label clusters using LLM or YAKE. Returns (labels, usage). Usage is None for YAKE."""
    if method == "yake":
        return _label_clusters_yake(chunks, labels, embeddings, k_nearest, verbose), None
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


def _to_json_serializable(obj: object) -> object:
    """Convert numpy types to Python types for JSON serialization."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    return obj


def _save_pipeline_results(
    out_dir: Path,
    *,
    chunks: list[str],
    embeddings: np.ndarray,
    umap_reduced: np.ndarray | None,
    labels: np.ndarray,
    topics: list[Topic],
    metadata: dict,
) -> None:
    """Save intermediate and final results for further analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    np.save(out_dir / "embeddings.npy", embeddings)
    if umap_reduced is not None:
        np.save(out_dir / "umap_reduced.npy", umap_reduced)
    with open(out_dir / "cluster_labels.json", "w", encoding="utf-8") as f:
        json.dump([int(x) for x in labels.tolist()], f)

    with open(out_dir / "topics.json", "w", encoding="utf-8") as f:
        json.dump(
            [_to_json_serializable(asdict(t)) for t in topics],
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(metadata), f, indent=2)


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
    cluster_method: str = "hdbscan",
    n_clusters: int = 15,
    reduce_method: str = "umap",
    reduce_components: int = 5,
    label_method: str = "llm",
    label_model: str = "gpt-4o-mini",
    cache_dir: str | Path = ".cache/topics",
    use_cache: bool = True,
    verbose: bool = False,
    usage: dict[str, int] | None = None,
    output_dir: str | Path | None = "output/topics",
) -> list[Topic]:
    """Extract topics from a PDF using the full embedding-based pipeline.

    If usage is provided (e.g. usage={}), it will be populated with embedding_tokens,
    llm_prompt_tokens, llm_completion_tokens, llm_total_tokens.

    If output_dir is provided, saves intermediate and final results to output_dir/{pdf_stem}/.
    """
    step_times: dict[str, float] = {}
    t0 = time.perf_counter()
    text = extract_text_pymupdf(path, verbose=verbose)
    step_times["extraction"] = time.perf_counter() - t0
    if verbose:
        _log(f"      Extraction: {step_times['extraction']:.2f}s", verbose)
    if not text.strip():
        raise ValueError(f"No text extracted from {path}. Is it a scanned/image PDF?")

    t1 = time.perf_counter()
    chunks = chunk_text_recursive(text, verbose=verbose)
    step_times["chunking"] = time.perf_counter() - t1
    if verbose:
        _log(f"      Chunking: {step_times['chunking']:.2f}s", verbose)
    if len(chunks) < 2:
        raise ValueError(f"Too few chunks ({len(chunks)}). Document may be too short.")

    chunk_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in chunks]
    cache_path = Path(cache_dir) / "embeddings.json" if use_cache else None
    embedding_tokens = 0

    t2 = time.perf_counter()
    if use_cache and cache_path:
        cached = _load_embedding_cache(cache_path, chunk_hashes)
        if cached is not None:
            embeddings = cached
            if verbose:
                _log("[3/6] Loaded embeddings from cache", verbose)
        else:
            embeddings, embedding_tokens = embed_chunks(
                chunks, model=model, dimensions=dimensions, verbose=verbose
            )
            _save_embedding_cache(cache_path, chunk_hashes, embeddings)
    else:
        embeddings, embedding_tokens = embed_chunks(
            chunks, model=model, dimensions=dimensions, verbose=verbose
        )
    step_times["embedding"] = time.perf_counter() - t2
    if verbose:
        _log(f"      Embedding: {step_times['embedding']:.2f}s", verbose)

    t3 = time.perf_counter()
    if cluster_method == "agglomerative":
        to_cluster = embeddings
        reduced = None
        if verbose:
            _log("[4/6] Skipping reduction (agglomerative clusters on raw embeddings)", verbose)
    elif reduce_method == "none":
        to_cluster = embeddings
        reduced = None
    elif reduce_method == "umap":
        to_cluster = reduce_umap(
            embeddings, n_components=reduce_components, verbose=verbose
        )
        reduced = to_cluster
    elif reduce_method == "pca":
        to_cluster = reduce_pca(
            embeddings, n_components=reduce_components, verbose=verbose
        )
        reduced = to_cluster
    step_times["reduction"] = time.perf_counter() - t3
    if verbose and cluster_method != "agglomerative":
        _log(f"      Reduction ({reduce_method}): {step_times['reduction']:.2f}s", verbose)

    t4 = time.perf_counter()
    labels = cluster_embeddings(
        to_cluster,
        method=cluster_method,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        verbose=verbose,
    )
    step_times["clustering"] = time.perf_counter() - t4
    if verbose:
        _log(f"      Clustering: {step_times['clustering']:.2f}s", verbose)

    t5 = time.perf_counter()
    cluster_labels, llm_usage = label_clusters(
        chunks, labels, embeddings, method=label_method, label_model=label_model, verbose=verbose
    )
    step_times["labeling"] = time.perf_counter() - t5
    if verbose:
        _log(f"      Labeling: {step_times['labeling']:.2f}s", verbose)

    topics: list[Topic] = []
    for cluster_id in sorted(cluster_labels.keys()):
        raw_label = cluster_labels[cluster_id]
        is_paratext = raw_label.strip().lower().startswith("paratext:")
        indices = list(np.where(labels == cluster_id)[0])
        centroid = embeddings[indices].mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        nearest = np.argsort(distances)[:5]
        excerpts = [chunks[i][:200] + ("..." if len(chunks[i]) > 200 else "") for i in nearest]
        topics.append(
            Topic(
                label=raw_label,
                cluster_id=cluster_id,
                chunk_indices=indices,
                representative_excerpts=excerpts,
                is_paratext=is_paratext,
            )
        )

    total_time = time.perf_counter() - t0
    if verbose:
        _log(f"      Total: {total_time:.2f}s", verbose)
        _log("", verbose)
        _log("Step timing:", verbose)
        for step, sec in step_times.items():
            _log(f"  {step}: {sec:.2f}s", verbose)

    if output_dir is not None:
        pdf_stem = Path(path).stem
        out_path = Path(output_dir) / pdf_stem
        metadata = {
            "pdf_path": str(path),
            "model": model,
            "dimensions": dimensions,
            "cluster_method": cluster_method,
            "min_cluster_size": min_cluster_size,
            "n_clusters": n_clusters,
            "reduce_method": "skipped" if cluster_method == "agglomerative" else reduce_method,
            "reduce_components": reduce_components,
            "label_method": label_method,
            "label_model": label_model,
            "n_chunks": len(chunks),
            "n_clusters_found": len(cluster_labels),
            "total_time_seconds": round(total_time, 2),
            "step_times_seconds": {k: round(v, 2) for k, v in step_times.items()},
            "embedding_tokens": embedding_tokens,
            "llm_prompt_tokens": llm_usage["prompt_tokens"] if llm_usage else 0,
            "llm_completion_tokens": llm_usage["completion_tokens"] if llm_usage else 0,
            "llm_total_tokens": llm_usage["total_tokens"] if llm_usage else 0,
        }
        _save_pipeline_results(
            out_path,
            chunks=chunks,
            embeddings=embeddings,
            umap_reduced=reduced,
            labels=labels,
            topics=topics,
            metadata=metadata,
        )
        if verbose:
            _log(f"      Results saved to {out_path}", verbose)

    if usage is not None:
        usage["embedding_tokens"] = embedding_tokens
        if llm_usage:
            usage["llm_prompt_tokens"] = llm_usage["prompt_tokens"]
            usage["llm_completion_tokens"] = llm_usage["completion_tokens"]
            usage["llm_total_tokens"] = llm_usage["total_tokens"]
        else:
            usage["llm_prompt_tokens"] = 0
            usage["llm_completion_tokens"] = 0
            usage["llm_total_tokens"] = 0
        if verbose:
            _log("", verbose)
            _log("Token usage:", verbose)
            _log(f"  Embedding: {usage['embedding_tokens']:,} tokens", verbose)
            _log(f"  LLM:      {usage['llm_total_tokens']:,} tokens (prompt: {usage['llm_prompt_tokens']:,}, completion: {usage['llm_completion_tokens']:,})", verbose)
            _log(f"  Total:    {usage['embedding_tokens'] + usage['llm_total_tokens']:,} tokens", verbose)
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
        "--cluster-method",
        choices=["hdbscan", "agglomerative"],
        default="hdbscan",
        help="Clustering method: hdbscan (default) or agglomerative. Agglomerative skips reduction.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=15,
        help="Number of clusters for agglomerative (default: 15). Ignored for hdbscan.",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="HDBSCAN min cluster size (default: 3). Used only with --cluster-method hdbscan.",
    )
    parser.add_argument(
        "--reduce-method",
        choices=["umap", "pca", "none"],
        default="umap",
        help="Dimensionality reduction: umap (default), pca (faster), or none",
    )
    parser.add_argument(
        "--reduce-components",
        type=int,
        default=5,
        help="Output dimensions for umap/pca (default: 5)",
    )
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip dimensionality reduction (same as --reduce-method none)",
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
    parser.add_argument(
        "--output-dir",
        default="output/topics",
        help="Directory to save intermediate and final results (default: output/topics)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Do not save results to disk",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show processing steps")
    args = parser.parse_args()

    try:
        usage: dict[str, int] = {}
        topics = extract_topics_from_pdf(
            args.pdf_path,
            model=args.model,
            dimensions=args.dimensions,
            min_cluster_size=args.min_cluster_size,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            reduce_method="none" if args.no_umap else args.reduce_method,
            reduce_components=args.reduce_components,
            label_method=args.label_method,
            label_model=args.label_model,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            verbose=args.verbose,
            usage=usage,
            output_dir=None if args.no_output else args.output_dir,
        )
        for t in topics:
            marker = " [Paratext]" if t.is_paratext else ""
            print(f'Topic: "{t.label}" ({len(t.chunk_indices)} chunks){marker}')
    except FileNotFoundError:
        print(f"Error: File not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
