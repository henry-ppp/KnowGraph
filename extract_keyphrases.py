"""Extract key phrases from PDF documents using YAKE, RAKE, TextRank, or KeyBERT."""

import argparse
import sys
import time

import pdfplumber
import pytextrank
import yake
from rake_nltk import Rake


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg, file=sys.stderr)


def extract_text_from_pdf(path: str, verbose: bool = False) -> str:
    """Extract text from all pages of a PDF file."""
    chunks: list[str] = []
    with pdfplumber.open(path) as pdf:
        pages = pdf.pages
        _log(f"[1/3] Opening PDF: {path}", verbose)
        _log(f"      Pages: {len(pages)}", verbose)
        for i, page in enumerate(pages, 1):
            text = page.extract_text()
            if text:
                chunks.append(text)
            if verbose and i % 10 == 0:
                _log(f"      Processed {i}/{len(pages)} pages...", verbose)
    full_text = "\n".join(chunks)
    if verbose:
        words = len(full_text.split())
        _log(f"      Extracted {len(full_text):,} chars, ~{words:,} words", verbose)
    return full_text


def extract_keyphrases_yake(
    text: str,
    top: int = 20,
    lan: str = "en",
    n: int = 3,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """Extract key phrases from text using YAKE. Lower score = more relevant."""
    _log(f"[2/3] Running YAKE (lan={lan}, n={n}, top={top})...", verbose)
    extractor = yake.KeywordExtractor(lan=lan, n=n, top=top)
    keywords = extractor.extract_keywords(text)
    _log(f"      Extracted {len(keywords)} key phrases", verbose)
    return keywords


def extract_keyphrases_rake(
    text: str,
    top: int = 20,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """Extract key phrases from text using RAKE. Higher score = more relevant."""
    import nltk

    nltk.download("stopwords", quiet=True)
    _log(f"[2/3] Running RAKE (top={top})...", verbose)
    r = Rake()
    r.extract_keywords_from_text(text)
    # RAKE returns (score, phrase); normalize to (phrase, score) for consistency
    raw = r.get_ranked_phrases_with_scores()[:top]
    keywords = [(phrase, score) for score, phrase in raw]
    _log(f"      Extracted {len(keywords)} key phrases", verbose)
    return keywords


def extract_keyphrases_textrank(
    text: str,
    top: int = 20,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """Extract key phrases from text using TextRank. Higher score = more relevant."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Run: uv run python -m spacy download en_core_web_sm"
        )
    nlp.add_pipe("textrank")
    _log(f"[2/3] Running TextRank (top={top})...", verbose)
    doc = nlp(text)
    keywords = [
        (phrase.text, phrase.rank)
        for phrase in doc._.phrases[:top]
    ]
    _log(f"      Extracted {len(keywords)} key phrases", verbose)
    return keywords


def extract_keyphrases_keybert(
    text: str,
    top: int = 20,
    n: int = 3,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """Extract key phrases using KeyBERT. Higher score = more relevant."""
    try:
        from keybert import KeyBERT
    except ImportError:
        raise RuntimeError(
            "KeyBERT not installed. Run: uv sync --extra embeddings"
        )
    _log(f"[2/3] Running KeyBERT (top={top})...", verbose)
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, n),
        top_n=top,
        stop_words="english",
    )
    _log(f"      Extracted {len(keywords)} key phrases", verbose)
    return keywords


def extract_keyphrases(
    text: str,
    method: str = "yake",
    top: int = 20,
    lan: str = "en",
    n: int = 3,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """Extract key phrases using YAKE, RAKE, TextRank, or KeyBERT."""
    if method == "rake":
        return extract_keyphrases_rake(text, top=top, verbose=verbose)
    if method == "textrank":
        return extract_keyphrases_textrank(text, top=top, verbose=verbose)
    if method == "keybert":
        return extract_keyphrases_keybert(text, top=top, n=n, verbose=verbose)
    return extract_keyphrases_yake(text, top=top, lan=lan, n=n, verbose=verbose)


def extract_keyphrases_from_pdf(
    path: str,
    method: str = "yake",
    top: int = 20,
    lan: str = "en",
    n: int = 3,
    verbose: bool = False,
) -> list[tuple[str, float]]:
    """Extract key phrases from a PDF file using YAKE, RAKE, TextRank, or KeyBERT."""
    t0 = time.perf_counter()
    text = extract_text_from_pdf(path, verbose=verbose)
    t1 = time.perf_counter()
    if verbose:
        _log(f"      PDF extraction: {t1 - t0:.2f}s", verbose)
    if not text.strip():
        raise ValueError(f"No text extracted from {path}. Is it a scanned/image PDF?")
    keywords = extract_keyphrases(
        text, method=method, top=top, lan=lan, n=n, verbose=verbose
    )
    t2 = time.perf_counter()
    if verbose:
        _log(f"      Keyphrase extraction ({method}): {t2 - t1:.2f}s", verbose)
        _log(f"      Total: {t2 - t0:.2f}s", verbose)
    return keywords


def main() -> None:
    # Fix Windows console encoding for Unicode output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Extract key phrases from a PDF using YAKE, RAKE, TextRank, or KeyBERT"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "-m",
        "--method",
        choices=["yake", "rake", "textrank", "keybert"],
        default="yake",
        help="Extraction method (default: yake)",
    )
    parser.add_argument("-t", "--top", type=int, default=20, help="Number of key phrases to extract")
    parser.add_argument("-l", "--language", default="en", help="Language code for YAKE (e.g. en, pt)")
    parser.add_argument("-n", "--ngram", type=int, default=3, help="Max n-gram size for YAKE/KeyBERT (1-3)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show processing steps")
    args = parser.parse_args()

    try:
        keywords = extract_keyphrases_from_pdf(
            args.pdf_path,
            method=args.method,
            top=args.top,
            lan=args.language,
            n=args.ngram,
            verbose=args.verbose,
        )
        if args.verbose:
            _log("[3/3] Results:", args.verbose)
        for phrase, score in keywords:
            print(f"{phrase}: {score:.4f}")
    except FileNotFoundError:
        print(f"Error: File not found: {args.pdf_path}", file=sys.stderr)
        sys.exit(1)
    except (ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
