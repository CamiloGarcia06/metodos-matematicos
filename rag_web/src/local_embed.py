import re
import math
from typing import List, Dict


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _build_vocab(paragraphs: List[str]) -> List[str]:
    return sorted({w for p in paragraphs for w in _tokenize(p)})


def _build_idf(paragraphs: List[str], vocab: List[str]) -> Dict[str, float]:
    N = len(paragraphs)
    df: Dict[str, int] = {w: 0 for w in vocab}
    for p in paragraphs:
        seen = set(_tokenize(p))
        for w in seen:
            if w in df:
                df[w] += 1
    return {w: (math.log(N / df[w]) if df[w] > 0 else 0.0) for w in vocab}


def _tf(words: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for w in words:
        out[w] = out.get(w, 0) + 1
    return out


def _tfidf_vector(text: str, idf: Dict[str, float]) -> List[float]:
    tf = _tf(_tokenize(text))
    # Preserve vocabulary order by iterating idf keys
    return [tf.get(w, 0) * idf.get(w, 0.0) for w in idf.keys()]


def embed_paragraphs_tfidf(paragraphs: List[str]) -> List[List[float]]:
    """Compute TF‑IDF vectors for all paragraphs.
    Returns the matrix of vectors. The vocabulary order is internal to IDF dict.
    """
    if not paragraphs:
        return []
    vocab = _build_vocab(paragraphs)
    idf = _build_idf(paragraphs, vocab)
    return [_tfidf_vector(p, idf) for p in paragraphs]


def embed_query_tfidf(query: str, paragraphs: List[str]) -> List[float]:
    """Compute TF‑IDF vector for a query using IDF built from the same paragraphs."""
    if not paragraphs:
        return []
    vocab = _build_vocab(paragraphs)
    idf = _build_idf(paragraphs, vocab)
    return _tfidf_vector(query, idf)


