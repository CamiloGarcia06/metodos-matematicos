import re
import math
from typing import List, Dict, Tuple

from qdrant_client import QdrantClient


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def build_vocab(paragraphs: List[str]) -> List[str]:
    return sorted({w for p in paragraphs for w in tokenize(p)})


def build_idf(paragraphs: List[str], vocab: List[str]) -> Dict[str, float]:
    N = len(paragraphs)
    df: Dict[str, int] = {w: 0 for w in vocab}
    for p in paragraphs:
        seen = set(tokenize(p))
        for w in seen:
            if w in df:
                df[w] += 1
    return {w: (math.log(N / df[w]) if df[w] > 0 else 0.0) for w in vocab}


def tf_vector(words: List[str]) -> Dict[str, int]:
    tf: Dict[str, int] = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    return tf


def tfidf_vector(text: str, idf: Dict[str, float]) -> Dict[str, float]:
    tf = tf_vector(tokenize(text))
    return {w: tf.get(w, 0) * idf.get(w, 0.0) for w in idf.keys()}


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    # dot product over intersection
    dot = 0.0
    # iterate over smaller dict for speed
    keys = vec_a.keys() if len(vec_a) < len(vec_b) else vec_b.keys()
    for k in keys:
        dot += vec_a.get(k, 0.0) * vec_b.get(k, 0.0)
    # norms
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_paragraphs_by_tfidf_cosine(query: str, paragraphs: List[str], top_k: int = 5) -> List[Tuple[float, str]]:
    if not paragraphs:
        return []
    vocab = build_vocab(paragraphs)
    idf = build_idf(paragraphs, vocab)
    q_vec = tfidf_vector(query, idf)

    scored: List[Tuple[float, str]] = []
    for p in paragraphs:
        p_vec = tfidf_vector(p, idf)
        score = cosine_similarity(q_vec, p_vec)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[: max(1, min(top_k, len(scored)))]


def fetch_paragraphs_from_qdrant(client: QdrantClient, collection_name: str, limit: int = 1_000_000) -> List[str]:
    resp = client.scroll(collection_name=collection_name, with_payload=True, limit=limit)
    if hasattr(resp, "points"):
        points_list = resp.points
    elif isinstance(resp, tuple):
        points_list = resp[0]
    else:
        points_list = resp
    return [pt.payload.get("text", "") for pt in points_list if pt.payload and pt.payload.get("text")]


def rank_qdrant_by_tfidf_cosine(client: QdrantClient, collection_name: str, query: str, top_k: int = 5) -> List[Tuple[float, str]]:
    paragraphs = fetch_paragraphs_from_qdrant(client, collection_name)
    return rank_paragraphs_by_tfidf_cosine(query, paragraphs, top_k=top_k)


if __name__ == "__main__":
    import os
    import sys

    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    collection = os.environ.get("QDRANT_COLLECTION", "reglamento_estudiantil")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        try:
            query = input("🔎 Escribe tu pregunta: ")
        except EOFError:
            print("Uso: python -m app.local_rank '<pregunta>'")
            sys.exit(1)

    client = QdrantClient(host=host, port=port)
    results = rank_qdrant_by_tfidf_cosine(client, collection, query, top_k=5)

    print(f"\nTop {len(results)} párrafos relevantes (TF‑IDF + coseno):\n")
    for i, (score, text) in enumerate(results, 1):
        print(f"[{i}] score={score:.4f}\n{text}\n{'─'*40}")


