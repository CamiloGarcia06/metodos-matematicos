import re
import math
from qdrant_client import QdrantClient

COLLECTION_NAME = "reglamento_estudiantil"

def build_vocab(paragraphs):
    return sorted({w for p in paragraphs for w in re.findall(r'\w+', p.lower())})

def build_idf(paragraphs, vocab):
    N = len(paragraphs)
    df = {w: 0 for w in vocab}
    for p in paragraphs:
        for w in set(re.findall(r'\w+', p.lower())):
            if w in df:
                df[w] += 1
    return {w: math.log(N / df[w]) if df[w] > 0 else 0.0 for w in vocab}

def tfidf_vector(text, vocab, idf):
    words = re.findall(r'\w+', text.lower())
    tf = {}
    for w in words:
        if w in idf:
            tf[w] = tf.get(w, 0) + 1
    return [tf.get(w, 0) * idf.get(w, 0) for w in vocab]

if __name__ == "__main__":
    client = QdrantClient(host="qdrant", port=6333)

    # 1) Recupera todos los puntos via scroll
    resp = client.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=1_000_000
    )
    # Soporta ambos retornos: objeto o tupla
    if hasattr(resp, "points"):
        points_list = resp.points
    elif isinstance(resp, tuple):
        points_list = resp[0]
    else:
        points_list = resp

    paragraphs = [pt.payload["text"] for pt in points_list]

    # 2) Reconstruye vocabulario e IDF
    vocab = build_vocab(paragraphs)
    idf   = build_idf(paragraphs, vocab)

    # 3) Lee la pregunta y vectoriza
    query = input("🔎 Escribe tu pregunta: ")
    q_vec = tfidf_vector(query, vocab, idf)

    # 4) Busca los 20 más cercanos
    resp_hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=20,
        with_payload=True,
    )
    raw_hits = resp_hits.points if hasattr(resp_hits, "points") else resp_hits

    # 5) Filtra duplicados y toma top‑5
    seen, unique = set(), []
    for hit in raw_hits:
        txt = hit.payload["text"]
        if txt not in seen:
            seen.add(txt)
            unique.append(hit)
        if len(unique) >= 5:
            break

    # 6) Muestra resultados
    print(f"\nTop {len(unique)} párrafos relevantes:\n")
    for i, hit in enumerate(unique, 1):
        print(f"[{i}] score={hit.score:.4f}\n{hit.payload['text']}\n{'─'*40}")
