from qdrant_client import QdrantClient
import re

VECTOR_DIM = 300
COLLECTION = "reglamento_estudiantil"

def basic_embedding(text):
    vector = [0] * VECTOR_DIM
    words = re.findall(r'\w+', text.lower())
    for i, w in enumerate(words[:VECTOR_DIM]):
        vector[i] = len(w)
    return vector

if __name__ == "__main__":
    client = QdrantClient(host="qdrant", port=6333)
    query = input("🔎 Escribe tu pregunta: ")
    q_vec  = basic_embedding(query)

    # 1) Pedimos más resultados de los que necesitamos
    resp = client.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        limit=20,           # recuperar 20
        with_payload=True,
    )

    raw_hits = resp.points if hasattr(resp, "points") else resp

    # 2) Filtrar duplicados manteniendo orden
    seen = set()
    unique_hits = []
    for hit in raw_hits:
        text = hit.payload["text"]
        if text not in seen:
            seen.add(text)
            unique_hits.append(hit)
        if len(unique_hits) >= 5:
            break

    # 3) Mostrar sólo top‑5 únicos
    print(f"\nTop {len(unique_hits)} párrafos distintos:\n")
    for i, hit in enumerate(unique_hits, 1):
        print(f"[{i}] score={hit.score:.4f}\n{hit.payload['text']}\n{'─'*40}")
