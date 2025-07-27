from qdrant_client import QdrantClient
import re


# Embedding “dummy” igual al del indexado
def basic_embedding(text):
    VECTOR_DIM = 300
    vector = [0] * VECTOR_DIM
    words = re.findall(r"\w+", text.lower())
    for i, w in enumerate(words[:VECTOR_DIM]):
        vector[i] = len(w)
    return vector


if __name__ == "__main__":
    client = QdrantClient(host="qdrant", port=6333)
    query = input("🔎 Escribe tu pregunta: ")
    query_vec = basic_embedding(query)

    # ← Aquí cambiamos de `search` a `query_points`
    hits = client.query_points(
        collection_name="reglamento_estudiantil",
        query_vector=query_vec,
        limit=5,
    )

    print(f"\nSe han recuperado {len(hits)} resultados:\n")
    for i, hit in enumerate(hits, start=1):
        print(f"[{i}] score={hit.score:.4f}\n{hit.payload['text']}\n{'─'*40}")
