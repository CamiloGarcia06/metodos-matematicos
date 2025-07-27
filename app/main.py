import os
import fitz  # PyMuPDF
import uuid
import re
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

VECTOR_DIM = 300
COLLECTION_NAME = "reglamento_estudiantil"


def basic_embedding(text):
    vector = [0] * VECTOR_DIM
    words = re.findall(r"\w+", text.lower())
    for i, w in enumerate(words[:VECTOR_DIM]):
        vector[i] = len(w)
    return vector


def extract_paragraphs_from_pdf(file_path):
    doc = fitz.open(file_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text() + "\n\n"
    raw_paras = re.split(r"\n\n", all_text)
    paragraphs = []
    for p in raw_paras:
        single_line = " ".join([ln.strip() for ln in p.splitlines() if ln.strip()])
        if single_line:
            paragraphs.append(single_line)
    return paragraphs


def create_collection_if_not_exists(client, name):
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def store_chunks(paragraphs, client):
    print(f"🔍 Procesando {len(paragraphs)} párrafos...")
    points = [
        PointStruct(
            id=str(uuid.uuid4()), vector=basic_embedding(p), payload={"text": p}
        )
        for p in paragraphs
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    total = client.scroll(collection_name=COLLECTION_NAME, limit=1_000_000)
    print(f"✅ Almacenados. Total en Qdrant: {len(total)} puntos.")


def main():
    file_path = "/app/reglamento.pdf"
    if not os.path.exists(file_path):
        print("❌ Archivo no encontrado.")
        return

    print("📄 Extrayendo párrafos del PDF...")
    paragraphs = extract_paragraphs_from_pdf(file_path)
    print(f"✅ Extraídos {len(paragraphs)} párrafos. Ejemplos:")
    for i, p in enumerate(paragraphs[:5], 1):
        print(f"  {i}. {p[:80]}…")

    client = QdrantClient(host="qdrant", port=6333)
    create_collection_if_not_exists(client, COLLECTION_NAME)
    store_chunks(paragraphs, client)


if __name__ == "__main__":
    main()
