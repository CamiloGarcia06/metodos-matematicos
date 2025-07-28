import os
import fitz  # PyMuPDF
import uuid
import re
import math
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

COLLECTION_NAME = "reglamento_estudiantil"


def build_vocab(paragraphs):
    """Construye un vocabulario ordenado de todas las palabras."""
    return sorted({w for p in paragraphs for w in re.findall(r"\w+", p.lower())})


def build_idf(paragraphs, vocab):
    """
    Calcula el IDF para cada palabra del vocabulario:
      idf(w) = log(N / df(w))
    donde df(w) = número de párrafos que contienen w.
    """
    N = len(paragraphs)
    df = {w: 0 for w in vocab}
    for p in paragraphs:
        for w in set(re.findall(r"\w+", p.lower())):
            if w in df:
                df[w] += 1
    return {w: math.log(N / df[w]) if df[w] > 0 else 0.0 for w in vocab}


def tfidf_vector(text, vocab, idf):
    """
    Calcula el vector TF‑IDF de un texto dado:
      tf(w) = count(w en text)
      luego vector[i] = tf(vocab[i]) * idf[vocab[i]]
    """
    words = re.findall(r"\w+", text.lower())
    tf = {}
    for w in words:
        if w in idf:
            tf[w] = tf.get(w, 0) + 1
    return [tf.get(w, 0) * idf.get(w, 0) for w in vocab]


def extract_paragraphs_from_pdf(file_path):
    """Extrae párrafos agrupando líneas sin ruptura de bloque."""
    doc = fitz.open(file_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text() + "\n\n"
    raw_paras = re.split(r"\n\s*\n", all_text)
    paragraphs = []
    for p in raw_paras:
        single = " ".join([ln.strip() for ln in p.splitlines() if ln.strip()])
        if single:
            paragraphs.append(single)
    return paragraphs


def create_collection(client, name, dim):
    """Crea o resetea la colección con la dimensión adecuada."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def store_chunks(paragraphs, client, vocab, idf):
    print(f"🔍 Procesando {len(paragraphs)} párrafos...")
    points = []
    for p in paragraphs:
        vec = tfidf_vector(p, vocab, idf)
        points.append(
            PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": p})
        )
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    resp = client.scroll(collection_name=COLLECTION_NAME, limit=1_000_000)
    
    # Soporta ambos formatos: ScrollResponse con .points, o tupla (points, offset)
    if hasattr(resp, "points"):
        points_list = resp.points
    elif isinstance(resp, tuple):
        points_list = resp[0]
    else:
        # por si acaso
        points_list = resp
    
    print(f"✅ Almacenados. Total en Qdrant: {len(points_list)} puntos.")


def main():
    file_path = "/app/reglamento.pdf"
    if not os.path.exists(file_path):
        print("❌ Archivo no encontrado:", file_path)
        return

    print("📄 Extrayendo párrafos del PDF...")
    paragraphs = extract_paragraphs_from_pdf(file_path)
    print(f"✅ Extraídos {len(paragraphs)} párrafos.")

    # Construir vocabulario e IDF
    vocab = build_vocab(paragraphs)
    idf = build_idf(paragraphs, vocab)
    dim = len(vocab)
    print(f"✏️  Vocabulario de tamaño {dim} palabras.")

    # Conectar y crear colección
    client = QdrantClient(host="qdrant", port=6333)
    create_collection(client, COLLECTION_NAME, dim)

    # Almacenar chunks
    store_chunks(paragraphs, client, vocab, idf)


if __name__ == "__main__":
    main()
