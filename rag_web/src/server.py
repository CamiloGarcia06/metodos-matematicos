from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import fitz  # PyMuPDF
import re
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
from typing import List, Set
import httpx
import unicodedata

app = FastAPI(title="RAG Web API")

# Mount static and templates
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
templates = Jinja2Templates(directory="/app/templates")

COLLECTION_NAME = "reglamento_estudiantil"

async def embed_text(text: str) -> List[float]:
    ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    async with httpx.AsyncClient(timeout=120) as client_http:
        r = await client_http.post(f"{ollama_host}/api/embeddings", json={"model": embed_model, "prompt": text})
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding") or (data.get("data", [{}])[0].get("embedding") if isinstance(data.get("data"), list) else None)
        if not emb:
            raise RuntimeError("No embedding returned by Ollama")
        return emb


async def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for t in texts:
        vectors.append(await embed_text(t))
    return vectors


def _normalize(text: str) -> str:
    # Minúsculas y sin tildes para comparaciones robustas
    nf = unicodedata.normalize("NFD", text)
    no_accents = "".join(ch for ch in nf if unicodedata.category(ch) != "Mn")
    return no_accents.lower()


def _tokens(s: str) -> Set[str]:
    return set(re.findall(r"\w+", _normalize(s)))


def _pattern_bonus(text: str, query: str) -> float:
    t = _normalize(text)
    q = _normalize(query)
    # Palabras clave relevantes para calificaciones
    keywords = [
        "nota", "aprob", "promedio", "minima", "mínima", "calificacion", "calificación",
        "papa", "ponderado", "aprobatoria", "aprobatorio"
    ]
    bonus = 0.0
    if any(k in t for k in keywords):
        bonus += 0.6
    # Números típicos (3.0 / 3,0)
    if re.search(r"\b3[\.,]0\b", t):
        bonus += 0.6
    # Coincidencia de bigramas de la consulta
    qtoks = list(_tokens(q))
    bigrams = {f"{qtoks[i]} {qtoks[i+1]}" for i in range(len(qtoks)-1)} if len(qtoks) > 1 else set()
    if bigrams:
        for bg in bigrams:
            if bg in t:
                bonus += 0.3
                break
    return min(bonus, 1.2)


@app.get("/")
async def root():
    return JSONResponse({
        "message": "RAG Web running",
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://ollama:11434"),
    })


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return templates.TemplateResponse("chat.html", {"request": {}})


class IngestResponse(BaseModel):
    total_paragraphs: int
    vector_size: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    file_path = "/app/reglamento.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Archivo PDF no encontrado en /app/reglamento.pdf")

    doc = fitz.open(file_path)
    all_text = ""
    for page in doc:
        all_text += page.get_text() + "\n\n"
    raw_paras = re.split(r"\n\s*\n", all_text)
    paragraphs: List[str] = []
    for p in raw_paras:
        single = " ".join([ln.strip() for ln in p.splitlines() if ln.strip()])
        if single:
            paragraphs.append(single)
    if not paragraphs:
        raise HTTPException(status_code=400, detail="No se extrajeron párrafos del PDF")

    # Generar embeddings con Ollama
    vectors = await embed_texts(paragraphs)
    if not vectors:
        raise HTTPException(status_code=500, detail="No se generaron embeddings")
    dim = len(vectors[0])

    client = QdrantClient(host=os.environ.get("QDRANT_HOST", "qdrant"), port=int(os.environ.get("QDRANT_PORT", "6333")))
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    points: List[PointStruct] = []
    for p, vec in zip(paragraphs, vectors):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": p}))
        if len(points) >= 1024:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    return IngestResponse(total_paragraphs=len(paragraphs), vector_size=dim)


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


@app.post("/search")
async def search(req: SearchRequest):
    client = QdrantClient(host=os.environ.get("QDRANT_HOST", "qdrant"), port=int(os.environ.get("QDRANT_PORT", "6333")))
    if not client.collection_exists(COLLECTION_NAME):
        raise HTTPException(status_code=404, detail="Colección no encontrada. Ejecuta /ingest primero.")

    # Embedding de la consulta
    q_vec = await embed_text(req.query)
    raw_hits = client.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=50, with_payload=True)

    # Re-ranking híbrido (embeddings + coincidencia + patrones de dominio)
    q_tokens = _tokens(req.query)
    if not raw_hits:
        return {"results": []}
    scores = [float(h.score) for h in raw_hits]
    s_min, s_max = min(scores), max(scores)

    def _norm(x: float) -> float:
        return 0.0 if s_max == s_min else (x - s_min) / (s_max - s_min)

    ranked = []
    seen = set()
    for h in raw_hits:
        txt = h.payload.get("text", "")
        if txt in seen:
            continue
        seen.add(txt)
        tks = _tokens(txt)
        overlap = 0.0 if not q_tokens else len(q_tokens & tks) / max(1, len(q_tokens))
        patt = _pattern_bonus(txt, req.query)
        # Pesos: 0.6 embeddings + 0.25 token overlap + 0.15 patrón
        hybrid = 0.6 * _norm(float(h.score)) + 0.25 * overlap + 0.15 * patt
        ranked.append({"text": txt, "score": float(h.score), "hybrid": hybrid})

    ranked.sort(key=lambda x: x["hybrid"], reverse=True)
    return {"results": [{"score": r["score"], "text": r["text"]} for r in ranked[: max(1, min(req.limit, 20))]]}


class UploadResponse(BaseModel):
    total_paragraphs: int
    vector_size: int


@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(file: UploadFile = File(...)):
    # Guardar temporalmente
    tmp_path = "/app/media/uploaded.pdf" if file.filename.lower().endswith(".pdf") else "/app/media/uploaded.txt"
    os.makedirs("/app/media", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # Si es TXT, convertir a párrafos simples
    if tmp_path.endswith(".txt"):
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    else:
        doc = fitz.open(tmp_path)
        all_text = ""
        for page in doc:
            all_text += page.get_text() + "\n\n"
        paragraphs = [" ".join([ln.strip() for ln in p.splitlines() if ln.strip()]) for p in re.split(r"\n\s*\n", all_text) if p.strip()]

    if not paragraphs:
        raise HTTPException(status_code=400, detail="No se extrajeron párrafos del archivo")

    vectors = await embed_texts(paragraphs)
    if not vectors:
        raise HTTPException(status_code=500, detail="No se generaron embeddings")
    dim = len(vectors[0])

    client = QdrantClient(host=os.environ.get("QDRANT_HOST", "qdrant"), port=int(os.environ.get("QDRANT_PORT", "6333")))
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    points: List[PointStruct] = []
    for p, vec in zip(paragraphs, vectors):
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"text": p}))
        if len(points) >= 1024:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    return UploadResponse(total_paragraphs=len(paragraphs), vector_size=dim)


class ChatRequest(BaseModel):
    prompt: str
    limit: int = 5
    model: str | None = None


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    client = QdrantClient(host=os.environ.get("QDRANT_HOST", "qdrant"), port=int(os.environ.get("QDRANT_PORT", "6333")))
    if not client.collection_exists(COLLECTION_NAME):
        raise HTTPException(status_code=404, detail="Colección no encontrada. Sube un archivo o ejecuta /ingest.")

    # Buscar contextos mediante embedding de la pregunta
    q_vec = await embed_text(req.prompt)
    raw_hits = client.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=50, with_payload=True)

    # Re-ranking híbrido (embeddings + coincidencia + patrones de dominio)
    q_tokens = _tokens(req.prompt)
    results = []
    if raw_hits:
        scores = [float(h.score) for h in raw_hits]
        s_min, s_max = min(scores), max(scores)
        def _norm(x: float) -> float:
            return 0.0 if s_max == s_min else (x - s_min) / (s_max - s_min)
        seen = set()
        for h in raw_hits:
            txt = h.payload.get("text", "")
            if txt in seen:
                continue
            seen.add(txt)
            tks = _tokens(txt)
            overlap = 0.0 if not q_tokens else len(q_tokens & tks) / max(1, len(q_tokens))
            patt = _pattern_bonus(txt, req.prompt)
            hybrid = 0.6 * _norm(float(h.score)) + 0.25 * overlap + 0.15 * patt
            results.append({"text": txt, "score": float(h.score), "hybrid": hybrid})
        results.sort(key=lambda x: x["hybrid"], reverse=True)
        results = [{"score": r["score"], "text": r["text"]} for r in results[: max(1, min(req.limit, 10))]]

    # Responder con el TOP N de párrafos más relacionados
    answer_lines = []
    for idx, item in enumerate(results, 1):
        answer_lines.append(f"[{idx}] score={item['score']:.4f}\n{item['text']}")
    answer = "\n\n".join(answer_lines) if answer_lines else "(sin resultados)"

    return {"answer": answer, "contexts": [r["text"] for r in results]}
