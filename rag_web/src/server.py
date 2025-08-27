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
import json
from .local_rank import rank_qdrant_by_tfidf_cosine
from .local_embed import embed_query_tfidf

app = FastAPI(title="RAG Web API")

# Mount static and templates
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
templates = Jinja2Templates(directory="/app/templates")

COLLECTION_NAME = "reglamento_estudiantil"

async def embed_text(text: str) -> List[float]:
    """Obtener el embedding de una cadena usando el modelo configurado en Ollama.

    Parametros:
        text: Texto de entrada (pregunta o pasaje) a vectorizar.

    Retorna:
        Lista de floats que representa el vector de embedding.

    Lanza:
        httpx.HTTPStatusError si el endpoint responde != 2xx
        RuntimeError si la respuesta no contiene un vector.
    """
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
    """Vectorizar en serie una lista de textos con Ollama.

    Parametros:
        texts: Lista de textos a vectorizar.

    Retorna:
        Lista de vectores (uno por texto).
    """
    vectors: List[List[float]] = []
    for t in texts:
        vectors.append(await embed_text(t))
    return vectors


def _normalize(text: str) -> str:
    """Normalizar texto: minúsculas y sin tildes/diacríticos."""
    # Minúsculas y sin tildes para comparaciones robustas
    nf = unicodedata.normalize("NFD", text)
    no_accents = "".join(ch for ch in nf if unicodedata.category(ch) != "Mn")
    return no_accents.lower()


def _tokens(s: str) -> Set[str]:
    """Tokenizar un string en palabras (\w+) tras normalización."""
    return set(re.findall(r"\w+", _normalize(s)))


def _pattern_bonus(text: str, query: str) -> float:
    """Heurística de bonus para patrones de dominio (notas/calificaciones).

    Aumenta el score si el pasaje contiene términos clave o números como 3.0/3,0.
    """
    t = _normalize(text)
    q = _normalize(query)
    # Palabras clave relevantes para calificaciones
    keywords = [
        "nota", "aprob", "promedio", "minima", "mínima", "calificacion", "calificación",
        "papa", "ponderado", "aprobatoria", "aprobatorio"
    ]
    bonus = 0.0
    if any(k in t for k in keywords):
        bonus += 0.8
    # Números típicos (3.0 / 3,0)
    if re.search(r"\b3[\.,]0\b", t):
        bonus += 0.8
    # Texto en palabras (tres punto cero)
    if "tres punto cero" in t or "tres con cero" in t:
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


def _is_strong_grade_match(text: str) -> bool:
    """Determinar si un pasaje es una coincidencia fuerte de nota aprobatoria."""
    t = _normalize(text)
    has_number = bool(re.search(r"\b3[\.,]0\b", t)) or ("tres punto cero" in t or "tres con cero" in t)
    has_keyword = any(k in t for k in ["calificacion", "calificación", "nota", "aprob", "aprobatoria", "aprobatorio"])
    return has_number and has_keyword


async def rerank_with_llm(query: str, passages: List[str], top_k: int, base_scores: List[float] | None = None) -> List[int]:
    """Re‑rank de pasajes usando el LLM (gpt-oss) vía Ollama.

    Parametros:
        query: Pregunta del usuario.
        passages: Lista de pasajes (texto) a ordenar.
        top_k: Número máximo de índices a devolver.
        base_scores: (Opcional) puntajes de embedding para mostrar como pista.

    Retorna:
        Lista de índices (0‑basados) que define el nuevo orden sugerido.
    """
    if not passages:
        return []
    model = os.environ.get("OLLAMA_RERANK_MODEL", "gpt-oss:latest")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://ollama:11434")

    numbered_lines = []
    if base_scores and len(base_scores) == len(passages):
        for i, (p, s) in enumerate(zip(passages, base_scores)):
            numbered_lines.append(f"{i}. [emb_score={s:.4f}] {p}")
    else:
        for i, p in enumerate(passages):
            numbered_lines.append(f"{i}. {p}")
    numbered = "\n".join(numbered_lines)
    prompt = (
        "Eres un re-ranker. Dada la pregunta y los pasajes provenientes de una búsqueda por embeddings, "
        "reordénalos mejorando el ranking. Respeta el contenido y favorece pasajes que respondan directamente. "
        "Devuelve un JSON con el formato: [{\"index\": <int>, \"score\": <float>}], índices 0-basados. "
        "Considera los emb_score como una pista inicial, pero puedes cambiar el orden si el contenido lo justifica. "
        "Puntúa score en [0,1].\n\n"
        f"Pregunta: {query}\n\n"
        f"Pasajes:\n{numbered}\n\n"
        f"Devuelve sólo JSON, sin texto adicional. Top {top_k}."
    )

    try:
        async with httpx.AsyncClient(timeout=120) as client_http:
            r = await client_http.post(
                f"{ollama_host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            r.raise_for_status()
            data = r.json()
            text = data.get("response", "")
            # Extraer JSON
            match = re.search(r"\[.*\]", text, re.DOTALL)
            payload = json.loads(match.group(0) if match else text)
            scored = [
                (int(item.get("index", -1)), float(item.get("score", 0.0)))
                for item in payload
                if isinstance(item, dict) and isinstance(item.get("index", None), (int, float))
            ]
            scored = [(i, s) for (i, s) in scored if 0 <= i < len(passages)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [i for i, _ in scored[: top_k]]
    except Exception:
        return []


def _is_heading(text: str) -> bool:
    """Filtrar encabezados/secciones vacías o demasiado cortas."""
    t = _normalize(text).strip()
    if not t:
        return True
    # Muy corto o termina en ':' suele ser encabezado
    if len(t) < 24 or t.endswith(":"):
        return True
    # Alto ratio de mayúsculas (en original) sugiere título
    upper = sum(1 for ch in text if ch.isupper())
    letters = sum(1 for ch in text if ch.isalpha())
    if letters > 0 and upper / letters > 0.6:
        return True
    # Consta sólo de números romanos/una palabra
    if re.match(r"^(i|v|x|l|c|d|m)+\.?$", t.replace(".", "")):
        return True
    tokens = _tokens(text)
    return len(tokens) < 8


@app.get("/")
async def root():
    """Endpoint raíz: estado básico de la API."""
    return JSONResponse({
        "message": "RAG Web running",
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://ollama:11434"),
    })


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Servir la UI de chat."""
    return templates.TemplateResponse("chat.html", {"request": {}})


class IngestResponse(BaseModel):
    total_paragraphs: int
    vector_size: int


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """Ingerir el PDF de ejemplo en `/app/reglamento.pdf`.

    Extrae párrafos, calcula embeddings con Ollama y (re)crea la colección en Qdrant.
    """
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
    """Buscar pasajes relevantes para una consulta.

    Por defecto usa embedding del modelo + ranking según el modo seleccionado.
    """
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
        # Filtrar encabezados/secciones vacías o cortas
        if _is_heading(txt):
            continue
        tks = _tokens(txt)
        overlap = 0.0 if not q_tokens else len(q_tokens & tks) / max(1, len(q_tokens))
        patt = _pattern_bonus(txt, req.query)
        # Pesos: 0.6 embeddings + 0.25 token overlap + 0.15 patrón
        hybrid = 0.6 * _norm(float(h.score)) + 0.25 * overlap + 0.15 * patt
        strong = 1.0 if _is_strong_grade_match(txt) else 0.0
        hybrid += 0.6 * strong
        ranked.append({"text": txt, "score": float(h.score), "hybrid": hybrid, "strong": strong})

    # Primer filtro por ranking híbrido, luego re-ranking con LLM si está disponible
    ranked.sort(key=lambda x: x["hybrid"], reverse=True)
    prelim = ranked[: max(1, min(20, len(ranked)))]
    # Forzar que pasajes con match fuerte queden en el tope
    prelim.sort(key=lambda x: (x.get("strong", 0.0), x["hybrid"]), reverse=True)
    prelim_texts = [r["text"] for r in prelim]
    order = await rerank_with_llm(req.query, prelim_texts, top_k=max(1, min(req.limit, 20)))
    if order:
        picked = []
        seen_idx = set()
        for i in order:
            if 0 <= i < len(prelim) and i not in seen_idx:
                picked.append(prelim[i])
                seen_idx.add(i)
                if len(picked) >= req.limit:
                    break
        if len(picked) < req.limit:
            for j, item in enumerate(prelim):
                if j not in seen_idx:
                    picked.append(item)
                    if len(picked) >= req.limit:
                        break
        final = picked
    else:
        final = prelim[: max(1, min(req.limit, len(prelim)))]
    return {"results": [{"score": r["score"], "text": r["text"]} for r in final]}


class UploadResponse(BaseModel):
    total_paragraphs: int
    vector_size: int


@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(file: UploadFile = File(...)):
    """Subir e indexar un archivo PDF/TXT desde la UI.

    El archivo se guarda temporalmente, se extraen párrafos y se actualiza Qdrant.
    """
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
    mode: str | None = None  # "modelo" | "propio"
    embedMode: str | None = None  # "modelo" | "propio"


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    """Chat RAG: devuelve top‑N pasajes relevantes según los modos configurados.

    Modos:
      - embedMode: `modelo` (Ollama) | `propio` (TF‑IDF local)
      - mode (ranking): `modelo` | `propio` | `open IA`
    """
    client = QdrantClient(host=os.environ.get("QDRANT_HOST", "qdrant"), port=int(os.environ.get("QDRANT_PORT", "6333")))
    if not client.collection_exists(COLLECTION_NAME):
        raise HTTPException(status_code=404, detail="Colección no encontrada. Sube un archivo o ejecuta /ingest.")

    # Modo propio de ranking: usar rankeo TF‑IDF + coseno desde local_rank
    if (req.mode or "modelo").lower() == "propio":
        ranked = rank_qdrant_by_tfidf_cosine(client, COLLECTION_NAME, req.prompt, top_k=max(1, min(req.limit, 10)))
        answer_lines = []
        for idx, (score, text) in enumerate(ranked, 1):
            answer_lines.append(f"[{idx}] score={score:.4f}\n{text}")
        answer = "\n\n".join(answer_lines) if ranked else "(sin resultados)"
        return {"answer": answer, "contexts": [t for _, t in ranked]}

    # Selección de embedding: modelo (Ollama) vs propio (TF‑IDF)
    if (req.embedMode or "modelo").lower() == "propio":
        # Construir embedding TF‑IDF de la consulta usando el corpus de Qdrant
        resp = client.scroll(collection_name=COLLECTION_NAME, with_payload=True, limit=1_000_000)
        points_list = resp.points if hasattr(resp, "points") else (resp[0] if isinstance(resp, tuple) else resp)
        paragraphs = [pt.payload.get("text", "") for pt in points_list if pt.payload and pt.payload.get("text")]
        q_vec = embed_query_tfidf(req.prompt, paragraphs)
        # Búsqueda coseno manual: calcularemos similitud ya dentro del rank propio, así que aquí usaremos search aproximado con el vector TF‑IDF
        # Como Qdrant tiene dimensión de embeddings del modelo por defecto, no coincide con TF‑IDF; por simplicidad haremos rankeo local completo cuando embed propio.
        ranked = rank_qdrant_by_tfidf_cosine(client, COLLECTION_NAME, req.prompt, top_k=max(1, min(req.limit, 10)))
        answer_lines = []
        for idx, (score, text) in enumerate(ranked, 1):
            answer_lines.append(f"[{idx}] score={score:.4f}\n{text}")
        answer = "\n\n".join(answer_lines) if ranked else "(sin resultados)"
        return {"answer": answer, "contexts": [t for _, t in ranked]}
    else:
        # Caso por defecto (modelo): buscar por embedding de la pregunta con Ollama
        q_vec = await embed_text(req.prompt)
    raw_hits = client.search(collection_name=COLLECTION_NAME, query_vector=q_vec, limit=50, with_payload=True)

    # Modo ranking "modelo": usa exclusivamente similitud del embedding del modelo (Qdrant)
    rank_mode = (req.mode or "modelo").lower()
    if rank_mode == "modelo":
        results = []
        seen = set()
        for h in raw_hits:
            txt = h.payload.get("text", "")
            if not txt or txt in seen:
                continue
            seen.add(txt)
            results.append({"text": txt, "score": float(h.score)})
            if len(results) >= req.limit:
                break
    elif rank_mode == "open ia":
        # Re‑rank con LLM (gpt‑oss) sobre el top del embedding del modelo
        prelim = []
        seen = set()
        for h in raw_hits:
            txt = h.payload.get("text", "")
            if not txt or txt in seen:
                continue
            seen.add(txt)
            prelim.append({"text": txt, "score": float(h.score)})
            if len(prelim) >= max(1, min(20, req.limit * 3)):
                break
        prelim_texts = [r["text"] for r in prelim]
        prelim_scores = [r["score"] for r in prelim]
        order = await rerank_with_llm(req.prompt, prelim_texts, top_k=max(1, min(req.limit, 20)), base_scores=prelim_scores)
        if order:
            picked = []
            used = set()
            for i in order:
                if 0 <= i < len(prelim) and i not in used:
                    picked.append(prelim[i])
                    used.add(i)
                    if len(picked) >= req.limit:
                        break
            if len(picked) < req.limit:
                for j, item in enumerate(prelim):
                    if j not in used:
                        picked.append(item)
                        if len(picked) >= req.limit:
                            break
            results = picked
        else:
            results = prelim[: max(1, min(req.limit, len(prelim)))]
    else:
        # Re-ranking local (híbrido) cuando el modo no es "modelo"
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
                if _is_heading(txt):
                    continue
                tks = _tokens(txt)
                overlap = 0.0 if not q_tokens else len(q_tokens & tks) / max(1, len(q_tokens))
                patt = _pattern_bonus(txt, req.prompt)
                hybrid = 0.6 * _norm(float(h.score)) + 0.25 * overlap + 0.15 * patt
                strong = 1.0 if _is_strong_grade_match(txt) else 0.0
                hybrid += 0.6 * strong
                results.append({"text": txt, "score": float(h.score), "hybrid": hybrid, "strong": strong})
            results.sort(key=lambda x: x["hybrid"], reverse=True)
            results = results[: max(1, min(req.limit, len(results)))]

    # Responder con el TOP N de párrafos más relacionados
    answer_lines = []
    for idx, item in enumerate(results, 1):
        answer_lines.append(f"[{idx}] score={item['score']:.4f}\n{item['text']}")
    answer = "\n\n".join(answer_lines) if answer_lines else "(sin resultados)"

    return {"answer": answer, "contexts": [r["text"] for r in results]}
