## rag_web

Descripción: Servicio FastAPI (puerto 8001) con un sidecar de Ollama (puerto 11434) para ejecutar modelos locales. Incluye una UI de chat con RAG y un Makefile con atajos de operación.

### Requisitos
- Docker y Docker Compose instalados
- Directorio de trabajo: `rag_web`

### Puertos
- App: `http://localhost:8001/`
- Ollama API: `http://localhost:11434/`
- Qdrant (panel): `http://localhost:6333/dashboard`

---

## Cómo funciona el chat (vista general)
1. Subes un PDF/TXT desde la UI con “Subir e indexar”.
   - Se extraen párrafos y se crean vectores en Qdrant.
2. Enviando una pregunta, puedes escoger:
   - Modo embedding: cómo se vectoriza la pregunta
     - `modelo`: usa el modelo de embeddings de Ollama (`OLLAMA_EMBED_MODEL`)
     - `propio`: usa un embedding local TF‑IDF basado en el corpus subido
   - Modo re‑ranking: cómo se ordenan los pasajes recuperados
     - `modelo`: devuelve el top‑N de Qdrant por similitud del embedding
     - `propio`: ranking híbrido local (embeddings + coincidencia de términos + patrones)
     - `open IA`: re‑ranking con el modelo LLM local (gpt‑oss) sobre el top del embedding

Flujos detallados:
- Embedding = `modelo` → la pregunta se vectoriza con `OLLAMA_EMBED_MODEL` (Ollama) y se consulta Qdrant.
- Embedding = `propio` → se usa TF‑IDF (archivo `src/local_embed.py`) para vectorizar y rankear localmente.
- Re‑ranking = `modelo` → devuelve directamente el top de Qdrant por similitud (sin post‑proceso).
- Re‑ranking = `propio` → aplica ranking híbrido local (archivo `src/server.py`).
- Re‑ranking = `open IA` → gpt‑oss re‑ordena el top preliminar (archivo `src/server.py`).

Indicadores UI:
- Cuando el re‑ranking es `modelo` u `open IA`, verás un mensaje “⏳ Re‑rankeando...” mientras se obtiene la respuesta.

---

## Makefile (atajos esenciales)

Arranque y ciclo de vida
- `make up`: levanta los servicios y espera a que la API responda
- `make down`: detiene y limpia
- `make build` / `make rebuild`: construye o reconstruye imágenes
- `make status` / `make logs` / `make logs-app`: inspección rápida
- `make refresh`: reconstruye y reinicia solo la app (`rag_web`), esperando que esté arriba
- `make refresh-all`: reconstruye y reinicia app + ollama + qdrant

Ingesta y pruebas
- `make ingest`: llama a `POST /ingest` tras comprobar disponibilidad de la API
- `make open`: abre `http://localhost:8001/`

Gestión de modelos (Ollama)
- `make models-list`: muestra modelos populares para descarga
- `make models-menu`: menú interactivo para descargar un modelo
- `make model MODEL=llama3.2:3b`: descarga directa por nombre
- `make models-installed`: lista modelos instalados
- `make models-du` / `make models-prune` / `make models-rm MODEL=...`: espacio y limpieza

Selección de modelo de embeddings
- `make embed-list`: lista sugerida (p.ej., `nomic-embed-text`, `bge-m3`, `all-minilm:latest`)
- `make embed-menu`: selector interactivo
- `make embed-model MODEL=bge-m3`: fija el modelo (descarga si falta) y reinicia servicios
- `make embed-installed` / `make embed-installed-menu`: trabajar con modelos ya instalados
- `make embed-current`: muestra el modelo actual (persistido en `.env`)

Bootstrap “todo en uno”
- `make bootstrap OLLAMA_DATA=/ruta/grande/ollama QDRANT_DATA=/ruta/grande/qdrant OLLAMA_EMBED_MODEL=bge-m3`
  - Prepara carpetas, levanta servicios, descarga el modelo de embeddings y ejecuta ingesta del PDF de ejemplo.

Variables útiles
- `OLLAMA_DATA`: ruta host donde se guardan los modelos (soporta discos grandes)
- `QDRANT_DATA`: ruta host para almacenamiento de Qdrant
- `OLLAMA_EMBED_MODEL`: modelo de embeddings activo (por defecto `nomic-embed-text`)
- `OLLAMA_RERANK_MODEL`: modelo para re‑ranking con LLM (por defecto `gpt-oss:latest`)

---

## Estructura de código
- `src/server.py`: API FastAPI y lógica de RAG, modos de embedding/ranking y endpoints (`/ingest`, `/search`, `/api/upload`, `/api/chat`).
- `src/local_rank.py`: ranking propio (TF‑IDF + coseno) y helpers para leer de Qdrant.
- `src/local_embed.py`: embeddings locales TF‑IDF para consulta y párrafos.
- `templates/chat.html` + `static/app.js` + `static/app.css`: UI del chat.

---

## Ejemplos de uso
1) Levantar todo y abrir UI
```bash
make up
make open
```
2) Descargar y fijar modelo de embeddings, reingestar
```bash
make embed-model MODEL=bge-m3
make ingest
```
3) Seleccionar embedding/ranking en la UI y hacer preguntas (e.g., “¿cuál es la nota aprobatoria?”).

---

## Resolución de problemas
- “No space left on device” al descargar modelos:
  - Usa `OLLAMA_DATA` y `QDRANT_DATA` apuntando a un disco grande y `make up`
  - Limpieza: `make models-prune` / `make models-rm MODEL=...`
- Embeddings 500/404:
  - Asegúrate de usar un modelo que soporte `/api/embeddings` (ej.: `bge-m3`, `nomic-embed-text`)
  - Verifica con: `curl -X POST http://localhost:11434/api/embeddings -d '{"model":"bge-m3","prompt":"hola"}'`
- No se ven cambios en la UI:
  - `make refresh` y luego hard refresh (Ctrl+F5)

---

## Notas
- La ingesta borra y recrea la colección en Qdrant.
- Si cambias el modelo de embeddings, re‑ingesta para regenerar vectores.
- Los modelos instalados se guardan en `OLLAMA_DATA`; los datos vectoriales en `QDRANT_DATA`.
