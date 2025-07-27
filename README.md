````markdown
# RAG‑MM Project

Este proyecto implementa un mini‑RAG (Retrieval‑Augmented Generation) usando **Qdrant** y **Python** dentro de Docker. A continuación encontrarás los requisitos, pasos de instalación y uso básico.

---

## 📋 Requisitos Previos

Antes de empezar, asegúrate de tener instalados en tu máquina:

- **Git** (para clonar el repositorio)  
- **Docker** (versión ≥ 20.x)  
- **Docker Compose** v2 (normalmente viene con Docker Desktop)  
- **GNU Make**  

> En sistemas basados en Ubuntu/Debian puedes instalarlos con:  
> ```bash
> sudo apt update
> sudo apt install -y git docker.io make
> sudo systemctl enable --now docker
> # Para Docker Compose v2:
> sudo apt install -y docker-compose-plugin
> ```

---

## ⚙️ Estructura del Proyecto

````

rag-mm/
├── app/
│   ├── main.py            # Indexa el PDF y crea los chunks en Qdrant
│   ├── search.py          # Script de consulta de top‑5 párrafos
│   └── requirements.txt   # pymupdf, qdrant-client
├── data/
│   └── reglamento.pdf     # PDF montado en /app/reglamento.pdf
├── Dockerfile             # Imagen base Python 3.10 slim
├── docker-compose.yml     # Servicios: qdrant + app
├── Makefile               # Atajos para build, shell, main, search, logs…
└── README.md              # (Este archivo)

````

---

## 🚀 Primer Arranque

1. **Clona el repositorio**  
   ```bash
   git clone https://tu‑repo‑git/rag‑mm.git
   cd rag‑mm
````

2. **Coloca tu PDF**
   Sitúa tu reglamento en `./data/reglamento-de-estudiantes-universidad-javeriana.pdf`.

3. **Construye y levanta los servicios**

   ```bash
   make build
   ```

   Esto hace:

   * `docker compose up --build -d`

4. **Verifica Qdrant**
   Abre en el navegador:

   ```
   http://localhost:6333/dashboard
   ```

   Deberías ver tu colección `reglamento_estudiantil`.

---

## 🛠️ Uso Diario

| Comando            | Descripción                                                   |
| ------------------ | ------------------------------------------------------------- |
| `make shell`       | Abre un shell interactivo dentro del contenedor `app`         |
| `make python`      | Entra al REPL de Python dentro del contenedor `app`           |
| `make main`        | Indexa el PDF (genera y sube los chunks a Qdrant)             |
| `make search`      | Ejecuta `search.py` para preguntar y recuperar top‑5 párrafos |
| `make logs`        | Muestra logs de todos los servicios                           |
| `make logs-app`    | Muestra solo logs del servicio `app`                          |
| `make logs-qdrant` | Muestra solo logs del servicio `qdrant`                       |
| `make down`        | Detiene y elimina contenedores/redes creadas                  |

---

## 🔧 Configuración Adicional

* **Ruta del PDF**:

  * Montada en `/app/reglamento.pdf` via `docker-compose.yml`.
* **Dimensión de vectores**:

  * Ajusta `VECTOR_DIM` en `app/main.py` si cambias tu método de embedding.
* **Persistencia de datos**:

  * El volumen `qdrant_data` guarda todos los vectores de Qdrant.

---

## 📖 Flujo de Trabajo

1. Cada vez que actualices la lógica de extracción o embeddings, ejecuta:

   ```bash
   make rebuild   # para rebuild + up -d
   make main      # reprocesa y sube los párrafos
   ```
2. Para probar consultas:

   ```bash
   make search
   ```
3. Para detener todo:

   ```bash
   make down
   ```

---

¡Listo! Con esto ya tienes toda la guía de instalación y uso para tu mini‑RAG con Qdrant y Python en Docker. Si surge cualquier duda o quieres añadir más funcionalidades, aquí estoy para ayudarte.
