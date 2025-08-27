DC := docker compose
FILE := docker-compose.yml
SERVICE := rag_web
PORT := 8001
URL := http://localhost:$(PORT)
IMAGE := rag-web:latest
MODELS := llama3.2:3b llama3.1 mistral qwen2.5:3b phi3 neural-chat:7b tinyllama orca-mini:3b gpt-oss gpt-oss:latest 

# Storage paths (override on invocation):
OLLAMA_DATA ?= ./ollama_data
QDRANT_DATA ?= ./qdrant_data
# Embedding model used for vectorization
OLLAMA_EMBED_MODEL ?= nomic-embed-text
EMBED_MODELS := nomic-embed-text bge-m3 all-minilm:latest

.PHONY: all build up down restart rebuild status open shell python logs logs-app ollama-pull ollama-run ollama-ps model models-installed models-list models-menu models-rm models-prune models-du bootstrap ingest embed-model embed-list embed-menu embed-current embed-installed embed-installed-menu embed-installed-model refresh refresh-all

all: up

build:
	@echo "🔧 Construyendo imagen Docker..."; \
	$(DC) -f $(FILE) build; \
	echo "✅ Imagen construida."

up:
	@echo "🔎 Verificando imagen $(IMAGE)..."; \
	if ! docker image inspect $(IMAGE) >/dev/null 2>&1; then \
	  echo "🔧 Imagen no encontrada, construyendo..."; \
	  $(DC) -f $(FILE) build; \
	else \
	  echo "✅ Imagen existente: $(IMAGE)"; \
	fi; \
	echo "🚀 Iniciando servicios..."; \
	$(DC) -f $(FILE) up -d --remove-orphans; \
	echo "⏳ Verificando disponibilidad en $(URL)..."; \
	i=0; \
	until curl -sSf -o /dev/null $(URL)/ || [ $$i -ge 30 ]; do \
	  i=$$((i+1)); printf "."; sleep 1; \
	done; echo ""; \
	if [ $$i -ge 30 ]; then \
	  echo "❌ El servicio no respondió a tiempo en $(URL)."; exit 1; \
	else \
	  echo "✅ Servicio disponible en $(URL)"; \
	fi

down:
	@echo "🛑 Deteniendo y limpiando..."; \
	$(DC) -f $(FILE) down --remove-orphans --volumes; \
	echo "✅ Servicios detenidos."

restart:
	@$(MAKE) down; $(MAKE) up

rebuild:
	@echo "🧹 Limpiando y reconstruyendo sin caché..."; \
	$(DC) -f $(FILE) down --remove-orphans --volumes || true; \
	$(DC) -f $(FILE) build --no-cache; \
	$(MAKE) up

status:
	@echo "📦 Contenedores:"; \
	$(DC) -f $(FILE) ps | cat; \
	echo "\n🌐 Probing endpoint:"; \
	if curl -sS -D - $(URL)/ -o /dev/null | head -n 1; then :; else echo "❌ No hay respuesta"; fi

open:
	@echo "🌍 Abriendo $(URL) en el navegador..."; \
	(if command -v xdg-open >/dev/null 2>&1; then xdg-open $(URL); \
	elif command -v open >/dev/null 2>&1; then open $(URL); \
	else echo "ℹ️  Abre manualmente: $(URL)"; fi) >/dev/null 2>&1 || true

shell:
	@echo "🐚 Shell en contenedor $(SERVICE)..."; \
	$(DC) -f $(FILE) run --rm $(SERVICE) bash

python:
	@echo "🐍 REPL Python en contenedor $(SERVICE)..."; \
	$(DC) -f $(FILE) run --rm $(SERVICE) python

logs:
	@$(DC) -f $(FILE) logs -f | cat

logs-app:
	@$(DC) -f $(FILE) logs -f $(SERVICE) | cat

# Ollama helpers
ollama-pull:
	@echo "🤖 Descargando modelo Ollama: $$MODEL (ej: llama3.1, mistral, qwen)"; \
	if [ -z "$$MODEL" ]; then echo "❌ Usa: make ollama-pull MODEL=llama3.1"; exit 1; fi; \
	$(DC) -f $(FILE) exec -T ollama bash -lc "ollama pull $$MODEL"

ollama-run:
	@echo "🤖 Probando generación con modelo $$MODEL"; \
	if [ -z "$$MODEL" ]; then echo "❌ Usa: make ollama-run MODEL=llama3.1"; exit 1; fi; \
	$(DC) -f $(FILE) exec -T ollama bash -lc "echo '{\"model\": \"'$$MODEL'\", \"prompt\": \"Hola\"}' | curl -s http://localhost:11434/api/generate -d @- | head -n 5"

ollama-ps:
	@$(DC) -f $(FILE) exec -T ollama bash -lc "ollama ps || true"

# Download a model by name (alias)
model:
	@echo "📥 Descargando modelo (MODEL): $$MODEL"; \
	if [ -z "$$MODEL" ]; then echo "❌ Usa: make model MODEL=llama3.2:3b"; exit 1; fi; \
	$(MAKE) ollama-pull MODEL="$$MODEL"

# List installed models in Ollama
models-installed:
	@$(DC) -f $(FILE) exec -T ollama bash -lc "ollama list || true"

# Show curated list of popular models available to pull
models-list:
	@echo "📚 Modelos populares:"; \
	i=1; for m in $(MODELS); do echo "  $$i) $$m"; i=$$((i+1)); done

# Interactive picker: list -> select by number -> pull
models-menu:
	@$(DC) -f $(FILE) up -d ollama >/dev/null 2>&1 || true; \
	echo "📚 Modelos disponibles:"; \
	i=1; for m in $(MODELS); do echo "  $$i) $$m"; i=$$((i+1)); done; \
	read -p "Elige un número y presiona Enter: " n; \
	i=1; sel=""; for m in $(MODELS); do if [ "$$i" = "$$n" ]; then sel="$$m"; fi; i=$$((i+1)); done; \
	if [ -z "$$sel" ]; then echo "❌ Selección inválida"; exit 1; fi; \
	echo "📥 Descargando $$sel ..."; \
	$(DC) -f $(FILE) exec -T ollama bash -lc "ollama pull $$sel"

# Remove a model to free space
models-rm:
	@echo "🗑️  Eliminando modelo (MODEL): $$MODEL"; \
	if [ -z "$$MODEL" ]; then echo "❌ Usa: make models-rm MODEL=llama3.2:3b"; exit 1; fi; \
	$(DC) -f $(FILE) exec -T ollama bash -lc "ollama rm $$MODEL || true"; \
	$(MAKE) models-du

# Prune unused layers in Ollama store
models-prune:
	@$(DC) -f $(FILE) exec -T ollama bash -lc "ollama prune || true"; \
	$(MAKE) models-du

# Show Ollama data directory size inside container
models-du:
	@$(DC) -f $(FILE) exec -T ollama bash -lc "du -sh /root/.ollama 2>/dev/null || true; df -h | head -n 2"

# One-shot: start stack with large disks, pull embed model, and ingest sample PDF
bootstrap:
	@echo "🗂️  Preparando almacenamiento..."; \
	mkdir -p $(OLLAMA_DATA) $(QDRANT_DATA); \
	echo "🚀 Levantando servicios con montajes: OLLAMA_DATA=$(OLLAMA_DATA) QDRANT_DATA=$(QDRANT_DATA)"; \
	OLLAMA_DATA=$(OLLAMA_DATA) QDRANT_DATA=$(QDRANT_DATA) $(DC) -f $(FILE) up -d --remove-orphans; \
	echo "⏳ Esperando a que la API responda en $(URL)/ ..."; \
	i=0; until curl -sSf -o /dev/null $(URL)/ || [ $$i -ge 60 ]; do i=$$((i+1)); printf "."; sleep 1; done; echo ""; \
	if [ $$i -ge 60 ]; then echo "❌ La API no respondió a tiempo"; exit 1; fi; \
	echo "🤖 Descargando modelo de embeddings: $(OLLAMA_EMBED_MODEL)"; \
	$(MAKE) ollama-pull MODEL=$(OLLAMA_EMBED_MODEL); \
	echo "📥 Ejecutando ingesta del PDF de ejemplo..."; \
	if ! curl -sS -X POST $(URL)/ingest -o /dev/stdout; then echo "\n❌ Ingesta falló"; exit 1; fi; \
	echo "\n✅ Bootstrap completado."

# Trigger ingestion explicitly (vectoriza con embeddings)
ingest:
	@echo "⏳ Esperando API en $(URL)/ ..."; \
	i=0; until curl -sSf -o /dev/null $(URL)/ || [ $$i -ge 60 ]; do i=$$((i+1)); printf "."; sleep 1; done; echo ""; \
	if [ $$i -ge 60 ]; then echo "❌ La API no respondió a tiempo"; exit 1; fi; \
	echo "📥 Ejecutando ingesta vía $(URL)/ingest ..."; \
	if ! curl -sS -X POST $(URL)/ingest -o /dev/stdout; then echo "\n❌ Ingesta falló"; exit 1; fi; echo ""

# Show curated embedding models
embed-list:
	@echo "🧠 Modelos de embeddings populares:"; \
	i=1; for m in $(EMBED_MODELS); do echo "  $$i) $$m"; i=$$((i+1)); done

# Set embedding model non-interactively: make embed-model MODEL=bge-m3
embed-model:
	@echo "🧠 Configurando modelo de embeddings: $$MODEL"; \
	if [ -z "$$MODEL" ]; then echo "❌ Usa: make embed-model MODEL=bge-m3"; exit 1; fi; \
	$(MAKE) ollama-pull MODEL="$$MODEL"; \
	# Persistir en .env
	if [ -f .env ]; then \
	  if grep -q '^OLLAMA_EMBED_MODEL=' .env; then \
	    sed -i 's/^OLLAMA_EMBED_MODEL=.*/OLLAMA_EMBED_MODEL='"$$MODEL"'/' .env; \
	  else \
	    echo 'OLLAMA_EMBED_MODEL='"$$MODEL" >> .env; \
	  fi; \
	else \
	  echo 'OLLAMA_EMBED_MODEL='"$$MODEL" > .env; \
	fi; \
	echo "🔁 Reiniciando servicios con OLLAMA_EMBED_MODEL=$$MODEL ..."; \
	OLLAMA_EMBED_MODEL=$$MODEL $(DC) -f $(FILE) up -d --remove-orphans

# Interactive embedding model picker
embed-menu:
	@echo "🧠 Selecciona modelo de embeddings:"; \
	i=1; for m in $(EMBED_MODELS); do echo "  $$i) $$m"; i=$$((i+1)); done; \
	read -p "Número: " n; \
	i=1; sel=""; for m in $(EMBED_MODELS); do if [ "$$i" = "$$n" ]; then sel="$$m"; fi; i=$$((i+1)); done; \
	if [ -z "$$sel" ]; then echo "❌ Selección inválida"; exit 1; fi; \
	$(MAKE) embed-model MODEL="$$sel"

# Show current embedding model
embed-current:
	@echo "📌 Embedding model actual:"; \
	if [ -f .env ] && grep -q '^OLLAMA_EMBED_MODEL=' .env; then grep '^OLLAMA_EMBED_MODEL=' .env; else echo "(no definido en .env, por defecto: $(OLLAMA_EMBED_MODEL))"; fi

# Rebuild y reinicio rápido del servicio de la app tras cambios en código/templates/static
refresh:
	@echo "🔄 Reconstruyendo imagen y reiniciando $(SERVICE)..."; \
	$(DC) -f $(FILE) up -d --no-deps --build --force-recreate $(SERVICE); \
	echo "⏳ Esperando API en $(URL)/ ..."; \
	i=0; until curl -sSf -o /dev/null $(URL)/ || [ $$i -ge 60 ]; do i=$$((i+1)); printf "."; sleep 1; done; echo ""; \
	if [ $$i -ge 60 ]; then echo "❌ La API no respondió a tiempo"; exit 1; else echo "✅ Servicio disponible en $(URL)"; fi

# Rebuild y reinicio de todos los servicios (app/ollama/qdrant)
refresh-all:
	@echo "🔄 Reconstruyendo imágenes y reiniciando todos los servicios..."; \
	$(DC) -f $(FILE) up -d --build --force-recreate; \
	echo "⏳ Esperando API en $(URL)/ ..."; \
	i=0; until curl -sSf -o /dev/null $(URL)/ || [ $$i -ge 60 ]; do i=$$((i+1)); printf "."; sleep 1; done; echo ""; \
	if [ $$i -ge 60 ]; then echo "❌ La API no respondió a tiempo"; exit 1; else echo "✅ Servicios listos"; fi

# Show models installed locally in Ollama
embed-installed:
	@$(DC) -f $(FILE) exec -T ollama bash -lc "ollama list || true"

# Pick from installed models interactively and set as embedding model
embed-installed-menu:
	@echo "🔎 Leyendo modelos instalados..."; \
	list=$$($(DC) -f $(FILE) exec -T ollama bash -lc "ollama list | tail -n +2 | awk '{print $$1}'" | tr -d '\r'); \
	if [ -z "$$list" ]; then echo "❌ No hay modelos instalados. Usa: make models-menu o make model"; exit 1; fi; \
	i=1; echo "🧠 Modelos instalados:"; \
	for m in $$list; do echo "  $$i) $$m"; i=$$((i+1)); done; \
	read -p "Número: " n; \
	i=1; sel=""; for m in $$list; do if [ "$$i" = "$$n" ]; then sel="$$m"; fi; i=$$((i+1)); done; \
	if [ -z "$$sel" ]; then echo "❌ Selección inválida"; exit 1; fi; \
	$(MAKE) embed-installed-model MODEL="$$sel"

# Set embedding model from an installed model name (non-interactive)
embed-installed-model:
	@echo "🧠 Configurando modelo de embeddings desde instalados: $$MODEL"; \
	if [ -z "$$MODEL" ]; then echo "❌ Usa: make embed-installed-model MODEL=<nombre>"; exit 1; fi; \
	# Probar endpoint de embeddings (puede fallar si el modelo no soporta embeddings)
	status=$$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:11434/api/embeddings -H 'Content-Type: application/json' -d '{"model":"'"$$MODEL"'","prompt":"hola"}'); \
	if [ "$$status" != "200" ]; then echo "⚠️  Advertencia: el modelo $$MODEL no parece soportar embeddings (HTTP $$status)."; fi; \
	# Persistir en .env
	if [ -f .env ]; then \
	  if grep -q '^OLLAMA_EMBED_MODEL=' .env; then \
	    sed -i 's/^OLLAMA_EMBED_MODEL=.*/OLLAMA_EMBED_MODEL='"$$MODEL"'/' .env; \
	  else \
	    echo 'OLLAMA_EMBED_MODEL='"$$MODEL" >> .env; \
	  fi; \
	else \
	  echo 'OLLAMA_EMBED_MODEL='"$$MODEL" > .env; \
	fi; \
	echo "🔁 Reiniciando servicios con OLLAMA_EMBED_MODEL=$$MODEL ..."; \
	OLLAMA_EMBED_MODEL=$$MODEL $(DC) -f $(FILE) up -d --remove-orphans

