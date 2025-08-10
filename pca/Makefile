# Makefile para app PCA (FastAPI) - con mensajes de progreso y sin entrar a logs

DC := docker compose
FILE := docker-compose.yml
SERVICE := pca
PORT := 8000
URL := http://localhost:$(PORT)
IMAGE := pca-app:latest

# Atajos
PY := $(DC) -f $(FILE) run --rm $(SERVICE) python

.PHONY: all build start stop restart rebuild status open shell python ps logs logs-pca

all: start

# Construcción de imagen
build:
	@echo "🔧 Construyendo imagen Docker..."; \
	$(DC) -f $(FILE) build; \
	echo "✅ Imagen construida."

# Levantar en background (no adjunta logs). Construye sólo si no existe la imagen
start:
	@echo "🔎 Verificando imagen $(IMAGE)..."; \
	if ! docker image inspect $(IMAGE) >/dev/null 2>&1; then \
	  echo "🔧 Imagen no encontrada, construyendo..."; \
	  $(DC) -f $(FILE) build; \
	else \
	  echo "✅ Imagen existente: $(IMAGE)"; \
	fi; \
	echo "🚀 Iniciando servicio en segundo plano..."; \
	$(DC) -f $(FILE) up -d --remove-orphans; \
	echo "⏳ Verificando disponibilidad en $(URL)..."; \
	i=0; \
	until curl -sSf -o /dev/null $(URL)/ || [ $$i -ge 30 ]; do \
	  i=$$((i+1)); printf "."; sleep 1; \
	done; echo ""; \
	if [ $$i -ge 30 ]; then \
	  echo "❌ El servicio no respondió a tiempo en $(URL). Revisa 'make status'"; exit 1; \
	else \
	  echo "✅ Servicio disponible en $(URL)"; \
	fi

# Detener y eliminar (sin logs interactivos)
stop:
	@echo "🛑 Deteniendo y limpiando contenedores/redes/volúmenes..."; \
	$(DC) -f $(FILE) down --remove-orphans --volumes; \
	echo "✅ Servicios detenidos."

# Reiniciar sin reconstruir
restart:
	@echo "🔁 Reiniciando servicio..."; \
	$(DC) -f $(FILE) down --remove-orphans || true; \
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

# Reconstruir desde cero y levantar
rebuild:
	@echo "🧹 Eliminando contenedores/volúmenes y reconstruyendo sin caché..."; \
	$(DC) -f $(FILE) down --remove-orphans --volumes || true; \
	$(DC) -f $(FILE) build --no-cache; \
	$(MAKE) start

# Estado rápido (contenedor + HTTP)
status:
	@echo "📦 Estado de contenedores:"; \
	$(DC) -f $(FILE) ps | cat; \
	echo "\n🌐 Comprobando endpoint raíz:"; \
	if curl -sS -D - $(URL)/ -o /dev/null | head -n 1; then :; else echo "❌ No hay respuesta"; fi

# Abrir navegador (Linux/Mac)
open:
	@echo "🌍 Abriendo $(URL) en el navegador..."; \
	(if command -v xdg-open >/dev/null 2>&1; then xdg-open $(URL); \
	elif command -v open >/dev/null 2>&1; then open $(URL); \
	else echo "ℹ️  Abre manualmente: $(URL)"; fi) >/dev/null 2>&1 || true

# Shell dentro del contenedor
shell:
	@echo "🐚 Abriendo shell en el contenedor $(SERVICE)..."; \
	$(DC) -f $(FILE) run --rm $(SERVICE) bash

# REPL de Python en contenedor
python:
	@echo "🐍 Abriendo REPL de Python en el contenedor $(SERVICE)..."; \
	$(PY)

# Ver logs (opcional, no se usa por defecto)
logs:
	@echo "📜 Mostrando logs de todos los servicios (Ctrl+C para salir)..."; \
	$(DC) -f $(FILE) logs -f | cat

logs-pca:
	@echo "📜 Mostrando logs de $(SERVICE) (Ctrl+C para salir)..."; \
	$(DC) -f $(FILE) logs -f $(SERVICE) | cat

# Alias
ps: status