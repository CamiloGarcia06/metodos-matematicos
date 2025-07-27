# Makefile para proyecto RAG con Qdrant

# Alias a docker compose
DC := docker compose

# Comando base para ejecutar Python en el contenedor 'app'
PY := $(DC) run --rm app python

.PHONY: all build up down rebuild shell python main search logs logs-app logs-qdrant

# Target por defecto
all: build

# 1. Build & levantar en background
build:
	$(DC) up --build -d

# 2. Levantar sin rebuild
up:
	$(DC) up -d

# 3. Parar y eliminar contenedores/redes
down:
	$(DC) down

# 4. Parar y volver a levantar (clean build)
rebuild: down build

# 5. Abrir un shell interactivo en el contenedor
shell:
	$(DC) run --rm app bash

# 6. Entrada al REPL de Python
python:
	$(DC) run --rm app python

# 7. Ejecutar tu script main.py
main:
	$(PY) main.py

# 8. Ejecutar tu script de búsqueda search.py
search:
	$(PY) search.py

# 9. Ver logs de todos los servicios
logs:
	$(DC) logs -f

# 10. Ver solo logs del contenedor 'app'
logs-app:
	$(DC) logs -f app

# 11. Ver solo logs de Qdrant
logs-qdrant:
	$(DC) logs -f qdrant
