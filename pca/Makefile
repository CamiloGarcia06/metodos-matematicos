# Makefile para app PCA (FastAPI) - con mensajes de progreso y sin entrar a logs

DC := docker compose
FILE := docker-compose.yml
OVERRIDE_TUNNEL := docker-compose.tunnel.yml
SERVICE := pca
PORT := 8000
URL := http://localhost:$(PORT)
IMAGE := pca-app:latest

# Atajos
PY := $(DC) -f $(FILE) run --rm $(SERVICE) python

.PHONY: all build start stop restart rebuild status open shell python ps logs logs-pca tunnel-up tunnel-down quick-tunnel-up quick-tunnel-down quick-url debug debug-down

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

# Exponer vía Cloudflare Tunnel (autenticado con token)
# Requiere que CLOUDFLARE_TUNNEL_TOKEN esté exportado en el entorno
# y que el túnel esté configurado en la UI de Cloudflare apuntando al servicio (hostname -> http://pca:8000)
tunnel-up: start
	@if [ -z "$$CLOUDFLARE_TUNNEL_TOKEN" ]; then \
	  echo "❌ Falta CLOUDFLARE_TUNNEL_TOKEN en el entorno"; exit 1; \
	fi; \
	echo "🌐 Iniciando Cloudflare Tunnel..."; \
	$(DC) -f $(FILE) -f $(OVERRIDE_TUNNEL) up -d cloudflared; \
	echo "✅ Tunnel iniciado. Verifica en Cloudflare el subdominio asignado."

# Detener Cloudflare Tunnel
tunnel-down:
	@echo "🛑 Deteniendo Cloudflare Tunnel..."; \
	$(DC) -f $(FILE) -f $(OVERRIDE_TUNNEL) down; \
	echo "✅ Tunnel detenido."

# Quick Tunnel (subdominio aleatorio, no requiere cuenta/token). Útil para demos.
quick-tunnel-up: start
	@echo "🌐 Iniciando Cloudflare Quick Tunnel (subdominio aleatorio)..."; \
	$(DC) -f $(FILE) -f docker-compose.quick.yml up -d cloudflared; \
	echo "⏳ Obteniendo URL pública..."; \
	url=""; tries=0; \
	while [ -z "$$url" ] && [ $$tries -lt 30 ]; do \
	  sleep 1; tries=$$((tries+1)); \
	  url=$$($(DC) -f $(FILE) -f docker-compose.quick.yml logs --no-color cloudflared 2>/dev/null | grep -Eo 'https://[^ ]*trycloudflare.com' | tail -1); \
	  if [ -z "$$url" ]; then \
	    url=$$(docker logs pca-tunnel --since=30m 2>/dev/null | grep -Eo 'https://[^ ]*trycloudflare.com' | tail -1); \
	  fi; \
	done; \
	if [ -n "$$url" ]; then \
	  echo "🌍 URL pública: $$url"; \
	else \
	  echo "ℹ️  No se detectó la URL todavía. Ejecuta: make quick-url"; \
	fi

quick-tunnel-down:
	@echo "🛑 Deteniendo Quick Tunnel..."; \
	$(DC) -f $(FILE) -f docker-compose.quick.yml down; \
	echo "✅ Quick Tunnel detenido."

# Mostrar la URL pública del Quick Tunnel
quick-url:
	@echo "🔎 Consultando URL del Quick Tunnel..."; \
	url=$$($(DC) -f $(FILE) -f docker-compose.quick.yml logs --no-color cloudflared 2>/dev/null | grep -Eo 'https://[^ ]*trycloudflare.com' | tail -1); \
	if [ -z "$$url" ]; then \
	  url=$$(docker logs pca-tunnel --since=60m 2>/dev/null | grep -Eo 'https://[^ ]*trycloudflare.com' | tail -1); \
	fi; \
	if [ -n "$$url" ]; then \
	  echo "🌍 URL pública: $$url"; \
	else \
	  echo "❌ No se encontró URL en logs. Últimas líneas de cloudflared:"; \
	  $(DC) -f $(FILE) -f docker-compose.quick.yml logs --no-color --tail=80 cloudflared || true; \
	fi

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

# Modo debug: levanta con debugpy exponiendo el puerto 5678 y código montado
debug:
	@echo "🪲 Iniciando servicio en modo DEBUG (debugpy en 0.0.0.0:5678)..."; \
	# Detener si hay servicio normal activo para liberar puertos\n; \
	$(DC) -f $(FILE) down --remove-orphans >/dev/null 2>&1 || true; \
	if ! docker image inspect $(IMAGE) >/dev/null 2>&1; then \
	  echo "🔧 Imagen no encontrada, construyendo..."; \
	  $(DC) -f $(FILE) build; \
	else \
	  echo "✅ Imagen existente: $(IMAGE)"; \
	fi; \
	$(DC) -f $(FILE) -f docker-compose.debug.yml up --remove-orphans | cat

debug-down:
	@echo "🛑 Deteniendo DEBUG y limpiando..."; \
	$(DC) -f $(FILE) -f docker-compose.debug.yml down --remove-orphans; \
	echo "✅ Debug detenido."