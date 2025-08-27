## rag_web

Descripción: Servicio FastAPI (puerto 8001) con un sidecar de Ollama (puerto 11434) para ejecutar modelos locales. Incluye Makefile con atajos para construir, levantar, y gestionar modelos en Ollama.

### Requisitos
- Docker y Docker Compose instalados
- Directorio de trabajo: `rag_web`

### Puertos
- App: `http://localhost:8001/`
- Ollama API: `http://localhost:11434/`

### Comandos principales del Makefile

- **build**: construye la imagen de la app.
```bash
make build
```

- **up**: levanta servicios (construye si la imagen no existe) y espera a que la app responda.
```bash
make up
```

- **down**: detiene y elimina contenedores, redes y volúmenes anónimos.
```bash
make down
```

- **rebuild**: limpia y reconstruye sin caché, luego levanta.
```bash
make rebuild
```

- **status**: muestra estado de contenedores y prueba el endpoint raíz.
```bash
make status
```

- **logs**: sigue logs de todos los servicios.  |  **logs-app**: sólo de la app.
```bash
make logs
make logs-app
```

- **shell**: abre una shell dentro del contenedor de la app.  |  **python**: REPL de Python en la app.
```bash
make shell
make python
```

- **open**: intenta abrir `http://localhost:8001/` en el navegador.
```bash
make open
```

### Probar la app
Con los servicios arriba (`make up`):
```bash
curl -s http://localhost:8001/
# {"message":"RAG Web running","ollama_host":"http://ollama:11434"}
```

### Gestión de modelos con Ollama
Los siguientes atajos ejecutan comandos dentro del contenedor `ollama`:

- **ollama-pull**: descarga un modelo. Reemplaza `MODEL` por el nombre (ej: `llama3.1`, `phi3`, `qwen2.5:3b`, `mistral`).
```bash
make ollama-pull MODEL=llama3.1
```

- **ollama-run**: hace una generación rápida con el modelo indicado.
```bash
make ollama-run MODEL=llama3.1
```

- **ollama-ps**: lista modelos/estados en Ollama.
```bash
make ollama-ps
```

Notas:
- Si el pull de un modelo grande falla por red, vuelve a ejecutar el comando o elige un modelo más pequeño (ej: `phi3` o `llama3.2:3b`).
- El volumen `ollama_data` persiste los modelos descargados entre reinicios.

### Variables y entorno
- La app expone `OLLAMA_HOST` apuntando a `http://ollama:11434` (configurable en `docker-compose.yml`).
- Para cambiar puertos, edita los mapeos en `docker-compose.yml` y ajusta los comandos si fuera necesario.
