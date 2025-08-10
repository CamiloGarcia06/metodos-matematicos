## App Web: Compresión de Imágenes en Color con PCA

Stack:
- Backend: FastAPI (Python)
- Frontend: TailwindCSS (CDN en plantilla)

### Estructura
- `app/server.py`: servidor FastAPI y endpoints
- `app/image_pca.py`: utilidades PCA por canal
- `templates/index.html`: formulario y resultados
- `static/`, `media/`: estáticos y (si se usan) archivos generados

### Requisitos
- Docker y Docker Compose

### Levantar la app (con mensajes de progreso, sin adjuntar logs)

```bash
make -C pca start
# Abre: http://localhost:8000
```

El Makefile imprime el estado de cada fase (build, up, verificación HTTP) y nunca entra a los logs automáticamente.

### Comandos del Makefile
- `make -C pca build`: construye la imagen Docker.
- `make -C pca start`: levanta en background y valida que `http://localhost:8000` responda (timeout ~30s).
- `make -C pca stop`: detiene y elimina contenedores/redes/volúmenes del proyecto.
- `make -C pca restart`: reinicia en background y valida el endpoint.
- `make -C pca rebuild`: limpia, reconstruye sin caché y levanta validando el endpoint.
- `make -C pca status` (alias `ps`): muestra estado de contenedores y hace un HEAD/GET al endpoint raíz.
- `make -C pca open`: intenta abrir el navegador en `http://localhost:8000` (Linux/Mac).
- `make -C pca shell`: shell interactiva dentro del contenedor.
- `make -C pca python`: REPL de Python dentro del contenedor.
- `make -C pca logs` / `logs-pca`: ver logs solo si los necesitas (no se usan por defecto).

### Exponer públicamente

Opción A) Cloudflare Tunnel autenticado (dominio propio)
- Configura túnel en Cloudflare (Zero Trust → Tunnels) y token.
- `export CLOUDFLARE_TUNNEL_TOKEN=tu_token`
- `make -C pca tunnel-up`
- Accede al subdominio configurado.

Opción B) Cloudflare Quick Tunnel (subdominio aleatorio)
- No requiere cuenta ni token.
- `make -C pca quick-tunnel-up`
- Luego mira logs para ver la URL pública generada:
  - `docker compose -f pca/docker-compose.yml -f pca/docker-compose.quick.yml logs -f cloudflared`
- Para detener: `make -C pca quick-tunnel-down`

Notas de seguridad:
- Quick Tunnel es para demos/temporal. Para producción usa túneles autenticados, WAF/reglas y si aplica auth básica o bearer.