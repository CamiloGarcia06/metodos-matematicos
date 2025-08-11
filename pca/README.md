## App Web: Compresión de Imágenes en Color con PCA

Stack:
- Backend: FastAPI (Python)
- Frontend: TailwindCSS (CDN en plantilla)

### Teoría de PCA (resumen)

- Objetivo: encontrar una base ortonormal (componentes principales) que maximice la varianza explicada del conjunto de datos y permita una reconstrucción con menos dimensiones k ≪ d.
- Para una matriz de datos X ∈ R^{n×d} centrada por columnas, la covarianza es C = (1/(n-1)) X^T X. Los vectores propios de C (autovectores) forman las columnas de V y sus autovalores λ_i son proporcionales a la energía (varianza) de cada componente.
- Usando SVD: X = U S V^T. Se cumple que C = V (S^2/(n-1)) V^T, por lo que λ_i = S_i^2/(n-1), y la fracción de varianza explicada es EVR_i = λ_i / Σ_j λ_j.
- Proyección a k componentes: T_k = X V_k. Reconstrucción: \hat{X} = T_k V_k^T = X V_k V_k^T.

En imágenes por canal (matriz H×W): cada fila es un vector de longitud W (columnas/píxeles horizontales). Se centra por columnas, se calcula SVD, se elige k según un umbral de varianza acumulada, y se reconstruye. Se aplica por separado a R, G y B.

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
- `make -C pca debug`: arranca en modo debug con `debugpy` (puerto 5678) y recarga en caliente.
- `make -C pca debug-down`: detiene el modo debug.

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

### Implementación de PCA en este proyecto

- Archivo: `app/image_pca.py`.
- Funciones principales:
  - `choose_k_for_threshold(evr, threshold)`: calcula el mínimo k tal que la varianza acumulada ≥ threshold.
  - `_pca_fit_via_svd(X)`: centra X, calcula SVD y retorna `components` (V^T), `mean` y `explained_variance_ratio`.
  - `pca_compress_channel(channel_uint8, variance_threshold)`: aplica PCA a un canal H×W, elige k por umbral, reconstruye y retorna `(X_hat, k, evr_full)`.
  - `compress_image_rgb(img_rgb, variance_threshold)`: aplica el proceso a R, G y B, apila la reconstrucción y retorna imagen, ks y EVR por canal.

Flujo resumido por canal:
1) Convertir a float32 y centrar por columnas.
2) SVD: X_c = U S V^T.
3) EVR_i = (S_i^2/(H-1)) / sum_j (S_j^2/(H-1)).
4) Elegir k con `choose_k_for_threshold`.
5) Proyección: T_k = X_c V_k.
6) Reconstrucción: \hat{X} = T_k V_k^T + mean.
7) Redondear y saturar [0, 255].

### Depuración (VS Code)

- Usa `make -C pca debug` y la configuración `.vscode/launch.json` (Attach to PCA) para adjuntar el depurador a `localhost:5678`. El contenedor espera al cliente y tiene `--reload` activo.