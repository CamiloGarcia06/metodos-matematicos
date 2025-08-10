import base64
from io import BytesIO
from typing import Dict
import os
import uuid

import numpy as np
import cv2
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .image_pca import compress_image_rgb

app = FastAPI(title="Compresión de Imágenes con PCA")

# Paths
MEDIA_DIR = "media"

# Static and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def decode_image_to_rgb(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen subida")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def encode_image_to_data_uri(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.png', img_bgr)
    if not ok:
        raise ValueError("No se pudo codificar la imagen resultante")
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f"data:image/png;base64,{b64}"


def save_image_with_codec(img_rgb: np.ndarray, output_format: str, quality: int) -> tuple[str, str]:
    os.makedirs(MEDIA_DIR, exist_ok=True)
    fmt = output_format.lower().strip()
    quality = int(max(1, min(100, quality)))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if fmt == "jpeg" or fmt == "jpg":
        filename = f"reconstructed_{uuid.uuid4().hex}.jpg"
        flags = [
            int(cv2.IMWRITE_JPEG_QUALITY), quality,
            int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
            int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1,
        ]
        ok = cv2.imwrite(os.path.join(MEDIA_DIR, filename), img_bgr, flags)
        media_type = "image/jpeg"
    elif fmt == "webp":
        filename = f"reconstructed_{uuid.uuid4().hex}.webp"
        flags = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        ok = cv2.imwrite(os.path.join(MEDIA_DIR, filename), img_bgr, flags)
        media_type = "image/webp"
        if not ok:
            # Fallback to JPEG if WebP unsupported
            filename = f"reconstructed_{uuid.uuid4().hex}.jpg"
            flags = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            ok = cv2.imwrite(os.path.join(MEDIA_DIR, filename), img_bgr, flags)
            media_type = "image/jpeg"
    else:
        # PNG (lossless). Use moderate compression (0-9)
        filename = f"reconstructed_{uuid.uuid4().hex}.png"
        png_level = 6
        flags = [int(cv2.IMWRITE_PNG_COMPRESSION), png_level]
        ok = cv2.imwrite(os.path.join(MEDIA_DIR, filename), img_bgr, flags)
        media_type = "image/png"

    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo guardar la imagen reconstruida")

    return filename, media_type


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "error": None, "defaults": {"variance": 0.95, "format": "jpeg", "quality": 85}},
    )


@app.post("/", response_class=HTMLResponse)
async def compress(
    request: Request,
    image: UploadFile = File(...),
    variance: float = Form(...),
    output_format: str = Form("jpeg"),
    quality: int = Form(85),
):
    try:
        if not (0.0 < variance <= 1.0):
            raise ValueError("El factor (varianza retenida) debe estar en (0, 1].")

        image_bytes = await image.read()
        img_rgb = decode_image_to_rgb(image_bytes)

        img_hat_rgb, ks, evrs, _ = compress_image_rgb(img_rgb, variance)

        # Build data URIs for display
        original_uri = encode_image_to_data_uri(img_rgb)
        reconstructed_uri = encode_image_to_data_uri(img_hat_rgb)

        # Save reconstructed to media for download with selected codec
        filename, media_type = save_image_with_codec(img_hat_rgb, output_format, quality)
        download_url = f"/download/{filename}"

        h, w = img_rgb.shape[:2]
        result: Dict[str, object] = {
            "width": w,
            "height": h,
            "variance": variance,
            "k": ks,
            "original_uri": original_uri,
            "reconstructed_uri": reconstructed_uri,
            "download_url": download_url,
            "download_name": filename,
        }

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": result, "error": None, "defaults": {"variance": variance, "format": output_format, "quality": quality}},
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "result": None, "error": str(exc), "defaults": {"variance": 0.95, "format": "jpeg", "quality": 85}},
        )


@app.get("/download/{filename}")
async def download(filename: str):
    # Simple safeguard against path traversal
    if "/" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
    path = os.path.join(MEDIA_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    ext = os.path.splitext(filename)[1].lower()
    media_type = "image/png"
    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".webp":
        media_type = "image/webp"
    return FileResponse(path, media_type=media_type, filename=filename)