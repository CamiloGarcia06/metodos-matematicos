"""
Utilidades de PCA para compresión de imágenes RGB sin usar librerías de IA.

Descripción matemática (resumen):
- Sea X \in R^{H x W} una imagen de un canal (H filas = píxeles verticales, W columnas = píxeles horizontales).
- Centramos por columnas: X_c = X - 1_H m^T, con m = media por columna.
- Descomposición SVD: X_c = U S V^T, con U\in R^{H x r}, V\in R^{W x r}, S diagonal (r = min(H, W)).
- Los autovectores de la covarianza C = (1/(H-1)) X_c^T X_c son las columnas de V y los autovalores son (S^2)/(H-1).
- La varianza explicada por la i-ésima componente es lambda_i / sum_j lambda_j, donde lambda_i = S_i^2/(H-1).
- Proyección (scores) en las k primeras componentes: T_k = X_c V_k.
- Reconstrucción truncada: \hat{X} = T_k V_k^T + 1_H m^T.

Este módulo implementa ese procedimiento con numpy.linalg.svd para cada canal RGB.
"""

from typing import Dict

import numpy as np


def choose_k_for_threshold(explained_variance_ratio: np.ndarray, variance_threshold: float) -> int:
    cumsum = np.cumsum(explained_variance_ratio)
    k = int(np.searchsorted(cumsum, variance_threshold, side="left") + 1)
    k = max(1, min(k, len(cumsum)))
    return k


def _pca_fit_via_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta PCA mediante SVD sobre los datos centrados.

    Parámetros
    - X: matriz HxW (float), cada fila es un vector de longitud W.

    Retorna
    - components: V^T (r x W), vectores principales fila a fila.
    - mean: media por columna (W,).
    - explained_variance_ratio: vector (r,) con la fracción de varianza explicada por cada componente.

    Detalles
    - X_c = X - mean.
    - SVD: X_c = U S V^T. La covarianza C = (1/(H-1)) X_c^T X_c = V (S^2/(H-1)) V^T.
    - explained_variance_ratio_i = (S_i^2/(H-1)) / sum_j (S_j^2/(H-1)).
    """
    n_samples, _ = X.shape
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    # SVD: X_centered = U @ diag(S) @ Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Explained variance for each singular value
    if n_samples > 1:
        explained_variance = (S ** 2) / (n_samples - 1)
    else:
        explained_variance = S ** 2
    total_variance = float(np.sum(explained_variance))
    if total_variance > 0.0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)
    components = Vt  # shape (r, n_features)
    return components, mean, explained_variance_ratio


def pca_compress_channel(channel_uint8: np.ndarray, variance_threshold: float):
    """Aplica PCA a un canal de imagen y reconstruye con k componentes.

    Parámetros
    - channel_uint8: matriz HxW de enteros [0, 255] del canal (R, G o B).
    - variance_threshold: umbral en (0, 1] para la varianza retenida acumulada (elige k mínimo tal que sum_{i<=k} EVR_i >= threshold).

    Retorna
    - X_hat: reconstrucción uint8 HxW con k componentes.
    - k: número de componentes utilizados.
    - evr_full: vector con varianza explicada por componente (para referencia/visualización).

    Método
    1) Convierte a float32 y centra por columnas.
    2) Calcula SVD para obtener componentes (V) y EVR.
    3) Elige k por el umbral de varianza.
    4) Proyecta y reconstruye: \hat{X} = (X-mean) V_k V_k^T + mean.
    5) Redondea y limita a [0, 255].
    """
    X = channel_uint8.astype(np.float32)
    components_full, mean, evr_full = _pca_fit_via_svd(X)
    k = choose_k_for_threshold(evr_full, variance_threshold)

    components_k = components_full[:k, :]  # (k, W)
    # Project centered data onto top-k components to obtain scores (H, k)
    scores = (X - mean) @ components_k.T
    # Reconstruct
    X_hat = (scores @ components_k) + mean
    X_hat = np.clip(np.rint(X_hat), 0, 255).astype(np.uint8)

    return X_hat, k, evr_full


def compress_image_rgb(img_rgb: np.ndarray, variance_threshold: float):
    """Comprensión PCA por canales y reconstrucción RGB.

    Parámetros
    - img_rgb: arreglo HxWx3 uint8.
    - variance_threshold: umbral de varianza retenida para cada canal.

    Retorna
    - img_hat: imagen reconstruida HxWx3 uint8.
    - ks: dict con k por canal {"R", "G", "B"}.
    - evrs: dict con EVR por canal para análisis/plot.
    - dict vacío reservado para metadatos futuros.
    """
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    r_hat, k_r, evr_r = pca_compress_channel(r, variance_threshold)
    g_hat, k_g, evr_g = pca_compress_channel(g, variance_threshold)
    b_hat, k_b, evr_b = pca_compress_channel(b, variance_threshold)

    img_hat = np.stack([r_hat, g_hat, b_hat], axis=2)
    ks: Dict[str, int] = {"R": k_r, "G": k_g, "B": k_b}
    evrs: Dict[str, np.ndarray] = {"R": evr_r, "G": evr_g, "B": evr_b}

    return img_hat, ks, evrs, {}