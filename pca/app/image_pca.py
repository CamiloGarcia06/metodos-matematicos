from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA


def choose_k_for_threshold(explained_variance_ratio: np.ndarray, variance_threshold: float) -> int:
    cumsum = np.cumsum(explained_variance_ratio)
    k = int(np.searchsorted(cumsum, variance_threshold, side="left") + 1)
    k = max(1, min(k, len(cumsum)))
    return k


def pca_compress_channel(channel_uint8: np.ndarray, variance_threshold: float):
    X = channel_uint8.astype(np.float32)
    pca_full = PCA(n_components=None, svd_solver="full").fit(X)
    k = choose_k_for_threshold(pca_full.explained_variance_ratio_, variance_threshold)

    pca_k = PCA(n_components=k, svd_solver="full")
    scores = pca_k.fit_transform(X)  # H x k
    components = pca_k.components_   # k x W
    mean = pca_k.mean_               # W

    X_hat = (scores @ components) + mean
    X_hat = np.clip(np.rint(X_hat), 0, 255).astype(np.uint8)

    return X_hat, k, pca_full.explained_variance_ratio_


def compress_image_rgb(img_rgb: np.ndarray, variance_threshold: float):
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