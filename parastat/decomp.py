from __future__ import annotations

from typing import Any, List, Tuple
from .utils import resolve_backend, with_precision, gpu_oom_fallback


@gpu_oom_fallback
def pca(df: Any, X: List[str], n_components: int = 2, device: str = "auto") -> Tuple[Any, Any]:
    xp, _ = resolve_backend(device)
    M = xp.asarray(df[X], dtype=with_precision(xp))
    M = M - xp.mean(M, axis=0, keepdims=True)
    U, S, Vt = xp.linalg.svd(M, full_matrices=False)
    components = Vt[:n_components]
    explained_variance = (S ** 2) / (M.shape[0] - 1)
    explained_variance_ratio = explained_variance[:n_components] / xp.sum(explained_variance)
    return (components, explained_variance_ratio)


@gpu_oom_fallback
def svd(matrix: Any, k: int | None = None, device: str = "auto") -> Tuple[Any, Any, Any]:
    xp, _ = resolve_backend(device)
    M = xp.asarray(matrix, dtype=with_precision(xp))
    U, S, Vt = xp.linalg.svd(M, full_matrices=False)
    if k is not None:
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
    return U, S, Vt


