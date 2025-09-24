from __future__ import annotations

from typing import Any, Callable, List, Tuple
from .utils import resolve_backend, set_seed as _set_seed
import numpy as np


def bootstrap(df: Any, func: Callable[[Any], Any], reps: int = 1000, device: str = "auto", seed: int | None = None) -> List[Any]:
    if seed is not None:
        _set_seed(seed)
    xp, _ = resolve_backend(device)
    n = len(df)
    results: List[Any] = []
    for _ in range(int(reps)):
        idx = (xp.random.randint(0, n, size=n) if hasattr(xp.random, "randint") else np.random.randint(0, n, size=n))
        try:
            sample = df.iloc[idx]
        except Exception:
            sample = df[idx]
        results.append(func(sample))
    return results


def monte_carlo(func: Callable[[Any], Any], n: int, device: str = "auto", seed: int | None = None) -> List[Any]:
    if seed is not None:
        _set_seed(seed)
    xp, _ = resolve_backend(device)
    results: List[Any] = []
    for _ in range(int(n)):
        rnd = xp.random.random()
        results.append(func(rnd))
    return results


def bootstrap_ci(values: List[float], alpha: float = 0.05) -> Tuple[float, float]:
    import numpy as np
    if not values:
        return float("nan"), float("nan")
    lo = 100 * (alpha / 2.0)
    hi = 100 * (1.0 - alpha / 2.0)
    q = np.percentile(values, [lo, hi])
    return float(q[0]), float(q[1])


