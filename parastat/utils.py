from __future__ import annotations

from typing import Dict, Tuple, Any, List, Callable
import importlib.util
import random
import numpy as _np

_CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None
if _CUPY_AVAILABLE:
    import cupy as _cp  # type: ignore
else:
    _cp = None  # type: ignore
_CUDF_AVAILABLE = importlib.util.find_spec("cudf") is not None
if _CUDF_AVAILABLE:
    import cudf as _cudf  # type: ignore
else:
    _cudf = None  # type: ignore


def set_seed(seed: int) -> None:
    _np.random.seed(seed)
    random.seed(seed)
    try:
        import os
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass
    if _CUPY_AVAILABLE and _cp is not None:
        try:
            _cp.random.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass


def device_info() -> Dict[str, object]:
    info: Dict[str, object] = {
        "cpu": True,
        "gpu_available": bool(_CUPY_AVAILABLE),
        "backend": "cupy" if _CUPY_AVAILABLE else "numpy",
        "gpu_count": 0,
        "gpus": [],
    }
    if _CUPY_AVAILABLE and _cp is not None:
        try:
            num = _cp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
            info["gpu_count"] = num
            gpus = []
            for i in range(num):
                props = _cp.cuda.runtime.getDeviceProperties(i)  # type: ignore[attr-defined]
                name = props.get("name", b"GPU").decode("utf-8", errors="ignore") if isinstance(props.get("name"), (bytes, bytearray)) else props.get("name", "GPU")
                total_mem = props.get("totalGlobalMem", 0)
                gpus.append({"id": i, "name": name, "total_mem": int(total_mem)})
            info["gpus"] = gpus
        except Exception:
            pass
    return info


def version() -> str:
    try:
        from importlib.metadata import version as _v
    except Exception:
        try:
            from importlib_metadata import version as _v  # type: ignore
        except Exception:
            _v = None  # type: ignore
    if _v is None:
        return "0.0.0"
    try:
        return _v("parastat")
    except Exception:
        return "0.0.0"


def resolve_backend(device: str = "auto") -> Tuple[Any, str]:
    d = (device or "auto").lower()
    if d == "gpu":
        if _CUPY_AVAILABLE and _cp is not None:
            return _cp, "gpu"
        from .errors import DeviceNotAvailableError
        raise DeviceNotAvailableError("请求 GPU，但未检测到可用的 CuPy/GPU 设备")
    if d == "cpu":
        return _np, "cpu"
    if _CUPY_AVAILABLE and _cp is not None:
        return _cp, "gpu"
    return _np, "cpu"


def drop_na_rows(df: Any, columns: List[str] | None = None) -> Any:
    try:
        if columns is None:
            return df.dropna()
        return df.dropna(subset=columns)
    except Exception:
        return df


def standardize_columns(df: Any, columns: List[str]) -> Any:
    try:
        import pandas as pd
        work = df.copy()
        for c in columns:
            mu = work[c].mean()
            sd = work[c].std(ddof=0) or 1.0
            work[c] = (work[c] - mu) / sd
        return work
    except Exception:
        return df


def to_cpu(obj: Any) -> Any:
    try:
        if _CUPY_AVAILABLE and _cp is not None and hasattr(obj, "get"):
            return obj.get()
    except Exception:
        pass
    try:
        if _CUDF_AVAILABLE and _cudf is not None and isinstance(obj, _cudf.DataFrame):
            return obj.to_pandas()
    except Exception:
        pass
    return obj


def to_gpu(obj: Any) -> Any:
    try:
        if _CUDF_AVAILABLE and _cudf is not None:
            import pandas as pd
            if isinstance(obj, pd.DataFrame):
                return _cudf.from_pandas(obj)
    except Exception:
        pass
    try:
        if _CUPY_AVAILABLE and _cp is not None:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return _cp.asarray(obj)
    except Exception:
        pass
    return obj


def choose_dtype(fp64: bool = True):
    return _np.float64 if fp64 else _np.float32


_GLOBAL_FP64: bool = True


def set_mixed_precision(fp64: bool = True) -> None:
    global _GLOBAL_FP64
    _GLOBAL_FP64 = bool(fp64)


def get_mixed_precision() -> bool:
    return _GLOBAL_FP64


def with_precision(xp: Any) -> Any:
    return xp.float64 if _GLOBAL_FP64 else xp.float32


def gpu_oom_fallback(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args, **kwargs):
        device = kwargs.get("device", "auto")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            if ("cuda" in msg or "out of memory" in msg or "cublas" in msg) and device in {"auto", "gpu"}:
                kwargs["device"] = "cpu"
                return func(*args, **kwargs)
            raise
    return wrapper


