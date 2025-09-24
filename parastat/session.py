from __future__ import annotations

from typing import Any, Dict
import logging
import pandas as pd

from .utils import device_info as _device_info, resolve_backend
from .utils import _CUDF_AVAILABLE, _cudf


class Session:
    device: str
    log_level: str
    cache_enabled: bool

    def __init__(self, device: str = "auto", log_level: str = "info") -> None:
        self.device = device
        self.log_level = log_level
        self.cache_enabled = False
        self._logger = logging.getLogger("parastat")
        level = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warn": logging.WARN,
            "warning": logging.WARN,
            "error": logging.ERROR,
        }.get((log_level or "info").lower(), logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(fmt)
            self._logger.addHandler(handler)
        self._logger.setLevel(level)

    def read_parquet(self, path: str):
        if self.device in {"auto", "gpu"} and _CUDF_AVAILABLE and _cudf is not None:
            try:
                return _cudf.read_parquet(path)
            except Exception:
                pass
        return pd.read_parquet(path)

    def read_csv(self, path: str, delimiter: str = ","):
        if self.device in {"auto", "gpu"} and _CUDF_AVAILABLE and _cudf is not None:
            try:
                return _cudf.read_csv(path, sep=delimiter)
            except Exception:
                pass
        return pd.read_csv(path, delimiter=delimiter)

    def info(self) -> Dict[str, Any]:
        info = _device_info()
        info.update({
            "session_device": self.device,
            "log_level": self.log_level,
            "cache_enabled": self.cache_enabled,
        })
        return info

    def close(self) -> None:
        return None


