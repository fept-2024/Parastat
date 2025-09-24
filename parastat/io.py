from __future__ import annotations

from typing import Any
import pandas as pd


def read_parquet(path: str) -> Any:
    return pd.read_parquet(path)


def read_csv(path: str, delimiter: str = ",") -> Any:
    return pd.read_csv(path, delimiter=delimiter)


