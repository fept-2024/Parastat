"""parastat 顶层包。

CPU/GPU 并行的数据统计分析库。
"""

from .session import Session
from .models import ols, glm
from .decomp import pca, svd
from .resampling import bootstrap, monte_carlo
from .utils import set_seed, device_info, version
from .types import RegressionResult
from .errors import (
    DeviceNotAvailableError,
    OutOfMemoryError,
    InvalidArgumentError,
    ComputationError,
)
from .formula import parse_formula, build_design_matrices, ols_formula, glm_formula

__all__ = [
    "Session",
    "ols",
    "glm",
    "pca",
    "svd",
    "bootstrap",
    "monte_carlo",
    "set_seed",
    "device_info",
    "version",
    "RegressionResult",
    "DeviceNotAvailableError",
    "OutOfMemoryError",
    "InvalidArgumentError",
    "ComputationError",
    "parse_formula",
    "build_design_matrices",
    "ols_formula",
    "glm_formula",
]


