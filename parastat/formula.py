from __future__ import annotations

from typing import List, Tuple, Any
import re
import pandas as pd

from .models import ols, glm
from .utils import drop_na_rows


_CATEG_RE = re.compile(r"^C\((?P<name>[^)]+)\)$")


def parse_formula(formula: str) -> Tuple[str, List[str]]:
    if "~" not in formula:
        raise ValueError("公式需要包含 '~'")
    left, right = [s.strip() for s in formula.split("~", 1)]
    y = left
    terms = [t.strip() for t in right.split("+") if t.strip()]
    return y, terms


def build_design_matrices(df: Any, formula: str, dropna: bool = True) -> Tuple[pd.DataFrame, str, List[str]]:
    y, terms = parse_formula(formula)
    work = pd.DataFrame(df, copy=True)
    X_cols: List[str] = []
    for t in terms:
        m = _CATEG_RE.match(t)
        if m:
            col = m.group("name")
            dummies = pd.get_dummies(work[col], prefix=f"C({col})", prefix_sep=":", drop_first=True)
            dummies.columns = [str(c) for c in dummies.columns]
            for c in dummies.columns:
                if c in work.columns:
                    raise ValueError(f"生成的哑变量列与现有列重名: {c}")
            work = pd.concat([work, dummies], axis=1)
            X_cols.extend(list(dummies.columns))
        else:
            X_cols.append(t)
    if dropna:
        cols = [y] + X_cols
        work = drop_na_rows(work, columns=cols)
    return work, y, X_cols


def ols_formula(df: Any, formula: str, robust: bool | str = False, device: str = "auto", dropna: bool = True):
    work, y, X_cols = build_design_matrices(df, formula, dropna=dropna)
    return ols(work, y=y, X=X_cols, robust=robust, device=device)


def glm_formula(df: Any, formula: str, family: str = "logit", device: str = "auto", dropna: bool = True):
    work, y, X_cols = build_design_matrices(df, formula, dropna=dropna)
    return glm(work, y=y, X=X_cols, family=family, device=device)


