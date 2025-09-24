import parastat as ms
import pandas as pd


def test_formula_with_category():
    df = pd.DataFrame({
        "y": [1, 0, 1, 0, 1, 0],
        "x1": [0.1, 0.2, 0.0, 0.5, 0.3, 0.4],
        "g": ["a", "b", "a", "b", "b", "a"],
    })
    work, y, X = ms.build_design_matrices(df, "y ~ x1 + C(g)")
    assert y == "y"
    assert any(c.startswith("C(g):") for c in X)
    # 公式 OLS/GLM 可调用
    res1 = ms.ols_formula(df, "y ~ x1 + C(g)", device="cpu")
    assert hasattr(res1, "params")
    res2 = ms.glm_formula(df, "y ~ x1 + C(g)", family="logit", device="cpu")
    assert hasattr(res2, "params")


def test_formula_dropna():
    import numpy as np
    df = pd.DataFrame({
        "y": [1.0, np.nan, 0.0, 1.0],
        "x1": [0.1, 0.2, np.nan, 0.3],
        "g": ["a", "b", "a", "b"],
    })
    work, y, X = ms.build_design_matrices(df, "y ~ x1 + C(g)", dropna=True)
    assert len(work) < len(df)


