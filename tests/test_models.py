import parastat as ms
import pandas as pd
import numpy as np


def test_imports():
    assert hasattr(ms, "ols")
    assert hasattr(ms, "glm")
    assert hasattr(ms, "pca")
    assert hasattr(ms, "svd")
    assert hasattr(ms, "bootstrap")
    assert hasattr(ms, "monte_carlo")
    assert hasattr(ms, "Session")


def _make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_ols_and_glm():
    df = _make_df(100)
    res = ms.ols(df, y="y", X=["x1", "x2"], robust=False, device="cpu")
    assert isinstance(res, ms.RegressionResult)
    assert set(["const", "x1", "x2"]).issubset(set(res.params.keys()))
    # summary 可调用
    txt = res.summary()
    assert isinstance(txt, str) and "Regression Result" in txt
    # robust 变体
    for mode in ["hc0", "hc1", "hc2", "hc3", True, False]:
        _ = ms.ols(df, y="y", X=["x1", "x2"], robust=mode, device="cpu")
    # cluster 与 HAC 可调用
    df2 = df.copy()
    df2["g"] = (df2.index % 3).astype(int)
    _ = ms.ols(df2, y="y", X=["x1", "x2"], device="cpu", cluster="g")
    _ = ms.ols(df2, y="y", X=["x1", "x2"], device="cpu", hac_lags=2)
    # AIC/BIC 与 CI
    assert isinstance(res.aic, float)
    assert isinstance(res.bic, float)
    assert "const" in res.conf_int
    res_g = ms.glm(df.assign(y=(df["y"] > df["y"].median()).astype(int)), y="y", X=["x1", "x2"], family="logit", device="cpu")
    assert isinstance(res_g, ms.RegressionResult)


def test_pca_and_svd():
    df = _make_df(60)
    comps, evr = ms.pca(df, X=["x1", "x2"], n_components=2, device="cpu")
    assert len(evr) == 2
    U, S, Vt = ms.svd(df[["x1", "x2"]].to_numpy(), k=2, device="cpu")
    assert len(S) == 2


def test_resampling():
    df = _make_df(40)
    res = ms.bootstrap(df, func=lambda d: float(d["y"].mean()), reps=10, device="cpu", seed=42)
    assert len(res) == 10
    mc = ms.monte_carlo(func=lambda r: float(r * 2), n=5, device="cpu", seed=0)
    assert len(mc) == 5



