from __future__ import annotations

from typing import Any, List

from .types import RegressionResult
from .utils import resolve_backend, with_precision, gpu_oom_fallback
from .errors import InvalidArgumentError
import numpy as np


@gpu_oom_fallback
def ols(
    df: Any,
    y: str,
    X: List[str],
    robust: bool | str = False,
    device: str = "auto",
    cluster: str | None = None,
    hac_lags: int | None = None,
) -> RegressionResult:
    xp, _ = resolve_backend(device)
    y_vec = xp.asarray(df[y]).reshape(-1, 1)
    X_mat = xp.asarray(df[X], dtype=with_precision(xp))
    n = X_mat.shape[0]
    intercept = xp.ones((n, 1), dtype=with_precision(xp))
    Xd = xp.concatenate([intercept, X_mat], axis=1)
    names = ["const"] + list(X)
    XtX = Xd.T @ Xd
    XtX_inv = xp.linalg.inv(XtX)
    beta = XtX_inv @ (Xd.T @ y_vec)
    resid = y_vec - Xd @ beta
    dof = max(n - Xd.shape[1], 1)

    robust_mode = None
    if isinstance(robust, bool):
        robust_mode = "hc1" if robust else None
    elif isinstance(robust, str):
        robust_mode = robust.lower()
        if robust_mode not in {"hc0", "hc1", "hc2", "hc3"}:
            raise InvalidArgumentError(f"不支持的 robust 选项: {robust}")
    else:
        raise InvalidArgumentError("robust 仅支持 bool 或 str: 'hc0'|'hc1'|'hc2'|'hc3'")

    if cluster is not None:
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            raise InvalidArgumentError("cluster 仅支持 pandas DataFrame")
        groups = df[cluster].values
        XtX_inv = xp.asarray(XtX_inv)
        meat = xp.zeros_like(XtX)
        Xd_cpu = Xd if not hasattr(xp, "asnumpy") else xp.asnumpy(Xd)
        resid_cpu = resid if not hasattr(xp, "asnumpy") else xp.asnumpy(resid)
        import numpy as _np
        by = {}
        for i, g in enumerate(groups):
            by.setdefault(g, []).append(i)
        for idxs in by.values():
            Xg = _np.asarray(Xd_cpu[idxs, :])
            ug = _np.asarray(resid_cpu[idxs, :])
            meat += Xg.T @ (ug @ ug.T) @ Xg
        cov = XtX_inv @ meat @ XtX_inv
    elif hac_lags is not None and int(hac_lags) > 0:
        L = int(hac_lags)
        e = resid
        S0 = Xd.T @ (e @ e.T) @ Xd
        S = S0
        for l in range(1, L + 1):
            w = 1.0 - l / (L + 1.0)
            e_f = e[l:]
            e_b = e[:-l]
            X_f = Xd[l:]
            X_b = Xd[:-l]
            Gamma = X_b.T @ (e_b @ e_f.T) @ X_f
            S += w * (Gamma + Gamma.T)
        cov = XtX_inv @ S @ XtX_inv
    elif robust_mode:
        r = xp.squeeze(resid)
        if robust_mode == "hc0":
            w = r ** 2
            S = xp.diagflat(w)
            meat = Xd.T @ S @ Xd
            cov = XtX_inv @ meat @ XtX_inv
        elif robust_mode == "hc1":
            w = (n / dof) * (r ** 2)
            S = xp.diagflat(w)
            meat = Xd.T @ S @ Xd
            cov = XtX_inv @ meat @ XtX_inv
        elif robust_mode == "hc2":
            H = Xd @ XtX_inv @ Xd.T
            h = xp.clip(xp.diag(H), 1e-12, 1.0)
            w = r ** 2 / (1.0 - h)
            S = xp.diagflat(w)
            meat = Xd.T @ S @ Xd
            cov = XtX_inv @ meat @ XtX_inv
        else:
            H = Xd @ XtX_inv @ Xd.T
            h = xp.clip(xp.diag(H), 1e-12, 1.0)
            w = r ** 2 / (1.0 - h) ** 2
            S = xp.diagflat(w)
            meat = Xd.T @ S @ Xd
            cov = XtX_inv @ meat @ XtX_inv
    else:
        sigma2 = float((resid.T @ resid) / dof)
        cov = XtX_inv * sigma2
    se = xp.sqrt(xp.diag(cov))
    tvals = xp.squeeze(beta) / se
    from math import erf, sqrt

    def two_sided_p(z):
        x = abs(float(z)) / sqrt(2.0)
        return 2.0 * (1.0 - (1.0 + erf(x)) / 2.0)

    pvals = [two_sided_p(v) for v in xp.asnumpy(tvals) if hasattr(xp, "asnumpy")] if hasattr(xp, "asnumpy") else [two_sided_p(v) for v in tvals]
    z = 1.959963984540054
    beta_vec = (xp.asnumpy(beta).flatten() if hasattr(xp, "asnumpy") else beta.flatten())
    se_vec = (xp.asnumpy(se).flatten() if hasattr(xp, "asnumpy") else se)
    params = {name: float(val) for name, val in zip(names, beta_vec)}
    std_err = {name: float(val) for name, val in zip(names, se_vec)}
    t_values = {name: float(val) for name, val in zip(names, (xp.asnumpy(tvals).flatten() if hasattr(xp, "asnumpy") else tvals))}
    p_values = {name: float(val) for name, val in zip(names, pvals)}
    conf_int = {name: (float(b - z * s), float(b + z * s)) for name, b, s in zip(names, beta_vec, se_vec)}

    y_mean = float(xp.mean(y_vec))
    ss_tot = float(xp.sum((y_vec - y_mean) ** 2))
    ss_res = float(xp.sum(resid ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    adj_r2 = 1.0 - (1 - r2) * (n - 1) / dof if dof > 0 else r2
    aic = _ols_aic(ss_res, n, Xd.shape[1])
    bic = _ols_bic(ss_res, n, Xd.shape[1])
    return RegressionResult(
        params=params,
        std_err=std_err,
        t_values=t_values,
        p_values=p_values,
        conf_int=conf_int,
        r2=r2,
        adj_r2=adj_r2,
        nobs=int(n),
        aic=aic,
        bic=bic,
    )


@gpu_oom_fallback
def glm(df: Any, y: str, X: List[str], family: str = "logit", device: str = "auto") -> RegressionResult:
    fam = (family or "logit").lower()
    if fam not in {"logit", "poisson", "probit"}:
        raise InvalidArgumentError(f"不支持的 family: {family}")

    xp, _ = resolve_backend(device)
    y_vec = xp.asarray(df[y]).reshape(-1, 1)
    X_mat = xp.asarray(df[X], dtype=with_precision(xp))
    n = X_mat.shape[0]
    Xd = xp.concatenate([xp.ones((n, 1), dtype=X_mat.dtype), X_mat], axis=1)
    p = Xd.shape[1]
    beta = xp.zeros((p, 1), dtype=X_mat.dtype)

    from math import sqrt, pi
    for _iter in range(50):
        eta = Xd @ beta
        if fam == "logit":
            mu = 1.0 / (1.0 + xp.exp(-eta))
            W = xp.squeeze(mu * (1 - mu))
        elif fam == "poisson":
            mu = xp.exp(eta)
            W = xp.squeeze(mu)
        else:
            def phi(z):
                return (1.0 / sqrt(2.0 * pi)) * xp.exp(-0.5 * z ** 2)

            def Phi_scalar(t: float) -> float:
                from math import erf
                return 0.5 * (1.0 + erf(t / sqrt(2.0)))

            mu = xp.vectorize(lambda t: Phi_scalar(float(t)))(eta)
            mu = xp.asarray(mu).reshape(-1, 1)
            W = xp.squeeze(phi(eta) ** 2 / (mu * (1 - mu) + 1e-8))

        z_adj = eta + (y_vec - mu) / (W.reshape(-1, 1) + 1e-8)
        WX = Xd * W.reshape(-1, 1)
        XtWX = Xd.T @ WX
        XtWz = Xd.T @ (W.reshape(-1, 1) * z_adj)
        try:
            beta_new = xp.linalg.solve(XtWX, XtWz)
        except Exception:
            beta_new = xp.linalg.pinv(XtWX) @ XtWz
        if float(xp.max(xp.abs(beta_new - beta))) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    try:
        cov = xp.linalg.inv(XtWX)
    except Exception:
        cov = xp.linalg.pinv(XtWX)
    se = xp.sqrt(xp.diag(cov))
    tvals = xp.squeeze(beta) / se
    from math import erf, sqrt

    def two_sided_p(z):
        x = abs(float(z)) / sqrt(2.0)
        return 2.0 * (1.0 - (1.0 + erf(x)) / 2.0)

    pvals = [two_sided_p(v) for v in xp.asnumpy(tvals) if hasattr(xp, "asnumpy")] if hasattr(xp, "asnumpy") else [two_sided_p(v) for v in tvals]
    names = ["const"] + list(X)
    beta_vec = (xp.asnumpy(beta).flatten() if hasattr(xp, "asnumpy") else beta.flatten())
    se_vec = (xp.asnumpy(se).flatten() if hasattr(xp, "asnumpy") else se)
    params = {name: float(val) for name, val in zip(names, beta_vec)}
    std_err = {name: float(val) for name, val in zip(names, se_vec)}
    t_values = {name: float(val) for name, val in zip(names, (xp.asnumpy(tvals).flatten() if hasattr(xp, "asnumpy") else tvals))}
    p_values = {name: float(val) for name, val in zip(names, pvals)}
    z = 1.959963984540054
    conf_int = {name: (float(b - z * s), float(b + z * s)) for name, b, s in zip(names, beta_vec, se_vec)}

    r2 = 0.0
    if fam in {"logit", "probit"}:
        yb = Xd @ beta
        yb = xp.asnumpy(yb).flatten() if hasattr(xp, "asnumpy") else yb.flatten()
        y0 = xp.asnumpy(y_vec).flatten() if hasattr(xp, "asnumpy") else y_vec.flatten()
        num = float(np.corrcoef(y0, yb)[0, 1]) if len(y0) > 1 else 0.0
        r2 = float(num ** 2)

    # 对数似然与信息准则
    eta = Xd @ beta
    if fam == "logit":
        mu = 1.0 / (1.0 + xp.exp(-eta))
        eps = 1e-12
        ll = float(xp.sum(y_vec * xp.log(mu + eps) + (1 - y_vec) * xp.log(1 - mu + eps)))
    elif fam == "poisson":
        mu = xp.exp(eta)
        y_arr = xp.asnumpy(y_vec).flatten() if hasattr(xp, "asnumpy") else y_vec.flatten()
        mu_arr = xp.asnumpy(mu).flatten() if hasattr(xp, "asnumpy") else mu.flatten()
        from math import lgamma
        ll = float(np.sum(y_arr * np.log(mu_arr + 1e-12) - mu_arr - np.vectorize(lambda t: lgamma(t + 1.0))(y_arr)))
    else:
        from math import erf, sqrt
        def Phi_scalar(t: float) -> float:
            return 0.5 * (1.0 + erf(t / sqrt(2.0)))
        mu = xp.vectorize(lambda t: Phi_scalar(float(t)))(eta)
        eps = 1e-12
        ll = float(xp.sum(y_vec * xp.log(mu + eps) + (1 - y_vec) * xp.log(1 - mu + eps)))
    k = Xd.shape[1]
    aic = 2 * k - 2 * ll
    bic = k * np.log(max(n, 1)) - 2 * ll

    return RegressionResult(
        params=params,
        std_err=std_err,
        t_values=t_values,
        p_values=p_values,
        conf_int=conf_int,
        r2=r2,
        adj_r2=r2,
        nobs=int(n),
        aic=float(aic),
        bic=float(bic),
    )


def _ols_aic(ss_res: float, n: int, k: int) -> float:
    from math import log, pi
    sigma2 = ss_res / n if n > 0 else float('nan')
    return n * (1.0 + (0.0 if sigma2 <= 0 else log(2.0 * pi * sigma2))) + 2 * k


def _ols_bic(ss_res: float, n: int, k: int) -> float:
    from math import log, pi
    sigma2 = ss_res / n if n > 0 else float('nan')
    return n * (1.0 + (0.0 if sigma2 <= 0 else log(2.0 * pi * sigma2))) + k * log(max(n, 1))


