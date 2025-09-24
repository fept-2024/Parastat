from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class RegressionResult:
    params: Dict[str, float] = field(default_factory=dict)
    std_err: Dict[str, float] = field(default_factory=dict)
    t_values: Dict[str, float] = field(default_factory=dict)
    p_values: Dict[str, float] = field(default_factory=dict)
    conf_int: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    r2: float = 0.0
    adj_r2: float = 0.0
    nobs: int = 0
    aic: float = float("nan")
    bic: float = float("nan")

    def summary(self) -> str:
        names = list(self.params.keys())
        lines = [
            "Regression Result",
            f"N = {self.nobs}",
            f"R^2 = {self.r2:.6f}, adj R^2 = {self.adj_r2:.6f}",
            f"AIC = {self.aic:.6f}, BIC = {self.bic:.6f}",
            "",
            f"{'param':<16}{'coef':>12}{'std_err':>12}{'t':>12}{'p':>12}{'ci_low':>14}{'ci_high':>14}",
        ]
        for name in names:
            coef = self.params.get(name, float('nan'))
            se = self.std_err.get(name, float('nan'))
            t = self.t_values.get(name, float('nan'))
            p = self.p_values.get(name, float('nan'))
            ci = self.conf_int.get(name, (float('nan'), float('nan')))
            lines.append(f"{name:<16}{coef:>12.6f}{se:>12.6f}{t:>12.6f}{p:>12.6f}{ci[0]:>14.6f}{ci[1]:>14.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


