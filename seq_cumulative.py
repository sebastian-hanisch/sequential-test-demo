"""Welch-Test nach jedem gesammelten Fenster, vektorisiert über viele Experimente (Grundlage der Zwischenblicke)."""

import numpy as np
from scipy.special import stdtr


def cumulative_welch(a, b):
    """Welch-Test nach jedem Fenster m = 1..n für viele Experimente zugleich. a, b: Form (R, n). Rückgabe: dict von Arrays
    (R, n); Spalte m-1 gehört zu m Fenstern je Arm. Für m = 1 ist nichts definiert (p = 1, Differenz NaN)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    shift = float(np.concatenate([a.ravel(), b.ravel()]).mean())     # zentrieren: stabile Quadratsummen
    a, b = a - shift, b - shift
    m = np.arange(1, a.shape[1] + 1, dtype=float)[None, :]
    moments = []
    for x in (a, b):
        s1 = np.cumsum(x, axis=1)
        s2 = np.cumsum(x * x, axis=1)
        mean = s1 / m
        with np.errstate(divide="ignore", invalid="ignore"):
            var = (s2 - s1 * s1 / m) / (m - 1)
        moments.append((mean, np.maximum(var, 0.0)))
    (ma, va), (mb, vb) = moments
    with np.errstate(divide="ignore", invalid="ignore"):
        qa, qb = va / m, vb / m
        diff = mb - ma
        se = np.sqrt(qa + qb)
        df = (qa + qb) ** 2 / (qa**2 / (m - 1) + qb**2 / (m - 1))
        t = diff / se
    ok = (m >= 2) & np.isfinite(t) & (se > 0)
    df = np.where(np.isfinite(df) & (df > 0), df, 1.0)
    t = np.where(ok, t, 0.0)
    p = np.where(ok, 2 * stdtr(df, -np.abs(t)), 1.0)
    return {"diff": np.where(m >= 2, diff, np.nan), "se": se, "df": df, "t": t, "p": p, "var_a": va, "var_b": vb, "m": m}
