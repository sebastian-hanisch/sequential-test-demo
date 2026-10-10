"""Gruppensequentielle Grenzen numerisch (rekursive Integration nach Armitage, McPherson und Rowe), nichts aus Tabellen.

Modell unter H0: Z_k = S_k / sqrt(k), S_k = Summe von k unabhängigen N(0,1)-Zuwächsen (gleich lange Abschnitte,
Informationszeit t_k = k/K). Zweiseitige Grenze c_k: Stopp, sobald |Z_k| >= c_k. Die Dichte der Teilsumme liegt auf einem Gitter,
ein Schritt ist die Faltung mit der Normaldichte (FFT); die Zelle an der Grenze wird mit ihrem Anteil gewichtet.

Die Konstanten für 1 bis 30 Blicke stehen vorgerechnet in `precomputed_boundaries.json` (`generate_precomputed.py`)."""

import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from scipy.signal import fftconvolve

H = 0.005  # Gitterweite
PRECOMPUTED = Path(__file__).resolve().parent / "precomputed_boundaries.json"


def _grid(cmax, K, h=H, pad=8.0):
    n = int(np.ceil((cmax * np.sqrt(K) + pad) / h))
    return np.arange(-n, n + 1) * h, n


def _inside_weight(s, thr, h):
    """Anteil jeder Zelle (Mitte s, Breite h), der innerhalb |s| < thr liegt."""
    return np.clip((thr - np.abs(s)) / h + 0.5, 0.0, 1.0)


def crossing_probs(c, h=H, drift=0.0):
    """Erst-Überschreitungswahrscheinlichkeit je Blick für Grenzen c (K,); np.inf = an diesem Blick kein Test.
    drift = Erwartungswert jedes N(., 1)-Zuwachses (0 = H0; größer 0 = Alternative)."""
    c = np.asarray(c, dtype=float)
    K = len(c)
    finite = c[np.isfinite(c)]
    cmax = finite.max() if len(finite) else 3.0
    s, n0 = _grid(cmax + abs(drift), K, h)
    m = int(np.ceil((8.5 + 2 * abs(drift)) / h))
    kern = stats.norm.pdf(np.arange(-m, m + 1) * h - drift)
    f = np.zeros_like(s)
    f[n0] = 1.0 / h
    out = np.zeros(K)
    for k in range(1, K + 1):
        f = fftconvolve(f, kern, mode="same") * h
        if np.isfinite(c[k - 1]):
            w = _inside_weight(s, c[k - 1] * np.sqrt(k), h)
            out[k - 1] = float((f * (1.0 - w)).sum() * h)
            f = f * w
    return out


def total_cross(c, h=H, drift=0.0):
    return float(crossing_probs(c, h, drift).sum())


def pocock_const(K, alpha=0.05, h=H):
    """Konstante c: gleiche Grenze an allen K Blicken."""
    return optimize.brentq(lambda x: total_cross(np.full(K, x), h) - alpha, 1.5, 4.5, xtol=1e-9)


def obf_const(K, alpha=0.05, h=H):
    """O'Brien-Fleming-Form c_k = C sqrt(K/k): Rückgabe (C, Grenzen)."""
    base = np.sqrt(K / np.arange(1, K + 1))
    C = optimize.brentq(lambda x: total_cross(x * base, h) - alpha, 1.5, 4.5, xtol=1e-9)
    return C, C * base


def spend_obf(t, alpha=0.05):
    """Ausgabefunktion vom O'Brien-Fleming-Typ (Lan und DeMets 1983): 2 - 2 Phi(z_{alpha/2} / sqrt(t))."""
    return 2.0 * (1.0 - stats.norm.cdf(stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(t)))


def lan_demets(K, spend=spend_obf, alpha=0.05, h=H):
    """Lan-DeMets-Grenzen für K gleichabständige Blicke aus der Ausgabefunktion `spend`. Ein Blick, dessen Budget so klein ist,
    dass eine Grenze über 8 nötig wäre, wird nicht getestet (Grenze inf); sein Budget wandert weiter."""
    t = np.arange(1, K + 1) / K
    cum = spend(t, alpha)
    cum[-1] = alpha
    s, n0 = _grid(6.0 if K <= 20 else 4.5, K, h)
    m = int(np.ceil(8.5 / h))
    kern = stats.norm.pdf(np.arange(-m, m + 1) * h)
    f = np.zeros_like(s)
    f[n0] = 1.0 / h
    c = np.full(K, np.inf)
    spent = 0.0
    for k in range(1, K + 1):
        f = fftconvolve(f, kern, mode="same") * h
        target = cum[k - 1] - spent
        sk = np.sqrt(k)

        def g(x):
            w = _inside_weight(s, x * sk, h)
            return float((f * (1.0 - w)).sum() * h) - target

        if target <= 1e-10 or g(8.0) > 0:
            continue
        c[k - 1] = optimize.brentq(g, 0.5, 8.0, xtol=1e-10)
        w = _inside_weight(s, c[k - 1] * sk, h)
        spent += float((f * (1.0 - w)).sum() * h)
        f = f * w
    return c


def load_boundaries(path=PRECOMPUTED):
    """{K: {"pocock": c, "obf_C": C, "obf": [c_k], "ld": [c_k mit None für kein Test]}} aus der vorgerechneten Datei."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for k, v in data["boundaries"].items():
        out[int(k)] = {
            "pocock": float(v["pocock"]),
            "obf_C": float(v["obf_C"]),
            "obf": np.asarray(v["obf"], dtype=float),
            "ld": np.asarray([np.inf if x is None else x for x in v["ld"]], dtype=float),
        }
    return out
