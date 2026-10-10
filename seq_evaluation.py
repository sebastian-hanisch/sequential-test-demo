"""Auswertung: Zwischenblicke auf sich füllenden Experimenten, sechs Entscheidungsregeln nebeneinander.

Wie in den Vorgängerstücken sind die Pools (je 10 000 simulierte Standort-Fenster je Sicherheitsfaktor, SplitMix64) die
Wahrheit; ein Experiment zieht seine Fenster nacheinander mit Zurücklegen. Nach jedem der K gleichmäßig verteilten Blicke wird
der Welch-Test auf allen bis dahin gesammelten Fenstern gerechnet. Die Regeln unterscheiden sich nur darin, wann sie stoppen."""

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr, ndtri

import seq_boundaries as B
import seq_constants as C
from seq_cumulative import cumulative_welch
from seq_rng import uniform_block

PRECOMPUTED = Path(__file__).resolve().parent / "precomputed_pools.json"


def load_pools(path=PRECOMPUTED):
    """{z: Array der Fenster-Servicegrade}; z auf eine Nachkommastelle gerundet."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {round(float(k), 1): np.asarray(v, dtype=float) for k, v in data["pools"].items()}


def nearest_z(z):
    return min(C.Z_GRID, key=lambda g: abs(g - z))


def true_effect(pools, z_b):
    """Wahre Wirkung von Regel B gegenüber A (Mittel der Pools, B minus A)."""
    return float(pools[nearest_z(z_b)].mean() - pools[nearest_z(C.Z_A)].mean())


def _indices(pool_size, count, seed):
    return np.minimum((uniform_block(seed, count) * pool_size).astype(np.int64), pool_size - 1)


def draw_paths(pools, z_b, n_max, reps, seed):
    """`reps` Experimente mit je n_max Fenstern je Arm in der Reihenfolge ihres Eintreffens (Form (reps, n_max))."""
    pa, pb = pools[nearest_z(C.Z_A)], pools[nearest_z(z_b)]
    count = reps * n_max
    a = pa[_indices(len(pa), count, seed).reshape(reps, n_max)]
    b = pb[_indices(len(pb), count, seed + 7_919).reshape(reps, n_max)]
    return a, b


def look_positions(n_max, k):
    """Fenster je Arm bei den k gleichmäßig verteilten Blicken (1-basiert, aufsteigend, mindestens 2)."""
    pos = np.ceil(n_max * np.arange(1, k + 1) / k).astype(int)
    return np.unique(np.maximum(pos, 2))


def z_from_p(p):
    """Betrag der Normalstatistik, die zum zweiseitigen p-Wert gehört (der saubere Weg für Grenzen auf der z-Skala)."""
    return ndtri(1.0 - np.clip(p, 1e-300, 1.0) / 2.0)


def msprt_lambda(diff, var_a, var_b, m, tau=C.MSPRT_TAU):
    """Mischungs-Likelihood-Quotient (Robbins-Mischung, Normalmodell mit Plug-in-Varianz) nach jedem Fenster m."""
    v = (var_a + var_b) / m
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lam = np.sqrt(v / (v + tau**2)) * np.exp(tau**2 * diff**2 / (2 * v * (v + tau**2)))
    return np.where(np.isfinite(lam), lam, np.where(np.isnan(diff) | (v <= 0), 1.0, np.inf))


def block_bounds(bounds, k, kk):
    """Grenzen für kk tatsächliche Blicke (bei kleinem n_max können Blicke zusammenfallen): vorgerechnet, sonst neu gerechnet."""
    bd = bounds[min(max(k, C.K_MIN), C.K_MAX)]
    if kk == len(bd["obf"]):
        return np.full(kk, bd["pocock"]), bd["obf"], bd["ld"]
    c_obf, obf = B.obf_const(kk)
    return np.full(kk, B.pocock_const(kk)), obf, B.lan_demets(kk)


def analyze(a, b, k, bounds):
    """Wendet alle sechs Regeln auf die Pfade an. Rückgabe: {Regel: {"reject", "stop", "est"}} (Arrays der Länge R:
    Befund ja/nein, Fenster je Arm beim Stopp (ohne Befund n_max), Schätzung beim Stopp) und "path" mit Statistiken je Fenster."""
    n_max = a.shape[1]
    w = cumulative_welch(a, b)
    pos = look_positions(n_max, k)
    idx = pos - 1
    p_l = w["p"][:, idx]
    z_l = z_from_p(p_l)
    diff_l = w["diff"][:, idx]
    kk = len(pos)
    pocock, obf, ld = block_bounds(bounds, k, kk)
    hits = {
        "naiv": p_l < C.ALPHA,
        "pocock": z_l >= pocock[None, :],
        "obf": z_l >= obf[None, :],
        "ld": z_l >= ld[None, :],
        "bayes": ndtr(diff_l / w["se"][:, idx]) >= C.BAYES_LEVEL,
    }
    stops = {}
    rows = np.arange(len(a))
    for rule, h in hits.items():
        h = h & np.isfinite(diff_l)
        any_ = h.any(axis=1)
        first = np.where(any_, h.argmax(axis=1), kk - 1)
        stops[rule] = (any_, pos[first], diff_l[rows, first])
    lam = msprt_lambda(w["diff"], w["var_a"], w["var_b"], w["m"])
    lam[:, : C.MSPRT_BURN - 1] = 1.0
    h = lam >= 1.0 / C.ALPHA
    any_ = h.any(axis=1)
    first = np.where(any_, h.argmax(axis=1), n_max - 1)
    stops["msprt"] = (any_, first + 1, w["diff"][rows, first])
    out = {rule: {"reject": r, "stop": s, "est": e} for rule, (r, s, e) in stops.items()}
    out["path"] = {"z": z_from_p(w["p"]) * np.sign(np.nan_to_num(w["diff"])), "diff": w["diff"], "lambda": lam, "pos": pos,
                   "var_a": w["var_a"], "var_b": w["var_b"], "bounds": {"pocock": pocock, "obf": obf, "ld": ld}}
    return out


def experiment_seed(z, n, k, tag=0):
    return 11_000_000 + 10_007 * int(round(z * 10)) + 131 * n + 17 * k + 13 * tag


def summarize(pools, z_b, n_max, k, reps=C.LIVE_REPS, bounds=None, tag=0, keep=False):
    """Viele Experimente bei (z_b, n_max, k): je Regel Entscheidungsraten, Fenster bis zum Stopp und Überschätzung."""
    bounds = bounds or B.load_boundaries()
    z_b = nearest_z(z_b)
    a, b = draw_paths(pools, z_b, n_max, reps, experiment_seed(z_b, n_max, k, tag))
    res = analyze(a, b, k, bounds)
    truth = true_effect(pools, z_b)
    rows = {}
    for rule in C.RULES:
        r = res[rule]
        rej, est = r["reject"], r["est"]
        wrong = rej & (est < 0)
        rows[rule] = {
            "reject": float(rej.mean()),
            "wrong_sign": float(wrong.mean()),
            "mean_stop": float(r["stop"].mean()),
            "mean_stop_rejected": float(r["stop"][rej].mean()) if rej.any() else float("nan"),
            "est_rejected": float(est[rej].mean()) if rej.any() else float("nan"),
            "exaggeration": float(est[rej].mean() / truth) if rej.any() and truth > 0 else float("nan"),
        }
    out = {"z": z_b, "n": n_max, "k": k, "reps": reps, "truth": truth, "rows": rows}
    if keep:
        out["analysis"] = res
    return out


def run_one(pools, settings, bounds=None):
    """Das eine Experiment der Seitenleiste: ein Pfad mit n_max Fenstern je Arm."""
    bounds = bounds or B.load_boundaries()
    a, b = draw_paths(pools, settings["z"], settings["n"], 1, 1_000_003 * (settings["seed"] + 1) + 5)
    res = analyze(a, b, settings["k"], bounds)
    return {"a": a[0], "b": b[0], "res": res, "truth": true_effect(pools, settings["z"])}
