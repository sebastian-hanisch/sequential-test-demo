"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Scrollen auf dem Handy nicht zoomt."""

import numpy as np
import plotly.graph_objects as go

import seq_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=10), height=380, legend=dict(orientation="h", y=-0.25))


def _finish(fig, title, xtitle=None, ytitle=None, **extra):
    fig.update_layout(title=dict(text=title, x=0), xaxis=dict(title=xtitle, fixedrange=True),
                      yaxis=dict(title=ytitle, fixedrange=True), **{**_LAYOUT, **extra})
    return fig


def msprt_boundary(m, var_a, var_b, tau=C.MSPRT_TAU, alpha=C.ALPHA):
    """Grenze des mSPRT auf der z-Skala bei m Fenstern je Arm: |Differenz| / Standardfehler >= b(m)."""
    v = (var_a + var_b) / np.asarray(m, dtype=float)
    return np.sqrt(2 * (1 + v / tau**2) * np.log(np.sqrt(1 + tau**2 / v) / alpha))


def build_path(z_path, pos, n_max, bounds, msprt_curve, stops, show):
    """Der z-Pfad eines Experiments mit den Grenzen der Regeln. show: Menge der Regeln, deren Grenze gezeichnet wird."""
    m = np.arange(1, n_max + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m[1:], y=z_path[1:], mode="lines", name="Pfad des Experiments", line=dict(color=C.COLOR_A, width=2)))

    def both(x, y, name, color, dash="dash", mode="lines", shape=None):
        for sign, first in ((1, True), (-1, False)):
            fig.add_trace(go.Scatter(x=x, y=sign * np.asarray(y), mode=mode, name=name, legendgroup=name, showlegend=first,
                                     line=dict(color=color, width=1.8, dash=dash, shape=shape or "linear"),
                                     marker=dict(color=color, size=6)))

    if "naiv" in show:
        both([2, n_max], [1.96, 1.96], C.RULE_SHORT["naiv"] + " (1,96)", C.RULE_COLORS["naiv"])
    if "pocock" in show:
        both(list(pos), list(bounds["pocock"]), C.RULE_SHORT["pocock"], C.RULE_COLORS["pocock"], "dot", "lines+markers")
    if "obf" in show:
        both(list(pos), list(bounds["obf"]), C.RULE_SHORT["obf"], C.RULE_COLORS["obf"], "solid", "lines+markers")
    if "msprt" in show and msprt_curve is not None:
        both(list(msprt_curve[0]), list(msprt_curve[1]), C.RULE_SHORT["msprt"], C.RULE_COLORS["msprt"], "dashdot")
    for rule, (window, value) in stops.items():
        if window is not None:
            fig.add_trace(go.Scatter(x=[window], y=[value], mode="markers", name=f"Stopp {C.RULE_SHORT[rule]}",
                                     marker=dict(color=C.RULE_COLORS[rule], size=13, symbol="x", line=dict(width=2))))
    fig.update_yaxes(range=[-6, 6])
    return _finish(fig, "Teststatistik nach jedem Fenster und die Grenzen der Regeln", "Fenster je Arm", "z-Statistik (B minus A)")


def build_rates(rows, aa):
    """Anteil mit Befund je Regel: bei A/A der Fehlalarm, sonst die Teststärke."""
    rules = list(C.RULES)
    y = [100 * rows[r]["reject"] for r in rules]
    fig = go.Figure(go.Bar(x=[C.RULE_SHORT[r] for r in rules], y=y, marker_color=[C.RULE_COLORS[r] for r in rules],
                           text=[f"{v:.1f} %".replace(".", ",") for v in y], textposition="outside"))
    if aa:
        fig.add_hline(y=5, line=dict(color="gray", dash="dot"), annotation_text="Soll 5 %", annotation_position="top right")
    fig.update_yaxes(range=[0, max(100, max(y) * 1.1)] if not aa else [0, max(35, max(y) * 1.2)])
    return _finish(fig, "Fehlalarmrate je Regel" if aa else "Teststärke je Regel", None, "Anteil der Experimente in %", height=320)


def build_blicke(curves, aa_target=True):
    """Fehlalarmrate gegen die Zahl der Blicke, eine Linie je Regel. curves: {Regel: [(k, rate), ...]}."""
    fig = go.Figure()
    for rule, pts in curves.items():
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[100 * p[1] for p in pts], mode="lines+markers",
                                 name=C.RULE_SHORT[rule], line=dict(color=C.RULE_COLORS[rule], width=3 if rule == "naiv" else 2)))
    fig.add_hline(y=5, line=dict(color="gray", dash="dot"), annotation_text="Soll 5 %", annotation_position="bottom right")
    return _finish(fig, "Ohne Wirkung: Fehlalarmrate gegen die Zahl der Blicke", "Zahl der Zwischenblicke", "Anteil in %")


def build_exaggeration(rows):
    """Überschätzung der Wirkung bei Stopp mit Befund je Regel (Schätzung / wahre Wirkung)."""
    rules = [r for r in C.RULES if np.isfinite(rows[r]["exaggeration"])]
    y = [rows[r]["exaggeration"] for r in rules]
    fig = go.Figure(go.Bar(x=[C.RULE_SHORT[r] for r in rules], y=y, marker_color=[C.RULE_COLORS[r] for r in rules],
                           text=[f"{v:.2f}".replace(".", ",") + "×" for v in y], textposition="outside"))
    fig.add_hline(y=1.0, line=dict(color="gray", dash="dot"), annotation_text="wahre Wirkung", annotation_position="bottom right")
    fig.update_yaxes(range=[0, max(2.0, max(y) * 1.2)])
    return _finish(fig, "Wie groß die Schätzung beim Stopp ist (Vielfaches der Wahrheit)", None, "Schätzung / Wahrheit", height=320)
