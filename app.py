"""Zwischenblicke - darf man beim Experiment hinsehen? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Konzepte-Linie "Experimente und Wirkungsmessung": Im A/B-Test stand der Umfang vorher fest. Wer
zwischendurch hinsieht und bei p unter 0,05 stoppt, bekommt mehr Fehlalarme als versprochen. Sechs Entscheidungsregeln
(naiv, Pocock, O'Brien-Fleming, Lan-DeMets, mSPRT, Bayes) treten gegeneinander an, mit bekannter Wahrheit. Siehe README.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import seq_boundaries as B
import seq_constants as C
from seq_evaluation import load_pools, nearest_z, run_one, summarize, true_effect
from seq_presets import (apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)
from seq_visualization import build_blicke, build_exaggeration, build_path, build_rates, msprt_boundary

st.set_page_config(page_title="Zwischenblicke – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _pools():
    return load_pools()


@st.cache_data(show_spinner=False)
def _bounds():
    return B.load_boundaries()


@st.cache_data(show_spinner=False)
def _one(z, n, k, seed):
    return run_one(_pools(), {"z": z, "n": n, "k": k, "seed": seed}, _bounds())


@st.cache_data(show_spinner=False)
def _many(z, n, k):
    return summarize(_pools(), z, n, k, reps=C.LIVE_REPS, bounds=_bounds())


@st.cache_data(show_spinner=False)
def _curve(n):
    ks = [k for k in (1, 2, 3, 5, 10, 20, 30) if k <= n]
    curves = {rule: [] for rule in C.RULES}
    for k in ks:
        rows = summarize(_pools(), 1.0, n, k, reps=C.LIVE_REPS, bounds=_bounds(), tag=5)["rows"]
        for rule in C.RULES:
            curves[rule].append((k, rows[rule]["reject"]))
    return curves


st.title("👀 Zwischenblicke: darf man beim Experiment hinsehen?")
st.markdown(
    """
Im A/B-Test stand der Umfang vorher fest. In der Praxis sieht man **zwischendurch hin** und möchte stoppen, sobald das
Ergebnis „signifikant“ ist. Das hat einen Preis: **bei jedem Blick gibt es eine neue Chance auf einen Zufallstreffer**, und
die Chancen addieren sich. Hier treten sechs Entscheidungsregeln gegeneinander an: das **naive** Hinsehen, drei
**gruppensequentielle** Verfahren (Pocock, O'Brien-Fleming, Lan-DeMets), ein **jederzeit gültiger** Test (mSPRT) und eine
**Bayes'sche** Regel. Weil die wahre Wirkung bekannt ist, lässt sich messen, was jede Regel an Fehlalarmen, Teststärke,
Dauer und Überschätzung kostet.
"""
)
st.caption(
    "Viertes Stück der Linie „Experimente und Wirkungsmessung“, aufbauend auf "
    "[ab-test-demo](https://sebastianhanisch-ab-test-demo.streamlit.app/) und "
    "[test-power-demo](https://sebastianhanisch-test-power-demo.streamlit.app/) (gleiches Vehikel: Standort-Fenster, "
    "Servicegrad, Regel A gegen Regel B). Jedes Folgestück hebt eine der Annahmen unter „Wo die Annahmen enden“ auf."
)

with st.expander("So funktionieren die sechs Regeln", expanded=True):
    st.markdown(
        """
- **Aufbau:** Die Fenster kommen nacheinander. Es gibt höchstens n Fenster je Arm und **K gleichmäßig verteilte Blicke**
  (nach je n/K Fenstern). Bei jedem Blick wird der Welch-Test auf allen bisherigen Fenstern gerechnet.
- **Naiv:** Stopp, sobald p unter 0,05. Jeder Blick hat 5 % Fehlalarmchance; über viele Blicke wird daraus ein Vielfaches.
- **Pocock:** an jedem Blick dieselbe, strengere Grenze (so gewählt, dass die Gesamtrate 5 % bleibt).
- **O'Brien-Fleming:** am Anfang sehr streng, am Ende fast wie der feste Test; kostet fast keine Teststärke.
- **Lan-DeMets:** gibt das Fehlalarm-Budget nach einer Ausgabefunktion aus; Zahl und Zeitpunkt der Blicke dürfen sich ändern.
- **mSPRT:** ein „jederzeit gültiger“ Test, der nach jedem einzelnen Fenster erlaubt ist; Preis ist Teststärke. Er darf erst ab
  10 Fenstern je Arm entscheiden, weil die geschätzte Streuung vorher zu unsicher ist.
- **Bayes:** stoppt, sobald die Wahrscheinlichkeit „B ist besser“ über 95 % liegt (flache Vorannahme). Das klingt nach einem
  Freibrief für Zwischenblicke, ist es aber nicht.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESET_ORDER))
for i, name in enumerate(C.PRESET_ORDER):
    with preset_cols[i]:
        st.button(name, key=f"preset_{i}", width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])
st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Wahrheit**")
    z = st.slider("Sicherheitsfaktor z von Regel B", *bounds("z_slider"), step=C.Z_STEP, key="z_slider", format="%.1f",
                  help="Regel A fährt z = 1,0. Bei z = 1,0 sind beide Regeln gleich: ein A/A-Test, jeder Befund ist ein Fehlalarm.")
    st.markdown("**Experiment**")
    n = st.slider("Höchstens so viele Fenster je Arm", *bounds("n_slider"), step=C.N_STEP, key="n_slider",
                  help="Der geplante Höchstumfang; ohne Stopp wird bis hierher gemessen.")
    k = st.slider("Zahl der Zwischenblicke", *bounds("k_slider"), step=1, key="k_slider",
                  help="Gleichmäßig über die Messung verteilt; 1 heißt: nur am Ende ansehen.")
    seed = st.number_input("Zufalls-Seed", min_value=bounds("seed_input")[0], max_value=bounds("seed_input")[1], step=1,
                           key="seed_input", help="Bestimmt die Fenster des einen Experiments oben.")
    st.button("🎲 Neues Experiment würfeln", on_click=randomize_seed)

z, n, k, seed = nearest_z(float(z)), int(n), int(k), int(seed)
k = min(k, n)
sync_query_params({"z_slider": z, "n_slider": n, "k_slider": k, "seed_input": seed})

pools = _pools()
truth = true_effect(pools, z)
aa = z <= C.Z_A

# --- Ein Experiment --------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.markdown("## 🧪 Ein Experiment")
one = _one(z, n, k, seed)
res = one["res"]
path = res["path"]
st.caption(
    f"Höchstens {n} Fenster je Arm, {k} Zwischenblicke; Regel B mit z = {C.fmt_num(z)}"
    + (" (gleich wie A: es gibt nichts zu finden)" if aa else f", wahre Wirkung **{C.fmt_pp(truth)}**")
    + ". Gezeichnet ist die Teststatistik nach jedem Fenster; Kreuze markieren, wo eine Regel stoppt."
)
show = st.multiselect("Grenzen zeigen", options=["naiv", "pocock", "obf", "msprt"], default=["naiv", "pocock", "obf"],
                      format_func=lambda r: C.RULE_SHORT[r], key="show_bounds")
stops = {}
rows_md = ["| Regel | Ergebnis | Fenster je Arm | Schätzung beim Stopp |", "|---|---|---|---|"]
for rule in C.RULES:
    r = res[rule]
    rej, stop, est = bool(r["reject"][0]), int(r["stop"][0]), float(r["est"][0])
    value = float(path["z"][0][stop - 1]) if rej else None
    stops[rule] = (stop if rej else None, value)
    rows_md.append(
        f"| {C.RULE_LABELS[rule]} | {'**Befund**' if rej else 'kein Befund'} | {stop if rej else str(n) + ' (Ende)'} | "
        f"{C.fmt_pp(est)} |"
    )
m_all = np.arange(C.MSPRT_BURN, n + 1)
msprt_curve = (m_all, msprt_boundary(m_all, path["var_a"][0][m_all - 1], path["var_b"][0][m_all - 1])) if n >= C.MSPRT_BURN else None
st.plotly_chart(build_path(path["z"][0], path["pos"], n, path["bounds"], msprt_curve,
                           {r: v for r, v in stops.items() if r in show}, set(show)),
                width="stretch", key=f"path_{z}_{n}_{k}_{seed}_{'-'.join(show)}")
st.markdown("\n".join(rows_md))
early = [r for r in C.RULES if stops[r][0] is not None and stops[r][0] < n]
if aa and any(stops[r][0] is not None for r in C.RULES):
    found = [C.RULE_SHORT[r] for r in C.RULES if stops[r][0] is not None]
    st.warning("Beide Regeln sind gleich, trotzdem melden diese Regeln hier einen Befund: **" + ", ".join(found) +
               "**. Jeder davon ist ein Fehlalarm; wie oft das über viele Experimente passiert, zeigt der nächste Abschnitt.")
elif not aa and early:
    st.info("Mehrere Regeln stoppen früh. Das spart Fenster, aber die Schätzung beim Stopp ist im Mittel zu groß: nur "
            "zufällig günstige Pfade überschreiten früh die Grenze (siehe Überschätzung unten).")

# --- Viele Experimente -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("📐 Tausend Experimente: was kostet jede Regel?")
many = _many(z, n, k)
rows = many["rows"]
left, right = st.columns(2)
with left:
    st.plotly_chart(build_rates(rows, aa), width="stretch", key=f"rates_{z}_{n}_{k}")
with right:
    if aa:
        st.plotly_chart(build_blicke(_curve(n)), width="stretch", key=f"blicke_{n}")
    else:
        st.plotly_chart(build_exaggeration(rows), width="stretch", key=f"exag_{z}_{n}_{k}")
tbl = ["| Regel | Anteil mit Befund | Fenster je Arm im Mittel | Davon mit Befund | Falsches Vorzeichen |", "|---|---|---|---|---|"]
for rule in C.RULES:
    r = rows[rule]
    tbl.append(f"| {C.RULE_SHORT[rule]} | {C.fmt_pct(r['reject'])} | {C.fmt_num(r['mean_stop'])} | "
               f"{C.fmt_num(r['mean_stop_rejected']) if np.isfinite(r['mean_stop_rejected']) else '–'} | "
               f"{C.fmt_pct(r['wrong_sign'])} |")
st.markdown("\n".join(tbl))
if aa:
    st.error(
        f"**Ohne jede Wirkung meldet das naive Hinsehen bei {k} Blicken in {C.fmt_pct(rows['naiv']['reject'])} der "
        f"Experimente einen Befund** (versprochen waren 5 %). Pocock hält {C.fmt_pct(rows['pocock']['reject'])}, "
        f"O'Brien-Fleming {C.fmt_pct(rows['obf']['reject'])}, mSPRT {C.fmt_pct(rows['msprt']['reject'])}. Die "
        f"Bayes'sche Regel schlägt in {C.fmt_pct(rows['bayes']['reject'])} an: **sie ist kein Freibrief fürs Hinsehen**."
    )
else:
    full = _many(z, n, 1)["rows"]
    st.info(
        f"Mit **einem** Blick am Ende finden alle Regeln den Effekt in {C.fmt_pct(full['naiv']['reject'])} der "
        f"Experimente. Mit {k} Blicken erreicht O'Brien-Fleming {C.fmt_pct(rows['obf']['reject'])} bei im Mittel "
        f"{C.fmt_num(rows['obf']['mean_stop'])} statt {n} Fenstern, Pocock {C.fmt_pct(rows['pocock']['reject'])} bei "
        f"{C.fmt_num(rows['pocock']['mean_stop'])}. Der mSPRT findet nur {C.fmt_pct(rows['msprt']['reject'])}: sein Preis ist "
        f"Teststärke. Das naive Hinsehen wirkt mit {C.fmt_pct(rows['naiv']['reject'])} stärker, ist aber nur ein anderer Test mit "
        "höherer Fehlalarmrate (siehe Tabelle bei z = 1,0)."
    )

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Blicke sind vorher festgelegt** (Pocock, O'Brien-Fleming) | Wer nach Belieben hinsieht, braucht die Ausgabefunktion (Lan-DeMets) oder den mSPRT; die Konstanten gelten nur für die geplanten Blicke. | (dieses Stück) |
| **Früh stoppen ist nicht schlimm** | Die Schätzung beim frühen Stopp ist zu groß; wer nach dem Stopp den Effekt berichtet, berichtet zu viel. | (dieses Stück) |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen mit je mehreren Blicken vervielfachen die Chancen auf Zufallstreffer. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |
| **Die Einheiten sind unabhängig und gleichartig** | Verhältniskennzahlen und Cluster machen die Statistik bei jedem Blick unsicherer. | Verhältniskennzahlen und Cluster |
| **Die Streuung ist aus den Daten schätzbar** | Der mSPRT mit geschätzter Streuung ist ohne Anlauf (hier 10 Fenster) nicht gültig. | Varianzreduktion |
| **Man darf stoppen und die Zuteilung bleibt gültig** | Stoppen ändert nicht die Zuteilung, aber Verlust nach dem Stopp kann eine Gültigkeitsprüfung brauchen. | Gültigkeit der Zuteilung (A/A, Sample-Ratio-Mismatch) |
"""
)
st.caption(
    "Die Zahlen stammen aus dem erfundenen Vehikel dieses Stücks. Die Bayes'sche Regel nutzt eine flache Vorannahme und die "
    "Normalnäherung; mit einer passenden Vorannahme ändert sich das Bild (siehe README)."
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Fehlalarme beim Hinsehen.** Sei $Z_k$ die Teststatistik nach $k$ von $K$ Blicken. Bei gleichen Regeln ist
$P(\max_k |Z_k| \ge 1{,}96) > 0{,}05$; für $K = 5$ gleichmäßige Blicke sind es etwa 14 % (hier durch Integration
nachgerechnet: 14,2 %). Das Phänomen beschreiben Armitage, McPherson und Rowe (1969).

**Gruppensequentielle Grenzen.** Pocock: eine Konstante $c_K$ für alle Blicke mit $P(\exists k: |Z_k| \ge c_K) = \alpha$.
O'Brien-Fleming: $c_k = C_K \sqrt{K/k}$. Die Konstanten werden hier durch rekursive Integration der Dichte der Teilsumme
$S_k = \sum_{j \le k} \xi_j$ bestimmt (Normalzuwächse). Auf die Welch-Statistik werden die Grenzen über den $p$-Wert angewandt:
$z_k = \Phi^{-1}(1 - p_k/2)$.

**Lan-DeMets.** Eine Ausgabefunktion $\alpha^*(t)$ legt das bis zum Zeitpunkt $t$ ausgegebene Fehlalarm-Budget fest; hier
$\alpha^*(t) = 2 - 2\Phi(z_{\alpha/2}/\sqrt t)$ (O'Brien-Fleming-Typ). Die Grenze am Blick $k$ ergibt sich aus dem noch
verfügbaren Budget (Lan und DeMets 1983).

**mSPRT.** Mit Mischung $\theta \sim N(0, \tau^2)$ und Varianz der Differenz $V_n$:
$\Lambda_n = \sqrt{V_n/(V_n + \tau^2)}\,\exp\bigl(\tau^2 \hat\theta_n^2 / (2 V_n (V_n + \tau^2))\bigr)$. Der
jederzeit gültige p-Wert ist $\min(1, 1/\Lambda)$ in Laufmittel-Minimum; stoppe bei $\Lambda \ge 1/\alpha$ (Ville-Ungleichung).
Die Gültigkeit ist exakt für bekannte Normalvarianz; mit geschätzter Varianz hier empirisch geprüft (Anlauf von 10 Fenstern je Arm).

**Bayes.** Flache Vorannahme, Normalnäherung: $P(\Delta > 0 \mid \text{Daten}) = \Phi(\hat\Delta/\widehat{SE})$. Die Regel „stoppe bei
über 95 %“ hat bei dauerndem Hinsehen eine Fehlalarmrate, die mit der Zahl der Blicke wächst; die Kalibrierung gilt nur im
Mittel über die Vorannahme.

Implementiert in `seq_boundaries.py` (rekursive Integration, Konstanten), `seq_cumulative.py` (Welch nach jedem Fenster),
`seq_evaluation.py` (sechs Regeln), `generate_precomputed.py` (Pools, Grenzen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
