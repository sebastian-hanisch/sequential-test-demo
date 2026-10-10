"""Konstanten: Vehikel, Regler-Grenzen, Farben, Presets. Das Vehikel (ein Standort-Fenster) ist das der ganzen Linie."""

# Vehikel: ein Standort, tägliche Nachbestellung (Order-up-to), Wiederbeschaffungszeit 2 Tage, Zentrallager unbegrenzt.
BASE_MEAN = 20.0          # mittlere Tagesnachfrage des Standorts (Stück)
DISP = 5                  # Dispersion der negativ-binomialen Tagesnachfrage (ganzzahlig: Gamma(5) = Summe von 5 Exponentialzahlen)
LEAD = 2                  # Wiederbeschaffungszeit in Tagen
BURN_DAYS = 28            # Einschwingtage vor dem Auswertefenster (gleiche Regel wie im Fenster)
WINDOW_DAYS = 28          # Auswertefenster: vier Wochen = ein "Standort-Fenster" = eine Einheit des Experiments
Z_A = 1.0                 # Sicherheitsfaktor der bestehenden Regel A

Z_MIN, Z_MAX, DEFAULT_Z = 1.0, 2.0, 1.4      # Regel B: Sicherheitsfaktor (1,0 = identisch mit A = A/A-Test)
Z_STEP = 0.1
Z_GRID = tuple(round(Z_MIN + i * Z_STEP, 1) for i in range(int(round((Z_MAX - Z_MIN) / Z_STEP)) + 1))
POOL_SIZE = 10000         # Einheiten je Pool (je Sicherheitsfaktor)
ALPHA = 0.05

# Stück 4: Zwischenblicke
N_MIN, N_MAX, DEFAULT_N, N_STEP = 20, 200, 30, 10        # geplanter Höchstumfang je Arm (Fenster)
K_MIN, K_MAX, DEFAULT_K = 1, 30, 5                       # Zahl der gleichmäßig verteilten Zwischenblicke
MSPRT_TAU = 0.02          # Mischungsbreite des mSPRT: kleinste lohnende Wirkung (2 Prozentpunkte)
MSPRT_BURN = 10           # Fenster je Arm, bevor der mSPRT entscheiden darf (Plug-in-Varianz braucht Anlauf)
BAYES_LEVEL = 0.95        # Entscheidungsregel: stoppe, sobald P(B besser) über diesem Wert liegt
SEED_MAX, DEFAULT_SEED = 999999, 35
LIVE_REPS = 1000          # Wiederholungen der Live-Auswertung (aus dem Pool gezogen)

RULES = ("naiv", "pocock", "obf", "ld", "msprt", "bayes")
RULE_LABELS = {
    "naiv": "Naiv (p unter 0,05 bei jedem Blick)",
    "pocock": "Pocock (gleiche Grenze)",
    "obf": "O'Brien-Fleming (streng am Anfang)",
    "ld": "Lan-DeMets (Ausgabefunktion)",
    "msprt": "mSPRT (jederzeit gültig)",
    "bayes": "Bayes (P(B besser) über 95 %)",
}
RULE_SHORT = {"naiv": "Naiv", "pocock": "Pocock", "obf": "O'Brien-Fleming", "ld": "Lan-DeMets", "msprt": "mSPRT", "bayes": "Bayes"}

COLOR_A = "#4c78a8"
COLOR_B = "#2F6B65"
COLOR_TRUTH = "#d95f02"
COLOR_MISS = "#c0392b"
COLOR_OK = "rgba(120,120,120,0.55)"
RULE_COLORS = {"naiv": "#c0392b", "pocock": "#7570b3", "obf": "#2F6B65", "ld": "#4c78a8", "msprt": "#d95f02", "bayes": "#8c564b"}

PRESETS = {
    "Fünf Blicke, ein echter Effekt (Voreinstellung)": {"z": DEFAULT_Z, "n": DEFAULT_N, "k": DEFAULT_K, "seed": DEFAULT_SEED},
    "Fünf Blicke, kein Effekt (A/A)": {"z": 1.0, "n": DEFAULT_N, "k": DEFAULT_K, "seed": DEFAULT_SEED},
    "Täglich hinsehen, kein Effekt": {"z": 1.0, "n": 100, "k": 30, "seed": DEFAULT_SEED},
    "Kleiner Effekt, viele Blicke": {"z": 1.2, "n": 100, "k": 10, "seed": DEFAULT_SEED},
}
PRESET_ORDER = tuple(PRESETS)
PRESET_HELP = {
    "Fünf Blicke, ein echter Effekt (Voreinstellung)": "Regel B mit Sicherheitsfaktor 1,4, höchstens 30 Fenster je Arm, fünf gleichmäßige Zwischenblicke.",
    "Fünf Blicke, kein Effekt (A/A)": "Beide Regeln gleich: jeder Stopp mit Befund ist ein Fehlalarm; wie oft er vorkommt, hängt an der Regel.",
    "Täglich hinsehen, kein Effekt": "100 Fenster je Arm, 30 Zwischenblicke und keine Wirkung: so entstehen die meisten Fehlalarme.",
    "Kleiner Effekt, viele Blicke": "Sicherheitsfaktor 1,2, höchstens 100 Fenster je Arm, zehn Blicke: früh stoppen heißt überschätzen.",
}


def fmt_pct(x, digits=1):
    """Anteil (0..1) als Prozentzahl mit Dezimalkomma, z. B. 0,934 -> '93,4 %'."""
    return f"{100 * x:.{digits}f}".replace(".", ",") + " %"


def fmt_pp(x, digits=2):
    """Differenz in Prozentpunkten (aus Anteilen 0..1) mit Vorzeichen und Dezimalkomma."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def fmt_pkt(x, digits=1):
    """Kurzform für enge Kacheln: '+4,0 Pkt.' (Differenz in Prozentpunkten, aus Anteilen 0..1)."""
    return f"{100 * x:+.{digits}f}".replace(".", ",") + " Pkt."


def fmt_num(x, digits=1):
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_int(x):
    """Ganzzahl mit Leerzeichen als Tausendertrenner."""
    return f"{int(x):,}".replace(",", " ")
