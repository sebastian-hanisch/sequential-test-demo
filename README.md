# Zwischenblicke: darf man beim Experiment hinsehen? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-sequential-test-demo.streamlit.app/)**

Interaktive Demo zum **Hinsehen während eines laufenden Experiments**. **Viertes Stück der Konzepte-Linie „Experimente und
Wirkungsmessung“** im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine
Learning), Kind des [Stücks zur Teststärke](https://github.com/sebastian-hanisch/test-power-demo): dort stand der Umfang vorher
fest, hier wird zwischendurch hingesehen und bei „signifikant“ gestoppt. Das Vehikel ist dasselbe: ein Standort mit täglicher
Nachbestellung, Regel A (Sicherheitsfaktor 1,0) gegen Regel B, Einheit ist ein Standort-Fenster von vier Wochen, Kennzahl der
Servicegrad.

## Kernfrage

Was kostet das Hinsehen an Fehlalarmen, und was kosten die Verfahren, die es erlauben, an Teststärke, Dauer und Genauigkeit
der Schätzung?

## Modell und Methodik

- **Aufbau:** Die Fenster kommen nacheinander (Pools des ersten Stücks, SplitMix64). Höchstens n Fenster je Arm, K gleichmäßig
  verteilte Blicke; bei jedem Blick der Welch-Test auf allen bisherigen Fenstern.
- **Sechs Regeln** (`seq_evaluation.analyze`):
  - *Naiv:* Stopp bei p unter 0,05.
  - *Pocock:* dieselbe Grenze an jedem Blick (Konstante so, dass die Gesamtrate 5 % ist).
  - *O'Brien-Fleming:* c_k = C·√(K/k).
  - *Lan-DeMets:* Ausgabefunktion 2 − 2Φ(z/√t) (O'Brien-Fleming-Typ).
  - *mSPRT:* jederzeit gültiger Test (Mischung N(0, τ²) mit τ = 2 Prozentpunkte, nach jedem Fenster, Anlauf 10 Fenster je Arm).
  - *Bayes:* stoppt bei P(B besser) über 95 % (flache Vorannahme, Normalnäherung).
- **Grenzen** (`seq_boundaries.py`): rekursive Integration der Dichte der Teilsumme (Faltung per FFT), nichts aus Tabellen;
  Konstanten für 1 bis 30 Blicke vorgerechnet (`precomputed_boundaries.json`). Auf die Welch-Statistik werden die Grenzen über den
  p-Wert angewandt (z = Φ⁻¹(1 − p/2)).
- **Messgrößen:** Anteil mit Befund (ohne Wirkung der Fehlalarm, sonst die Teststärke), mittlere Fenster je Arm bis zum Stopp
  (ohne Befund bis n), Schätzung beim Stopp.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (3 000 Experimente je Zelle, Fehler der Raten höchstens 0,8 Prozentpunkte).

| Frage | Befund |
|---|---|
| Wie viele Fehlalarme erzeugt naives Hinsehen? | Exakt (Integration): 2 Blicke **8,3 %**, 5 Blicke **14,2 %**, 10 Blicke **19,3 %**. Gemessen auf dem Vehikel bei 100 Fenstern je Arm ohne Wirkung: **9,2 %, 14,4 %, 19,6 %** und bei 30 Blicken **26,4 %** (versprochen 5 %). |
| Was halten die korrigierten Regeln? | Pocock, O'Brien-Fleming und Lan-DeMets bleiben bei 5, 10 und 30 Blicken zwischen **4 und 6 %**, der mSPRT bei **3 % oder weniger**. Die Konstanten für 5 Blicke: Pocock **2,413**, O'Brien-Fleming **2,040**. |
| Ist die Bayes'sche Regel ein Freibrief? | **Nein.** Mit flacher Vorannahme schlägt „P(B besser) über 95 %“ ohne Wirkung bei einem Blick in **5,7 %** an, bei 5 Blicken in **12,8 %**, bei 30 Blicken in **25,9 %**. |
| Was kostet die Korrektur an Teststärke? | Bei 30 Fenstern je Arm, 5 Blicken und Sicherheitsfaktor 1,4 (fest: **85,1 %**): O'Brien-Fleming **84,8 %**, Pocock **76,6 %**, mSPRT **56,9 %**. Das naive Hinsehen zeigt **88,9 %**, weil es mehr Fehlalarme mitzählt. |
| Was spart das Stoppen? | Im selben Fall im Mittel **23,6** Fenster je Arm statt 30 (O'Brien-Fleming) bzw. **21,1** (Pocock); das naive Hinsehen **16,8**. |
| Bei kleinem Effekt? | Sicherheitsfaktor 1,2, 100 Fenster, 5 Blicke (fest **81,3 %**): O'Brien-Fleming **82,1 %** bei **79,2** Fenstern im Mittel; Pocock **73,8 %** bei **69,6**; mSPRT **57,1 %**. |
| Wie groß ist die Schätzung beim frühen Stopp? | Im selben Fall ist die Schätzung unter den Befunden beim festen Test das **1,12-Fache** der Wahrheit, bei 5 Blicken beim naiven Hinsehen das **1,36-Fache**, bei Pocock das **1,42-Fache**, bei O'Brien-Fleming das **1,22-Fache**, beim mSPRT das **1,6-Fache**. Bei 30 Blicken: naiv **1,64**, O'Brien-Fleming **1,27**. |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen der Vorab-Messreihe (Gate) stimmen in der Größenordnung:** dort 13,2 % bei 5 Blicken, 30,4 % bei täglichem
  Hinsehen über 100 Tage. Die Konstanten sind selbst gerechnet und decken sich mit den üblicherweise genannten Werten (aus dem Gedächtnis, die Tabelle
  ist hier nicht nachgeschlagen); geprüft sind sie gegen Monte-Carlo-Pfade.
- **Der mSPRT braucht einen Anlauf.** Mit geschätzter Streuung schon ab 2 Fenstern ist er nicht gültig; die Messreihe vorab
  fand 6,6 bis 11,4 % Fehlalarm. Mit 10 Fenstern Anlauf bleibt er konservativ (hier 3 % oder weniger). Das ist ein Befund der
  Vorab-Messreihe auf dem Gate, hier nur mit Anlauf gemessen.
- **Die Überschätzung bei Stopp war im Plan nicht vorgesehen** und ist ein eigener Befund: früh stoppen heißt zu große
  Wirkung berichten, auch bei gültigen Verfahren (O'Brien-Fleming 1,22-fach bei 5 Blicken).
- **Die Bayes'sche Regel ist hier nur mit flacher Vorannahme und Normalnäherung gemessen.** Aussagen zu informativen
  Vorannahmen stehen nicht im Stück; die Vorab-Messreihe fand sie nur im Mittel über die Vorannahme kalibriert.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Blicke sind vorher festgelegt** (Pocock, O'Brien-Fleming) | Wer nach Belieben hinsieht, braucht die Ausgabefunktion (Lan-DeMets) oder den mSPRT. | (dieses Stück) |
| **Früh stoppen ist nicht schlimm** | Die Schätzung beim Stopp ist zu groß. | (dieses Stück) |
| **Ein Test, eine Kennzahl** | Mehrere Kennzahlen mit je mehreren Blicken vervielfachen die Zufallstreffer. | Viele Kennzahlen: Mehrfachtests und Gewinnerfluch |
| **Einheiten unabhängig und gleichartig** | Verhältniskennzahlen und Cluster machen die Statistik bei jedem Blick unsicherer. | Verhältniskennzahlen und Cluster |
| **Die Streuung ist aus den Daten schätzbar** | Ohne Anlauf ist der mSPRT nicht gültig. | Varianzreduktion |
| **Zuteilung bleibt gültig** | Verlust nach dem Stopp kann eine Gültigkeitsprüfung brauchen. | Gültigkeit der Zuteilung (A/A, Sample-Ratio-Mismatch) |

Die Zahlen gelten für dieses erfundene Vehikel mit unabhängigen Fenstern, gleichmäßig verteilten Blicken und einem Höchstumfang
je Arm; die Dauer ist in Fenstern gezählt, nicht in Wochen.

## Bezug zu OR

Ein Pilotbetrieb läuft nicht „bis zum Ende“: Entscheider sehen laufend hin und möchten bei klarem Ergebnis umstellen. Die
Demo zeigt, welche Regel das erlaubt, ohne die Fehlalarmrate zu sprengen, und was sie an Dauer spart.

## Verwandte Demos im Portfolio

- [ab-test-demo](https://github.com/sebastian-hanisch/ab-test-demo), [test-power-demo](https://github.com/sebastian-hanisch/test-power-demo),
  [sample-ratio-demo](https://github.com/sebastian-hanisch/sample-ratio-demo): die ersten Stücke der Linie.
- [bandit-demo](https://github.com/sebastian-hanisch/bandit-demo): Auswahlverfahren statt fester Zuteilung, ohne
  Teststatistik.
- [output-analysis-demo](https://github.com/sebastian-hanisch/output-analysis-demo): wann ein Simulationslauf lang genug ist.

## Verifikation

- **Grenzen** (`tests/test_oracle_sequential_test.py`): Konstanten gegen die üblicherweise genannten Werte (2,178; 2,413; 2,040; 2,087; aus dem
  Gedächtnis), gegen Monte-Carlo-Pfade, Lan-DeMets gibt das geplante Budget aus, Fehlalarm des naiven Hinsehens gegen die Integration
  (8,3 / 14,2 / 19,3 %).
- **Statistik:** Welch-Test nach jedem Fenster gegen `scipy.stats.ttest_ind` auf jedem Präfix; mSPRT-Quotient von Hand und
  gegen die numerische Integration der Mischung; Gültigkeit des mSPRT auf Normaldaten.
- **Regeln:** Mini-Instanzen von Hand (klarer Effekt stoppt alle Regeln beim ersten Blick; keine Differenz gibt keinen Befund);
  Zweig-Tests (jede Regel weicht von der naiven ab); bei einem Blick sind alle Festumfang-Regeln gleich.
- **Befunde:** jede Zahl dieses READMEs in `tests/test_claims.py`.
- **Presets, Oberfläche, Footer:** vollständig, Permalink eingerastet, AppTest-Rauchtest, Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: ein Experiment mit Grenzen, tausend Wiederholungen, Fehlalarm gegen Blicke, Grenzen |
| `seq_constants.py` | Vehikel-Konstanten, Regler-Grenzen, Regeln, Presets |
| `seq_rng.py`, `seq_vehicle.py`, `seq_formulas.py`, `seq_stats.py` | Zufall, Standort-Fenster, exakter Servicegrad, Welch-Test (aus dem ersten Stück) |
| `seq_boundaries.py` | gruppensequentielle Grenzen durch rekursive Integration, Lan-DeMets |
| `seq_cumulative.py` | Welch-Test nach jedem gesammelten Fenster, vektorisiert |
| `seq_evaluation.py` | Pfade, die sechs Regeln, Zusammenfassungen |
| `seq_presets.py`, `seq_visualization.py` | Permalink, Presets; Plotly-Abbildungen (Achsen gesperrt) |
| `generate_precomputed.py`, `precomputed_pools.json`, `precomputed_boundaries.json` | Pools und Grenzen |
| `tools/measure_stueck4.py` | die Messreihe vor dem Text |
| `tests/` | Vehikel, Orakel, Grenzen, Regeln, Presets, Befunde, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Einseitige Tests** und **Futility-Stopp** (früh aufgeben, wenn kein Effekt mehr möglich ist): nicht gebaut.
- **Ungleich verteilte Blicke und Korrekturen für Verhältniskennzahlen:** Lan-DeMets kann es, die Demo zeigt gleichmäßige Blicke.
- **Informative Vorannahmen für die Bayes'sche Regel:** nicht gebaut.
- **Konfidenzintervalle nach sequentiellem Stopp** (bias-adjustierte Schätzer): nur die Überschätzung ist gezeigt.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy (Verteilungsfunktionen, FFT).

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Experimente und Wirkungsmessung: A/B bis Logdaten](https://sebastianhanisch.net/konzepte-experimente-wirkungsmessung.html).
