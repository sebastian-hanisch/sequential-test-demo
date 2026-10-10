"""Messreihe vor dem Text: Zwischenblicke, sechs Regeln, auf dem Vehikel (Stück 4)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seq_constants as C  # noqa: E402
import seq_evaluation as E  # noqa: E402

pools = E.load_pools()
print("Anteil mit Befund je Regel (A/A: Fehlalarm; sonst Teststärke) und mittlere Fenster je Arm bis Stopp (ASN)")
for z, n in ((1.0, 30), (1.4, 30), (1.0, 100), (1.2, 100)):
    for k in (1, 2, 5, 10, 30):
        if k > n:
            continue
        s = E.summarize(pools, z, n, k, reps=3000)
        row = f"z={z} n={n:3d} K={k:2d} (wahr {100*s['truth']:+.2f}): "
        for rule in C.RULES:
            r = s["rows"][rule]
            row += f"{C.RULE_SHORT[rule][:5]} {100*r['reject']:5.1f}% asn {r['mean_stop']:5.1f} | "
        print(row)
print("\nUeberschaetzung bei Stopp (Mittel der Schaetzung unter den Befunden / wahre Wirkung), z=1.2 n=100:")
for k in (1, 5, 10, 30):
    s = E.summarize(pools, 1.2, 100, k, reps=3000)
    print(f"  K={k:2d}: " + "  ".join(f"{C.RULE_SHORT[r][:5]} {s['rows'][r]['exaggeration']:.2f}" for r in C.RULES))
