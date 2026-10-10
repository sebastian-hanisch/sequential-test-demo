"""Jede Zahl des README (Abschnitt "Befunde") wird hier über dieselben Funktionen wie in der App nachgerechnet.

Die Messreihen sind deterministisch (SplitMix64-Pools und -Ziehungen); die Bänder sind Rundungs- und Plattformreserve."""

import numpy as np
import pytest

import seq_boundaries as B
import seq_evaluation as E

REPS = 3000


def pct(x):
    return 100 * x


def test_exact_constants_and_the_inflation_of_naive_looks():
    assert B.pocock_const(5) == pytest.approx(2.413, abs=0.002)
    assert B.obf_const(5)[0] == pytest.approx(2.040, abs=0.002)
    exact = {k: pct(B.total_cross(np.full(k, 1.959964))) for k in (2, 5, 10)}
    assert exact[2] == pytest.approx(8.3, abs=0.2)
    assert exact[5] == pytest.approx(14.2, abs=0.2)
    assert exact[10] == pytest.approx(19.3, abs=0.2)


def test_naive_peeking_without_effect_on_the_vehicle(pools):
    expected = {1: 4.8, 2: 9.2, 5: 14.4, 10: 19.6, 30: 26.4}
    for k, value in expected.items():
        s = E.summarize(pools, 1.0, 100, k, reps=REPS)
        assert pct(s["rows"]["naiv"]["reject"]) == pytest.approx(value, abs=1.2), k


def test_corrected_rules_keep_the_level(pools):
    for k in (5, 10, 30):
        rows = E.summarize(pools, 1.0, 100, k, reps=REPS)["rows"]
        for rule in ("pocock", "obf", "ld"):
            assert 3.8 <= pct(rows[rule]["reject"]) <= 6.2, (rule, k)
        assert pct(rows["msprt"]["reject"]) <= 3.5, k


def test_bayes_rule_without_effect_is_no_free_pass(pools):
    s1 = E.summarize(pools, 1.0, 100, 1, reps=REPS)["rows"]["bayes"]["reject"]
    s5 = E.summarize(pools, 1.0, 100, 5, reps=REPS)["rows"]["bayes"]["reject"]
    s30 = E.summarize(pools, 1.0, 100, 30, reps=REPS)["rows"]["bayes"]["reject"]
    assert pct(s1) == pytest.approx(5.7, abs=1.2)
    assert pct(s5) == pytest.approx(12.8, abs=1.5)
    assert pct(s30) == pytest.approx(25.9, abs=2.0)


def test_power_and_windows_with_an_effect_30_windows_five_looks(pools):
    fixed = E.summarize(pools, 1.4, 30, 1, reps=REPS)["rows"]["naiv"]["reject"]
    assert pct(fixed) == pytest.approx(85.1, abs=1.5)
    rows = E.summarize(pools, 1.4, 30, 5, reps=REPS)["rows"]
    assert pct(rows["obf"]["reject"]) == pytest.approx(84.8, abs=1.5)
    assert rows["obf"]["mean_stop"] == pytest.approx(23.6, abs=0.8)
    assert pct(rows["pocock"]["reject"]) == pytest.approx(76.6, abs=1.8)
    assert rows["pocock"]["mean_stop"] == pytest.approx(21.1, abs=0.8)
    assert pct(rows["naiv"]["reject"]) == pytest.approx(88.9, abs=1.5)
    assert rows["naiv"]["mean_stop"] == pytest.approx(16.8, abs=0.8)
    assert pct(rows["msprt"]["reject"]) == pytest.approx(56.9, abs=2.0)


def test_small_effect_100_windows_ten_percent_fewer_windows_for_almost_no_power(pools):
    rows = E.summarize(pools, 1.2, 100, 5, reps=REPS)["rows"]
    assert pct(E.summarize(pools, 1.2, 100, 1, reps=REPS)["rows"]["naiv"]["reject"]) == pytest.approx(81.3, abs=1.6)
    assert pct(rows["obf"]["reject"]) == pytest.approx(82.1, abs=1.6)
    assert rows["obf"]["mean_stop"] == pytest.approx(79.2, abs=1.5)
    assert pct(rows["pocock"]["reject"]) == pytest.approx(73.8, abs=2.0)
    assert rows["pocock"]["mean_stop"] == pytest.approx(69.6, abs=1.5)
    assert pct(rows["msprt"]["reject"]) == pytest.approx(57.1, abs=2.0)


def test_early_stopping_exaggerates(pools):
    fixed = E.summarize(pools, 1.2, 100, 1, reps=REPS)["rows"]["naiv"]["exaggeration"]
    assert fixed == pytest.approx(1.12, abs=0.04)
    r5 = E.summarize(pools, 1.2, 100, 5, reps=REPS)["rows"]
    assert r5["naiv"]["exaggeration"] == pytest.approx(1.36, abs=0.06)
    assert r5["obf"]["exaggeration"] == pytest.approx(1.22, abs=0.05)
    assert r5["pocock"]["exaggeration"] == pytest.approx(1.42, abs=0.06)
    assert r5["msprt"]["exaggeration"] == pytest.approx(1.6, abs=0.1)
    r30 = E.summarize(pools, 1.2, 100, 30, reps=REPS)["rows"]
    assert r30["naiv"]["exaggeration"] == pytest.approx(1.64, abs=0.08)
    assert r30["obf"]["exaggeration"] == pytest.approx(1.27, abs=0.06)
