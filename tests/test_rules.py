"""Die sechs Regeln und die Pfade: Handrechnung an Mini-Instanzen, Eigenschaften, Zweig-Tests."""

import numpy as np
import pytest

import seq_boundaries as B
import seq_constants as C
import seq_evaluation as E


def test_look_positions_are_equally_spaced_and_end_at_n():
    assert list(E.look_positions(100, 5)) == [20, 40, 60, 80, 100]
    assert list(E.look_positions(30, 1)) == [30]
    assert list(E.look_positions(10, 30)) == list(range(2, 11))       # nie weniger als 2 Fenster je Arm
    assert E.look_positions(100, 10)[-1] == 100


def test_msprt_ratio_by_hand():
    # m = 10, beide Varianzen 0,001: v = 0,002 / 10 = 0,0002; tau^2 = 0,0004; Differenz 0,02
    # Lambda = sqrt(0,0002 / 0,0006) * exp(0,0004 * 0,0004 / (2 * 0,0002 * 0,0006)) = 0,57735 * exp(2/3) = 1,1245
    lam = E.msprt_lambda(np.array(0.02), np.array(0.001), np.array(0.001), np.array(10.0))
    assert float(lam) == pytest.approx(0.57735 * np.exp(2 / 3), rel=1e-4)
    # keine Differenz: Quotient unter 1 (Mischung bestraft die Breite)
    assert float(E.msprt_lambda(np.array(0.0), np.array(0.001), np.array(0.001), np.array(10.0))) < 1.0


def test_z_from_p_is_the_normal_equivalent():
    assert float(E.z_from_p(np.array(0.05))) == pytest.approx(1.95996, abs=1e-4)
    assert float(E.z_from_p(np.array(1.0))) == pytest.approx(0.0, abs=1e-9)


def test_a_clear_effect_stops_every_rule_at_the_first_look():
    n = 40
    a = np.tile(np.array([0.90, 0.91, 0.89, 0.92] * 10), (1, 1))
    b = a + 0.30                                                       # riesiger, gleichmäßiger Vorsprung
    res = E.analyze(a, b.clip(max=1.5), 4, B.load_boundaries())
    for rule in ("naiv", "pocock", "obf", "ld", "msprt", "bayes"):
        assert bool(res[rule]["reject"][0]), rule
    assert int(res["naiv"]["stop"][0]) == 10                           # erster Blick bei n/4
    assert int(res["msprt"]["stop"][0]) == C.MSPRT_BURN                # darf erst ab dem Anlauf entscheiden
    assert n == 40


def test_no_difference_no_finding_and_runs_to_the_end():
    rng = np.random.default_rng(1)
    base = rng.normal(0.93, 0.01, 30)
    res = E.analyze(base[None, :], base[None, ::-1] * 1.0, 5, B.load_boundaries())   # gleiche Werte in anderer Reihenfolge
    for rule in C.RULES:
        assert int(res[rule]["stop"][0]) >= 6
    assert not bool(res["pocock"]["reject"][0])


def test_the_naive_rule_rejects_more_than_the_corrected_ones_without_effect(pools):
    s = E.summarize(pools, 1.0, 100, 10, reps=1500)
    naive = s["rows"]["naiv"]["reject"]
    for rule in ("pocock", "obf", "ld", "msprt"):
        assert s["rows"][rule]["reject"] < naive - 0.05, rule
    assert 0.14 < naive < 0.26
    assert s["rows"]["pocock"]["reject"] == pytest.approx(0.05, abs=0.025)


def test_one_look_makes_all_fixed_sample_rules_identical(pools):
    s = E.summarize(pools, 1.4, 30, 1, reps=800)
    r = [s["rows"][x]["reject"] for x in ("naiv", "pocock", "obf", "ld")]
    assert max(r) - min(r) < 1e-12
    assert s["rows"]["naiv"]["mean_stop"] == pytest.approx(30.0)


def test_early_stopping_saves_windows_but_exaggerates(pools):
    s = E.summarize(pools, 1.2, 100, 10, reps=1500)
    assert s["rows"]["obf"]["mean_stop"] < 90
    assert s["rows"]["naiv"]["mean_stop"] < s["rows"]["obf"]["mean_stop"] < 100
    assert s["rows"]["naiv"]["exaggeration"] > 1.25
    assert s["rows"]["obf"]["exaggeration"] > 1.05


def test_each_rule_differs_from_the_naive_rule_somewhere(pools):
    """Zweig-Test: keine Regel ist ein toter Zweig; jede weicht auf mindestens einer Einstellung von der naiven ab."""
    s = E.summarize(pools, 1.0, 100, 10, reps=800)
    for rule in ("pocock", "obf", "ld", "msprt", "bayes"):
        assert s["rows"][rule]["reject"] != s["rows"]["naiv"]["reject"] or s["rows"][rule]["mean_stop"] != s["rows"]["naiv"]["mean_stop"]


def test_summaries_are_deterministic(pools):
    a = E.summarize(pools, 1.4, 30, 5, reps=200)
    b = E.summarize(pools, 1.4, 30, 5, reps=200)
    assert a["rows"] == b["rows"]


def test_run_one_has_one_path_of_n_windows(pools):
    one = E.run_one(pools, {"z": 1.4, "n": 30, "k": 5, "seed": 3})
    assert len(one["a"]) == len(one["b"]) == 30
    assert one["res"]["path"]["z"].shape == (1, 30)
