"""Orakel-Tests: der Kern gegen einen unabhängigen Rechenweg (ORACLE-PLAYBOOK §2).

1. Gruppensequentielle Grenzen (rekursive Integration) gegen veröffentlichte Werte und gegen Monte-Carlo-Pfade.
2. Fehlalarmrate des naiven Hinsehens gegen die Integration.
3. mSPRT-Quotient gegen numerische Integration der Mischung.
4. Welch-Test nach jedem Fenster gegen scipy.stats.ttest_ind.
5. Gültigkeit des mSPRT (Ville-Schranke) auf Normaldaten."""

import numpy as np
import pytest
from scipy import integrate, stats

import seq_boundaries as B
import seq_constants as C
import seq_evaluation as E
from seq_cumulative import cumulative_welch


def test_known_group_sequential_constants():
    assert B.pocock_const(1) == pytest.approx(1.95996, abs=1e-3)
    assert B.pocock_const(2) == pytest.approx(2.178, abs=2e-3)       # veröffentlicht: 2,178
    assert B.pocock_const(5) == pytest.approx(2.413, abs=2e-3)       # veröffentlicht: 2,413
    c, _ = B.obf_const(5)
    assert c == pytest.approx(2.040, abs=2e-3)                       # veröffentlicht: 2,040
    c10, _ = B.obf_const(10)
    assert c10 == pytest.approx(2.087, abs=3e-3)


def test_naive_repeated_looks_inflate_the_false_alarm_rate_as_published():
    # fünf gleichmäßige Blicke bei 1,96: etwa 14 % (Armitage, McPherson und Rowe)
    assert B.total_cross(np.full(5, 1.959964)) == pytest.approx(0.1417, abs=0.003)
    assert B.total_cross(np.full(2, 1.959964)) == pytest.approx(0.0831, abs=0.003)
    assert B.total_cross(np.full(10, 1.959964)) == pytest.approx(0.1934, abs=0.004)


def test_boundaries_hold_against_monte_carlo_paths():
    rng = np.random.default_rng(5)
    K, reps = 5, 200_000
    s = np.cumsum(rng.standard_normal((reps, K)), axis=1)
    z = np.abs(s) / np.sqrt(np.arange(1, K + 1))
    for name, c in (("pocock", np.full(K, B.pocock_const(K))), ("obf", B.obf_const(K)[1]), ("ld", B.lan_demets(K))):
        crossed = (z >= c[None, :]).any(axis=1).mean()
        assert crossed == pytest.approx(0.05, abs=0.0035), (name, crossed)


def test_lan_demets_spends_the_planned_budget():
    K = 8
    c = B.lan_demets(K)
    probs = B.crossing_probs(c)
    t = np.arange(1, K + 1) / K
    cum = np.cumsum(probs)
    target = B.spend_obf(t)
    target[-1] = 0.05
    assert cum[-1] == pytest.approx(0.05, abs=1e-3)
    mask = np.isfinite(c)
    assert np.all(np.abs(cum[mask] - target[mask]) < 2e-3)


def test_precomputed_boundaries_equal_a_fresh_computation():
    pre = B.load_boundaries()
    assert sorted(pre) == list(range(C.K_MIN, C.K_MAX + 1))
    for k in (3, 5):
        assert pre[k]["pocock"] == pytest.approx(B.pocock_const(k), abs=2e-5)
        assert np.allclose(pre[k]["obf"], B.obf_const(k)[1], atol=2e-5)
        assert np.allclose(pre[k]["ld"], B.lan_demets(k), atol=2e-5)


def test_msprt_ratio_equals_the_numerical_integral_of_the_mixture():
    diff, v, tau = 0.021, 0.0002, 0.02
    lam = float(E.msprt_lambda(np.array(diff), np.array(v * 5), np.array(v * 5), np.array(10.0), tau))   # v = (va + vb) / m
    # Mischung: Integral über theta von N(diff; theta, v) / N(diff; 0, v) * N(theta; 0, tau^2)
    f = lambda th: stats.norm.pdf(diff, th, np.sqrt(v)) / stats.norm.pdf(diff, 0, np.sqrt(v)) * stats.norm.pdf(th, 0, tau)
    ref, _ = integrate.quad(f, -1, 1, points=[0.0, diff])
    assert lam == pytest.approx(ref, rel=1e-6)


def test_cumulative_welch_equals_scipy_on_every_prefix():
    rng = np.random.default_rng(8)
    a, b = rng.normal(0.9, 0.05, (3, 25)), rng.normal(0.93, 0.04, (3, 25))
    w = cumulative_welch(a, b)
    for r in range(3):
        for m in (2, 5, 13, 25):
            ref = stats.ttest_ind(b[r, :m], a[r, :m], equal_var=False)
            assert w["p"][r, m - 1] == pytest.approx(ref.pvalue, rel=1e-7, abs=1e-12)
            assert w["t"][r, m - 1] == pytest.approx(ref.statistic, rel=1e-7)


def test_msprt_keeps_the_false_alarm_rate_below_five_percent_on_normal_data():
    rng = np.random.default_rng(9)
    reps, n = 3000, 120
    a = rng.normal(0.93, 0.05, (reps, n))
    b = rng.normal(0.93, 0.05, (reps, n))
    w = cumulative_welch(a, b)
    lam = E.msprt_lambda(w["diff"], w["var_a"], w["var_b"], w["m"])
    lam[:, : C.MSPRT_BURN - 1] = 1.0
    assert float((lam >= 1 / C.ALPHA).any(axis=1).mean()) < 0.05
