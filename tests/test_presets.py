"""Presets: Vollständigkeit, gültige Werte, Einrasten der Permalink-Werte."""

import seq_constants as C
import seq_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_ORDER)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_on_the_grid():
    for preset in C.PRESETS.values():
        assert C.Z_MIN <= preset["z"] <= C.Z_MAX and preset["z"] in C.Z_GRID
        assert C.N_MIN <= preset["n"] <= C.N_MAX and (preset["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.K_MIN <= preset["k"] <= C.K_MAX and preset["k"] <= preset["n"]
        assert 0 <= preset["seed"] <= C.SEED_MAX


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_rules_have_labels_and_colors():
    assert set(C.RULE_LABELS) == set(C.RULE_SHORT) == set(C.RULE_COLORS) == set(C.RULES)


def test_permalink_values_are_clamped_and_snapped_to_the_slider_grid():
    z_spec = P.SETTING_SPECS["z_slider"]
    assert P.clamp_to_spec(z_spec, 1.47) == 1.5 and P.clamp_to_spec(z_spec, 9.0) == 2.0
    n_spec = P.SETTING_SPECS["n_slider"]
    assert P.clamp_to_spec(n_spec, 33) == 30 and P.clamp_to_spec(n_spec, 5) == 20 and P.clamp_to_spec(n_spec, 9999) == 200
    k_spec = P.SETTING_SPECS["k_slider"]
    assert P.clamp_to_spec(k_spec, 0) == 1 and P.clamp_to_spec(k_spec, 99) == 30
