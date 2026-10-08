"""The textbook model must reproduce the course notebook to the printed digits."""
import numpy as np
import pytest

from cdsengine import CalibrationError, bootstrap_simple, df_log_linear, hazards_from_survival


def test_wfc_recovery_40_matches_notebook():
    P = bootstrap_simple(range(6), [1, .97, .94, .92, .89, .86],
                         [0, .005, .0077, .0094, .01095, .0125], 0.40)
    assert P[1:] == pytest.approx([0.991736, 0.974623, 0.953894, 0.928942, 0.899443], abs=5e-7)


def _panel():
    t = np.arange(8)
    df = df_log_linear([1, 2, 3, 5, 7], [.97, .94, .92, .86, .81], t)
    wfc = np.interp(t, [1, 2, 3, 5, 7], [50, 77, 94, 125, 133]) / 1e4
    ccmo = np.interp(t, [1, 2, 3, 5, 7], [751, 1164, 1874, 4156, 6083]) / 1e4
    wfc[0] = ccmo[0] = 0
    return t, df, wfc, ccmo


def test_log_linear_discount_factors():
    _, df, _, _ = _panel()
    assert df[4] == pytest.approx(0.889494, abs=5e-7)
    assert df[6] == pytest.approx(0.834626, abs=5e-7)


def test_wfc_recovery_50_matches_notebook():
    t, df, wfc, _ = _panel()
    P = bootstrap_simple(t, df, wfc, 0.50)
    assert 100 * np.array(P[1:]) == pytest.approx(
        [99.009901, 96.964924, 94.496611, 91.536605, 88.053648, 85.436020, 82.738795], abs=5e-6)
    lam = hazards_from_survival(t, P)
    assert lam[1:] == pytest.approx([0.0100, 0.0209, 0.0258, 0.0318, 0.0388, 0.0302, 0.0321], abs=5e-5)


def test_ccmo_negative_survival_is_reproduced_then_rejected():
    t, df, _, ccmo = _panel()
    raw = bootstrap_simple(t, df, ccmo, 0.10, strict=False)
    assert 100 * np.array(raw[1:]) == pytest.approx(
        [92.298226, 77.857959, 52.891991, 17.052397, -11.180822, -24.862936, -30.307031], abs=5e-6)
    with pytest.raises(CalibrationError, match="t=5"):
        bootstrap_simple(t, df, ccmo, 0.10)
