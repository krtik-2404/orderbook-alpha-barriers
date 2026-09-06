"""FI-2010 loading and feature construction.

The dataset itself is ~900 MB and gitignored, so nothing here touches it.
These test the parts that would silently corrupt the result if they were
wrong: instrument segmentation, causality of the features across a boundary,
and the fit/validation split.

The failure these are really guarding against is a return computed across a
stock boundary. FI-2010 concatenates five instruments with no delimiter, and
DecPre normalisation scales each by its own power of ten - so a boundary looks
like a 2,000 bp move. Feed that to a returns-only model and it learns a
phantom, which is exactly the kind of bug that inflates a result instead of
breaking it.
"""

import numpy as np
import pytest

from fi2010 import (FI_TO_PROJECT, HORIZONS, PUBLISHED, RETURN_LAGS, WARMUP,
                    build_features, segment_bounds, split_fit_val, valid_rows,
                    weighted_f1)
from lobforge.training import evaluate


def two_stock_mid(n=500, jump=10.0):
    """Two instruments spliced together, the second at a wildly different
    price level - which is what DecPre normalisation actually produces."""
    rng = np.random.default_rng(0)
    a = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    b = a[-1] * jump + np.cumsum(rng.normal(0, 0.01, n))
    return np.concatenate([a, b])


# ------------------------------------------------------------ segmentation

def test_segment_bounds_finds_the_instrument_boundary():
    mid = two_stock_mid(500)
    assert segment_bounds(mid, [], 100.0) == [[0, 500], [500, 1000]]


def test_segment_bounds_ignores_ordinary_moves():
    """A continuous series is one segment - the threshold must not fire on
    genuine volatility."""
    rng = np.random.default_rng(1)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.05, 2000))
    assert segment_bounds(mid, [], 100.0) == [[0, 2000]]


def test_segment_bounds_honours_known_file_edges():
    """Two test files joined end to end need not show a price jump, but the
    join is still not a continuous series."""
    rng = np.random.default_rng(2)
    mid = 100.0 + np.cumsum(rng.normal(0, 0.01, 400))
    assert segment_bounds(mid, [200], 100.0) == [[0, 200], [200, 400]]


# --------------------------------------------------------------- features

def test_returns_never_span_a_segment_boundary():
    """THE bug this file exists for. The first row of the second instrument
    has no history, so every return must read zero - not the 900% jump."""
    mid = two_stock_mid(500, jump=10.0)
    seg = segment_bounds(mid, [], 100.0)
    f = build_features(mid, seg)
    assert np.allclose(f[500], 0.0), f[500]
    # and nothing anywhere may carry the boundary's magnitude
    assert np.abs(f).max() < 1_000, np.abs(f).max()


def test_warmup_rows_are_zero_then_features_appear():
    mid = two_stock_mid(500)
    f = build_features(mid, segment_bounds(mid, [], 100.0))
    longest = max(RETURN_LAGS)
    assert np.allclose(f[0], 0.0)
    assert not np.allclose(f[longest + 5], 0.0)


def test_features_are_causal():
    """Changing the future must not change a feature computed now."""
    mid = two_stock_mid(500)
    seg = [[0, 1000]]
    base = build_features(mid, seg)
    tampered = mid.copy()
    tampered[600:] *= 1.05
    after = build_features(tampered, seg)
    assert np.allclose(base[:600], after[:600])


def test_realised_volatility_is_nonnegative_and_rises_with_noise():
    rng = np.random.default_rng(3)
    calm = 100.0 + np.cumsum(rng.normal(0, 0.001, 1000))
    wild = 100.0 + np.cumsum(rng.normal(0, 0.10, 1000))
    fc = build_features(calm, [[0, 1000]])
    fw = build_features(wild, [[0, 1000]])
    rv = len(RETURN_LAGS)                       # first rv column
    assert (fc[:, rv:] >= 0).all() and (fw[:, rv:] >= 0).all()
    assert fw[500:, rv].mean() > fc[500:, rv].mean()


# ------------------------------------------------------------------ splits

def test_valid_rows_drops_warmup_and_label_tail():
    seg = [[0, 1000]]
    idx = valid_rows(seg, 1000, max_k=100)
    assert idx.min() == WARMUP
    assert idx.max() == 1000 - 100 - 1


def test_valid_rows_skips_segments_too_short_to_use():
    idx = valid_rows([[0, 50], [50, 1000]], 1000, max_k=100)
    assert (idx >= 50).all()


def test_fit_and_val_are_disjoint_and_purged():
    seg = [[0, 1000], [1000, 2000]]
    fit, val = split_fit_val(seg, max_k=100, val_frac=0.1)
    assert not set(fit.tolist()) & set(val.tolist())
    for a, b in seg:
        f = fit[(fit >= a) & (fit < b)]
        v = val[(val >= a) & (val < b)]
        assert len(f) and len(v), "every instrument must appear in both"
        assert v.min() - f.max() > 100, "purge gap smaller than the horizon"


def test_val_is_drawn_from_every_instrument():
    """The bug found while building this: the obvious 'last 10% of train'
    split lands entirely inside the final stock, so early stopping selects on
    one instrument and fires almost immediately."""
    seg = [[0, 1000], [1000, 2000], [2000, 3000]]
    _, val = split_fit_val(seg, max_k=100, val_frac=0.1)
    for a, b in seg:
        assert ((val >= a) & (val < b)).any(), f"no val rows from [{a},{b})"


def test_val_rows_are_later_than_fit_rows_within_an_instrument():
    seg = [[0, 1000]]
    fit, val = split_fit_val(seg, max_k=100, val_frac=0.2)
    assert fit.max() < val.min()


# ------------------------------------------------------------- conventions

def test_label_mapping_is_a_bijection_onto_project_classes():
    """FI-2010 is 1=up, 2=stationary, 3=down; the project is (down, flat, up).
    Verified against the data itself: class 1 carries a mean smoothed label of
    +2.5 bp and class 3 -2.4 bp on Test_CF_9."""
    assert sorted(FI_TO_PROJECT) == [1, 2, 3]
    assert sorted(FI_TO_PROJECT.values()) == [0, 1, 2]
    assert FI_TO_PROJECT[1] == 2 and FI_TO_PROJECT[3] == 0


def test_published_numbers_only_cover_horizons_the_paper_reports():
    """Blanks stay blank. An interpolated 'published' number would be a
    fabricated citation."""
    assert set(PUBLISHED["setup2"]) == {10, 20, 50}
    assert set(PUBLISHED["setup1"]) == {10, 50, 100}
    for tbl in PUBLISHED.values():
        assert all(40.0 < v < 100.0 for v in tbl.values())
        assert set(tbl) <= set(HORIZONS)


# ------------------------------------------------------------------ metrics

def test_weighted_f1_differs_from_macro_on_imbalanced_classes():
    """The paper reports weighted F1; the project reports macro. On a skewed
    label set they are not the same number, which is the whole reason both are
    printed."""
    y_true = np.array([1] * 90 + [0] * 5 + [2] * 5)
    y_pred = np.array([1] * 100)
    m = evaluate(y_true, y_pred)
    assert weighted_f1(m) > m.macro_f1
    assert weighted_f1(m) == pytest.approx(0.9 * (2 * 0.9 / 1.9))


def test_weighted_f1_equals_macro_when_balanced_and_perfect():
    y = np.array([0, 1, 2] * 10)
    m = evaluate(y, y)
    assert weighted_f1(m) == pytest.approx(1.0)
    assert m.macro_f1 == pytest.approx(1.0)
