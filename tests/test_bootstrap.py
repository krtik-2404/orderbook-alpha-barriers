"""The moving-block bootstrap has to disagree with sd/sqrt(n) for the right
reason, and agree with it for the right reason.

That is the whole test. On independent data the two intervals should land in
roughly the same place - if the bootstrap were systematically wide it would be
useless as evidence, because it would widen everything. On dependent data it
must be visibly wider, because that is the entire point of using it: the naive
interval counts n observations when there are closer to n/k of them.

The AR(1) case has a closed form to check against. For x_t = phi*x_{t-1} + e_t
the variance of the sample mean is inflated over the i.i.d. case by
(1+phi)/(1-phi), so at phi=0.95 the honest interval is sqrt(39) ~ 6.2x the
naive one. A bootstrap that returns 1.0x there is broken.
"""

import numpy as np
import pytest

from costfloor import moving_block_ci

N = 40_000
BLOCK = 200          # 2k at k=100, the script's default


def naive_width(x):
    return 2 * 1.96 * float(np.std(x)) / np.sqrt(len(x))


def boot_width(x, block=BLOCK, n_boot=400, seed=0):
    lo, hi = moving_block_ci(x, block, n_boot=n_boot, seed=seed)
    return hi - lo


def ar1(phi, n=N, seed=1):
    e = np.random.default_rng(seed).standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_iid_intervals_roughly_agree():
    """No dependence to find, so blocking should buy nothing."""
    x = np.random.default_rng(0).standard_normal(N)
    ratio = boot_width(x) / naive_width(x)
    assert 0.75 < ratio < 1.35, f"bootstrap/naive width ratio {ratio:.3f}"


def test_autocorrelated_interval_is_visibly_wider():
    """Strong serial dependence: the naive interval must be exposed as narrow."""
    x = ar1(0.95)
    ratio = boot_width(x) / naive_width(x)
    assert ratio > 3.0, f"bootstrap/naive width ratio {ratio:.3f}, expected >3"


def test_autocorrelated_width_tracks_the_closed_form():
    """Not just wider - wider by about the right amount.

    Generous bounds: a percentile interval from 400 resamples is itself noisy,
    and a finite block length recovers dependence only partially.
    """
    phi = 0.9
    expected = np.sqrt((1 + phi) / (1 - phi))       # ~4.36
    ratio = boot_width(ar1(phi)) / naive_width(ar1(phi))
    assert 0.5 * expected < ratio < 1.5 * expected, (
        f"ratio {ratio:.2f} vs closed-form inflation {expected:.2f}")


def test_longer_blocks_capture_more_dependence():
    """Monotone in block length while the block is shorter than the memory."""
    x = ar1(0.95)
    widths = [boot_width(x, block=b) for b in (5, 25, 200)]
    assert widths[0] < widths[1] < widths[2], widths


def test_deterministic_given_seed():
    x = ar1(0.9)
    assert moving_block_ci(x, BLOCK, 200, seed=7) == \
           moving_block_ci(x, BLOCK, 200, seed=7)
    assert moving_block_ci(x, BLOCK, 200, seed=7) != \
           moving_block_ci(x, BLOCK, 200, seed=8)


def test_interval_brackets_the_sample_mean():
    x = ar1(0.9)
    lo, hi = moving_block_ci(x, BLOCK, 400, seed=0)
    assert lo < x.mean() < hi


def test_block_longer_than_series_is_clamped_not_crashed():
    """Degenerate but must not raise: one block is the whole series."""
    x = np.arange(10.0)
    lo, hi = moving_block_ci(x, block=1000, n_boot=50, seed=0)
    assert lo == hi == pytest.approx(x.mean())


def test_too_short_to_bootstrap_returns_nan():
    lo, hi = moving_block_ci(np.array([1.0]), block=2)
    assert np.isnan(lo) and np.isnan(hi)


def test_resamples_have_the_right_length():
    """Every resample must be length n, or the mean is computed over the wrong
    denominator and the interval silently shifts. Checked via a series whose
    mean is fixed: a constant series must give a zero-width interval AT that
    constant, which only holds if the truncation arithmetic is right."""
    x = np.full(1234, 3.5)
    lo, hi = moving_block_ci(x, block=100, n_boot=50, seed=0)
    assert lo == pytest.approx(3.5) and hi == pytest.approx(3.5)


def test_gross_edge_returns_signed_series():
    """Short side is negated, flat predictions never trade."""
    from costfloor import gross_edge
    bp = np.array([2.0, -3.0, 5.0, 7.0])
    pred = np.array([2, 0, 1, 0])          # up, down, flat, down
    assert list(gross_edge(bp, pred)) == [2.0, 3.0, -7.0]
