"""Tests for the canonical Black-Scholes module (option_math)."""

import math

import numpy as np
import pytest

from option_math import (
    bs_call_delta,
    bs_delta,
    bs_delta_vec,
    bs_d1,
    bs_gamma,
    bs_gamma_vec,
    bs_price,
    bs_price_vec,
    bs_put_delta,
    bs_theta,
    bs_theta_vec,
    bs_vega,
    bs_vega_vec,
    implied_volatility,
    norm_cdf,
    norm_pdf,
    normalize_put_delta,
)

# Hull textbook reference: S=K=100, T=1, r=0.05, sigma=0.2, q=0
S, K, T, R, SIG = 100.0, 100.0, 1.0, 0.05, 0.2
REF_CALL = 10.4506
REF_PUT = 5.5735


def test_norm_cdf_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_norm_cdf_vectorized_matches_scalar():
    xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    vec = norm_cdf(xs)
    assert np.allclose(vec, [norm_cdf(float(x)) for x in xs])


def test_norm_pdf():
    assert norm_pdf(0.0) == pytest.approx(1.0 / math.sqrt(2 * math.pi))


def test_bs_price_matches_textbook():
    assert bs_price(S, K, T, R, SIG, "C") == pytest.approx(REF_CALL, abs=1e-3)
    assert bs_price(S, K, T, R, SIG, "P") == pytest.approx(REF_PUT, abs=1e-3)


def test_put_call_parity():
    for opt_t, s, k, t, r, sig, q in [
        ("C", 100, 100, 1.0, 0.05, 0.2, 0.0),
        ("C", 150, 120, 0.5, 0.045, 0.35, 0.01),
        ("C", 80, 100, 2.0, 0.03, 0.5, 0.02),
    ]:
        call = bs_price(s, k, t, r, sig, "C", q)
        put = bs_price(s, k, t, r, sig, "P", q)
        parity = s * math.exp(-q * t) - k * math.exp(-r * t)
        assert call - put == pytest.approx(parity, rel=1e-9)


def test_delta_bounds_and_sign():
    assert 0.0 < bs_call_delta(S, K, T, R, SIG) < 1.0
    assert -1.0 < bs_put_delta(S, K, T, R, SIG) < 0.0
    assert bs_delta(S, K, T, R, SIG, "C") == pytest.approx(bs_call_delta(S, K, T, R, SIG))
    # deep ITM call -> delta near 1; deep OTM put -> delta near 0
    assert bs_call_delta(200, K, T, R, SIG) == pytest.approx(1.0, abs=1e-3)
    assert bs_put_delta(200, K, T, R, SIG) == pytest.approx(0.0, abs=1e-3)


def test_gamma_vega_positive_and_symmetric():
    g = bs_gamma(S, K, T, R, SIG)
    assert g > 0
    assert bs_vega(S, K, T, R, SIG) > 0


def test_theta_negative_for_long_option():
    # long options decay: theta (per year) should be negative
    assert bs_theta(S, K, T, R, SIG, "C") < 0
    assert bs_theta(S, K, T, R, SIG, "P") < 0


def test_scalar_invalid_inputs():
    assert math.isnan(bs_delta(100, 100, 0.0, R, SIG, "C"))  # T=0
    assert math.isnan(bs_delta(100, 100, T, R, -0.5, "C"))  # sigma<0
    assert math.isnan(bs_put_delta(100, 100, T, R, float("nan")))
    # price falls back to intrinsic on degenerate input
    assert bs_price(110, 100, 0.0, R, SIG, "C") == pytest.approx(10.0)
    assert bs_price(90, 100, 0.0, R, SIG, "P") == pytest.approx(10.0)


def test_implied_volatility_round_trip():
    for sig in (0.15, 0.3, 0.6, 1.2):
        for typ in ("C", "P"):
            price = bs_price(105, 100, 0.75, 0.045, sig, typ)
            iv = implied_volatility(price, 105, 100, 0.75, 0.045, typ)
            assert iv == pytest.approx(sig, rel=1e-4)


def test_implied_volatility_unbracketable_returns_none():
    assert implied_volatility(-5.0, S, K, T, R, "C") is None
    assert implied_volatility(0.0, S, K, T, R, "C") is None
    # absurdly high price for a far OTM option
    assert implied_volatility(1e9, S, 1000.0, T, R, "C") is None
    assert implied_volatility(float("nan"), S, K, T, R, "C") is None


def test_vec_matches_scalar():
    spots = np.array([80.0, 100.0, 120.0])
    strikes = np.array([100.0, 100.0, 100.0])
    ts = np.array([0.25, 1.0, 2.0])
    sigs = np.array([0.2, 0.3, 0.4])
    typs = np.array(["C", "P", "C"])
    for i in range(3):
        s, k, t, sg, ty = spots[i], strikes[i], ts[i], sigs[i], typs[i]
        assert bs_delta_vec(spots, strikes, ts, R, sigs, typs)[i] == pytest.approx(
            bs_delta(s, k, t, R, sg, ty), rel=1e-9
        )
        assert bs_gamma_vec(spots, strikes, ts, R, sigs)[i] == pytest.approx(
            bs_gamma(s, k, t, R, sg), rel=1e-9
        )
        assert bs_vega_vec(spots, strikes, ts, R, sigs)[i] == pytest.approx(
            bs_vega(s, k, t, R, sg), rel=1e-9
        )
        assert bs_theta_vec(spots, strikes, ts, R, sigs, typs)[i] == pytest.approx(
            bs_theta(s, k, t, R, sg, ty), rel=1e-9
        )
        assert bs_price_vec(spots, strikes, ts, R, sigs, typs)[i] == pytest.approx(
            bs_price(s, k, t, R, sg, ty), rel=1e-9
        )


def test_bs_d1_known_value():
    # d1 for the Hull reference case
    d1 = bs_d1(np.array([S]), np.array([K]), np.array([T]), R, np.array([SIG]))[0]
    assert d1 == pytest.approx(0.35, abs=1e-9)


def test_normalize_put_delta():
    assert normalize_put_delta(-0.14) == pytest.approx(-0.14)
    assert normalize_put_delta(0.14) == pytest.approx(-0.14)
    assert normalize_put_delta("-14") == pytest.approx(-0.14)
    assert normalize_put_delta("14") == pytest.approx(-0.14)
    assert math.isnan(normalize_put_delta(float("nan")))
    assert math.isnan(normalize_put_delta(None))
    assert math.isnan(normalize_put_delta(250))  # absurd -> nan
