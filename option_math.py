"""Canonical Black-Scholes option math for the Quant Portfolio Workbench.

Single source of truth for option pricing, Greeks, and implied-volatility
inference. Every reporting script used to carry its own copy of these
formulas; they are consolidated here so a fix or improvement lands
everywhere at once.

Conventions
-----------
* European Black-Scholes with continuous dividend yield ``q``. Listed equity
  options are American, so treat every output as an estimate, not a mark.
* Scalar functions (``bs_price``, ``bs_delta``, ...) accept floats and return
  floats; they return ``numpy.nan`` on invalid input (except
  ``implied_volatility``, which returns ``None`` when the target price cannot
  be bracketed).
* ``bs_theta`` returns per-year theta; divide by 365 for a daily figure.
* ``*_vec`` variants accept array-likes and return numpy arrays. They clamp
  inputs (spot/strike >= 1e-9, t >= 1e-9, sigma >= 1e-6) exactly as the
  historical per-script copies did, so migration is behavior-preserving.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_RISK_FREE_RATE = 0.045
DEFAULT_DIVIDEND_YIELD = 0.0

_IV_LOW = 1e-4
_IV_HIGH = 5.0
_IV_TOL = 1e-5
_IV_ITERS = 80


def norm_cdf(x):
    """Standard normal CDF. Accepts scalars and array-likes."""
    try:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    except TypeError:
        erf_vec = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf_vec(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def norm_pdf(x):
    """Standard normal PDF. Accepts scalars and array-likes."""
    x_arr = np.asarray(x, dtype=float)
    return (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * x_arr**2)


def _d1_scalar(s, k, t, r, sigma, q):
    return (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))


def _valid_scalar(*vals):
    return not any(isinstance(v, float) and math.isnan(v) for v in vals)


def bs_price(s, k, t, r, sigma, option_type, q=0.0):
    """European Black-Scholes price. Falls back to intrinsic value on bad input."""
    try:
        s, k, t, r, sigma, q = (float(s), float(k), float(t), float(r), float(sigma), float(q))
    except (TypeError, ValueError):
        return float("nan")
    if not _valid_scalar(s, k, t, r, sigma, q) or s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        if _valid_scalar(s, k):
            return max(s - k, 0.0) if option_type == "C" else max(k - s, 0.0)
        return float("nan")
    d1 = _d1_scalar(s, k, t, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(t)
    disc_s = s * math.exp(-q * t)
    disc_k = k * math.exp(-r * t)
    if option_type == "C":
        return disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    return disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1)


def bs_delta(s, k, t, r, sigma, option_type, q=0.0):
    """European Black-Scholes delta. Returns nan on invalid input."""
    try:
        s, k, t, r, sigma, q = (float(s), float(k), float(t), float(r), float(sigma), float(q))
    except (TypeError, ValueError):
        return float("nan")
    if not _valid_scalar(s, k, t, r, sigma, q) or s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return float("nan")
    d1 = _d1_scalar(s, k, t, r, sigma, q)
    call_delta = norm_cdf(d1)
    put_delta = math.exp(-q * t) * (call_delta - 1.0)
    call_delta = math.exp(-q * t) * call_delta
    return call_delta if option_type == "C" else put_delta


def bs_gamma(s, k, t, r, sigma, q=0.0):
    """European Black-Scholes gamma. Returns nan on invalid input."""
    try:
        s, k, t, r, sigma, q = (float(s), float(k), float(t), float(r), float(sigma), float(q))
    except (TypeError, ValueError):
        return float("nan")
    if not _valid_scalar(s, k, t, r, sigma, q) or s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return float("nan")
    d1 = _d1_scalar(s, k, t, r, sigma, q)
    return math.exp(-q * t) * norm_pdf(d1) / (s * sigma * math.sqrt(t))


def bs_vega(s, k, t, r, sigma, q=0.0):
    """European Black-Scholes vega (per 1.0 vol point). Returns nan on invalid input."""
    try:
        s, k, t, r, sigma, q = (float(s), float(k), float(t), float(r), float(sigma), float(q))
    except (TypeError, ValueError):
        return float("nan")
    if not _valid_scalar(s, k, t, r, sigma, q) or s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return float("nan")
    d1 = _d1_scalar(s, k, t, r, sigma, q)
    return s * math.exp(-q * t) * norm_pdf(d1) * math.sqrt(t)


def bs_theta(s, k, t, r, sigma, option_type, q=0.0):
    """European Black-Scholes theta, per year. Divide by 365 for daily. Nan on invalid input."""
    try:
        s, k, t, r, sigma, q = (float(s), float(k), float(t), float(r), float(sigma), float(q))
    except (TypeError, ValueError):
        return float("nan")
    if not _valid_scalar(s, k, t, r, sigma, q) or s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return float("nan")
    d1 = _d1_scalar(s, k, t, r, sigma, q)
    d2 = d1 - sigma * math.sqrt(t)
    pdf = norm_pdf(d1)
    first = -(s * math.exp(-q * t) * pdf * sigma) / (2.0 * math.sqrt(t))
    if option_type == "C":
        return first - r * k * math.exp(-r * t) * norm_cdf(d2) + q * s * math.exp(-q * t) * norm_cdf(d1)
    return first + r * k * math.exp(-r * t) * norm_cdf(-d2) - q * s * math.exp(-q * t) * norm_cdf(-d1)


def implied_volatility(target_price, s, k, t, r, option_type, q=0.0):
    """Bisection implied volatility. Returns None when the price is not bracketed."""
    try:
        target_price, s, k, t, r, q = (
            float(target_price), float(s), float(k), float(t), float(r), float(q),
        )
    except (TypeError, ValueError):
        return None
    if not _valid_scalar(target_price, s, k, t, r, q):
        return None
    if target_price <= 0 or s <= 0 or k <= 0 or t <= 0:
        return None
    low_price = bs_price(s, k, t, r, _IV_LOW, option_type, q)
    high_price = bs_price(s, k, t, r, _IV_HIGH, option_type, q)
    if target_price < low_price - 1e-6 or target_price > high_price + 1e-6:
        return None
    low, high = _IV_LOW, _IV_HIGH
    for _ in range(_IV_ITERS):
        mid = (low + high) / 2.0
        mid_price = bs_price(s, k, t, r, mid, option_type, q)
        if abs(mid_price - target_price) < _IV_TOL:
            return mid
        if mid_price < target_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def bs_put_delta(s, k, t, r, sigma, q=0.0):
    """European Black-Scholes put delta. Returns nan on invalid input."""
    return bs_delta(s, k, t, r, sigma, "P", q)


def bs_call_delta(s, k, t, r, sigma, q=0.0):
    """European Black-Scholes call delta. Returns nan on invalid input."""
    return bs_delta(s, k, t, r, sigma, "C", q)


def normalize_put_delta(d):
    """Normalize a broker delta to a signed put delta in [-1.05, 0]."""
    try:
        if d is None or (isinstance(d, float) and math.isnan(d)):
            return float("nan")
        d = float(d)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(d):
        return float("nan")
    if 1.5 < abs(d) <= 100:
        d /= 100.0
    d = -abs(d)
    return d if abs(d) <= 1.05 else float("nan")


# ---------------------------------------------------------------------------
# Vectorized variants (array-in, array-out). Input clamping matches the
# historical per-script copies so migration is behavior-preserving.
# ---------------------------------------------------------------------------

def _clamp_vec(spot, strike, t, sigma):
    spot = np.maximum(np.asarray(spot, dtype=float), 1e-9)
    strike = np.maximum(np.asarray(strike, dtype=float), 1e-9)
    t = np.maximum(np.asarray(t, dtype=float), 1e-9)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    return spot, strike, t, sigma


def bs_d1(spot, strike, t, r, sigma, q=0.0):
    """Vectorized d1."""
    spot, strike, t, sigma = _clamp_vec(spot, strike, t, sigma)
    return (np.log(spot / strike) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def bs_price_vec(spot, strike, t, r, sigma, option_type, q=0.0):
    """Vectorized European price."""
    spot, strike, t, sigma = _clamp_vec(spot, strike, t, sigma)
    d1 = bs_d1(spot, strike, t, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(t)
    disc_s = spot * np.exp(-q * t)
    disc_k = strike * np.exp(-r * t)
    call = disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    put = disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1)
    return np.where(np.asarray(option_type) == "C", call, put)


def bs_delta_vec(spot, strike, t, r, sigma, option_type, q=0.0):
    """Vectorized European delta."""
    d1 = bs_d1(spot, strike, t, r, sigma, q)
    t_arr = np.maximum(np.asarray(t, dtype=float), 1e-9)
    disc = np.exp(-q * t_arr)
    call_delta = disc * norm_cdf(d1)
    put_delta = disc * (call_delta / np.maximum(disc, 1e-12) - 1.0)
    return np.where(np.asarray(option_type) == "C", call_delta, put_delta)


def bs_gamma_vec(spot, strike, t, r, sigma, q=0.0):
    """Vectorized European gamma."""
    spot, strike, t, sigma = _clamp_vec(spot, strike, t, sigma)
    d1 = bs_d1(spot, strike, t, r, sigma, q)
    pdf_d1 = norm_pdf(d1)
    return np.exp(-q * t) * pdf_d1 / (spot * sigma * np.sqrt(t))


def bs_vega_vec(spot, strike, t, r, sigma, q=0.0):
    """Vectorized European vega (per 1.0 vol point)."""
    spot, strike, t, sigma = _clamp_vec(spot, strike, t, sigma)
    d1 = bs_d1(spot, strike, t, r, sigma, q)
    pdf_d1 = norm_pdf(d1)
    return spot * np.exp(-q * t) * pdf_d1 * np.sqrt(t)


def bs_theta_vec(spot, strike, t, r, sigma, option_type, q=0.0):
    """Vectorized European theta, per year. Divide by 365 for daily."""
    spot, strike, t, sigma = _clamp_vec(spot, strike, t, sigma)
    d1 = bs_d1(spot, strike, t, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(t)
    pdf_d1 = norm_pdf(d1)
    t_c = np.maximum(t, 1e-9)
    first = -(spot * np.exp(-q * t_c) * pdf_d1 * sigma) / (2.0 * np.sqrt(t_c))
    call_theta = first - r * strike * np.exp(-r * t_c) * norm_cdf(d2) + q * spot * np.exp(-q * t_c) * norm_cdf(d1)
    put_theta = first + r * strike * np.exp(-r * t_c) * norm_cdf(-d2) - q * spot * np.exp(-q * t_c) * norm_cdf(-d1)
    return np.where(np.asarray(option_type) == "C", call_theta, put_theta)
