"""
SSVI (Surface Stochastic Volatility Inspired) fitting and probability extraction.
Sourced from: Polymarket/poly-market-maker (implied vol / probability estimation),
              ThinkEnigmatic/polymarket-bot-arena (signal processing),
              JonathanPetersonn/oracle-lag-sniper (price analysis).

SSVI is used to model the implied volatility surface and extract
risk-neutral probabilities from market prices. For prediction markets,
we adapt it to extract the probability that a price will be above a
given level at expiry.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


@dataclass
class SSVIParams:
    theta: float  # ATM total variance
    rho: float    # correlation (-1, 1)
    phi: float    # curvature
    r_squared: float = 0.0


@dataclass
class ProbabilityEstimate:
    probability_above: float
    probability_below: float
    implied_vol: float
    r_squared: float


def ssvi_total_variance(k: np.ndarray, theta: float, rho: float, phi: float) -> np.ndarray:
    """
    SSVI parameterization of total implied variance w(k).
    w(k) = (theta / 2) * (1 + rho * phi * k + sqrt((phi * k + rho)^2 + (1 - rho^2)))

    Args:
        k: log-moneyness array (log(K/F))
        theta: ATM total variance (> 0)
        rho: correlation parameter (-1, 1)
        phi: curvature parameter (> 0)
    """
    disc = np.sqrt((phi * k + rho) ** 2 + (1 - rho**2))
    return (theta / 2) * (1 + rho * phi * k + disc)


def fit_ssvi(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    forward: float,
    time_to_expiry: float,
) -> SSVIParams:
    """
    Fit SSVI parameters to observed implied volatilities.

    Args:
        strikes: Array of strike prices
        implied_vols: Array of implied volatilities corresponding to strikes
        forward: Forward price
        time_to_expiry: Time to expiry in years

    Returns:
        Fitted SSVIParams with R² goodness-of-fit.
    """
    if len(strikes) < 3 or time_to_expiry <= 0:
        raise ValueError("Need at least 3 strikes and positive time to expiry")

    # Compute log-moneyness and total variance targets
    k = np.log(strikes / forward)
    total_var_target = (implied_vols ** 2) * time_to_expiry

    def objective(params):
        theta, rho, phi = params
        if theta <= 0 or phi <= 0 or abs(rho) >= 1:
            return 1e10
        w_model = ssvi_total_variance(k, theta, rho, phi)
        if np.any(w_model <= 0):
            return 1e10
        return np.sum((w_model - total_var_target) ** 2)

    # Initial guess based on data characteristics
    atm_var = float(np.mean(implied_vols ** 2) * time_to_expiry)
    bounds = [(1e-6, 10.0), (-0.99, 0.99), (1e-6, 10.0)]

    # Try multiple starting points to avoid local minima
    best_result = None
    best_cost = float("inf")
    for theta0 in [atm_var, atm_var * 0.5, atm_var * 2.0, 0.01, 0.1]:
        for rho0 in [-0.5, 0.0, 0.5]:
            for phi0 in [0.1, 0.5, 1.0, 2.0]:
                x0 = [max(theta0, 1e-6), rho0, phi0]
                try:
                    result = minimize(
                        objective, x0, method="L-BFGS-B", bounds=bounds,
                        options={"maxiter": 500},
                    )
                    if result.fun < best_cost:
                        best_cost = result.fun
                        best_result = result
                except Exception:
                    continue

    if best_result is None:
        # Fallback with Nelder-Mead
        x0 = [max(atm_var, 0.01), 0.0, 1.0]
        best_result = minimize(objective, x0, method="Nelder-Mead",
                               options={"maxiter": 2000})

    theta, rho, phi = best_result.x
    # Clamp parameters
    theta = max(theta, 1e-6)
    phi = max(phi, 1e-6)
    rho = max(-0.99, min(0.99, rho))

    # Compute R²
    w_fitted = ssvi_total_variance(k, theta, rho, phi)
    ss_res = np.sum((total_var_target - w_fitted) ** 2)
    ss_tot = np.sum((total_var_target - np.mean(total_var_target)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return SSVIParams(theta=theta, rho=rho, phi=phi, r_squared=r_squared)


def extract_probability(
    params: SSVIParams,
    spot: float,
    forward: float,
    time_to_expiry: float,
    strike: Optional[float] = None,
) -> ProbabilityEstimate:
    """
    Extract risk-neutral probability from SSVI parameters.

    For prediction markets, the probability that the underlying
    will be above the strike at expiry. Uses Black-Scholes-style
    probability calculation with SSVI-implied volatility.

    Args:
        params: Fitted SSVI parameters
        spot: Current spot price (or market mid-price for prediction markets)
        forward: Forward price
        time_to_expiry: Time to expiry in years
        strike: Strike price (defaults to spot for ATM probability)
    """
    if strike is None:
        strike = spot

    if time_to_expiry <= 0 or forward <= 0 or strike <= 0:
        return ProbabilityEstimate(
            probability_above=0.5,
            probability_below=0.5,
            implied_vol=0.0,
            r_squared=params.r_squared,
        )

    k = math.log(strike / forward)
    w = ssvi_total_variance(np.array([k]), params.theta, params.rho, params.phi)[0]
    implied_vol = math.sqrt(max(w / time_to_expiry, 1e-10))

    d1 = (math.log(forward / strike) + 0.5 * implied_vol**2 * time_to_expiry) / (
        implied_vol * math.sqrt(time_to_expiry)
    )
    d2 = d1 - implied_vol * math.sqrt(time_to_expiry)

    prob_above = norm.cdf(d2)

    return ProbabilityEstimate(
        probability_above=prob_above,
        probability_below=1 - prob_above,
        implied_vol=implied_vol,
        r_squared=params.r_squared,
    )


def generate_synthetic_surface(
    spot: float,
    base_vol: float = 0.5,
    time_to_expiry: float = 1.0 / 365,
    n_strikes: int = 11,
    skew: float = -0.1,
    noise_std: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic implied vol surface for testing/calibration.
    Useful for paper mode when real option data isn't available.

    Returns:
        (strikes, implied_vols) arrays
    """
    rng = np.random.default_rng(42)
    moneyness = np.linspace(-0.3, 0.3, n_strikes)
    strikes = spot * np.exp(moneyness)
    # Simple smile: vol = base_vol + skew * k + curvature * k^2
    implied_vols = base_vol + skew * moneyness + 0.5 * moneyness**2
    implied_vols += rng.normal(0, noise_std, n_strikes)
    implied_vols = np.maximum(implied_vols, 0.01)
    return strikes, implied_vols
