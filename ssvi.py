"""
SSVI surface fitting and risk-neutral probability extraction.

References:
  Gatheral & Jacquier (2014), "Arbitrage-free SVI volatility surfaces"
  Gatheral (2004), "A parsimonious arbitrage-free implied volatility parameterization"
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Bounds on fitted parameters
MIN_VARIANCE = 1e-6
MAX_VARIANCE_BOUND = 10.0
RHO_BOUND = 0.99


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
    SSVI total variance: w(k) = (theta/2)(1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))
    """
    disc = np.sqrt((phi * k + rho) ** 2 + (1 - rho**2))
    return (theta / 2) * (1 + rho * phi * k + disc)


def fit_ssvi(
    strikes: np.ndarray,
    implied_vols: np.ndarray,
    forward: float,
    time_to_expiry: float,
) -> SSVIParams:
    """Fit SSVI params to observed IVs. Returns fitted params with R^2."""
    if len(strikes) < 3 or time_to_expiry <= 0:
        raise ValueError("Need at least 3 strikes and positive time to expiry")

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

    bounds = [(MIN_VARIANCE, MAX_VARIANCE_BOUND), (-RHO_BOUND, RHO_BOUND),
              (MIN_VARIANCE, MAX_VARIANCE_BOUND)]

    # --- data-driven initial guess ---
    atm_var = float(np.mean(implied_vols ** 2) * time_to_expiry)
    theta0 = max(atm_var, MIN_VARIANCE)

    # Estimate rho from smile skew: regress total_var on k for a rough slope
    if len(k) >= 3 and np.std(k) > 1e-8:
        slope = float(np.polyfit(k, total_var_target, 1)[0])
        # rho ~ slope / (theta * phi), but we don't know phi yet — just use sign + magnitude
        rho0 = float(np.clip(slope / max(atm_var, 0.01), -0.8, 0.8))
    else:
        rho0 = 0.0

    # Estimate phi from wing curvature
    if len(k) >= 5:
        coeffs = np.polyfit(k, total_var_target, 2)
        curvature = float(coeffs[0])
        phi0 = float(np.clip(abs(curvature) / max(atm_var * 0.5, 1e-4), 0.1, 5.0))
    else:
        phi0 = 1.0

    x0 = [theta0, rho0, phi0]

    # Primary pass — L-BFGS-B with data-driven init
    best_result = None
    try:
        best_result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 500})
    except Exception:
        pass

    # If first pass failed or got stuck, perturb and retry once
    if best_result is None or best_result.fun > 1e-4:
        perturbed = [theta0 * 1.5, rho0 * 0.5, phi0 * 2.0]
        perturbed = [max(MIN_VARIANCE, perturbed[0]), np.clip(perturbed[1], -0.8, 0.8),
                     max(MIN_VARIANCE, perturbed[2])]
        try:
            result2 = minimize(objective, perturbed, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 500})
            if best_result is None or result2.fun < best_result.fun:
                best_result = result2
        except Exception:
            pass

    # Last resort — Nelder-Mead (unbounded, slower, but robust)
    if best_result is None:
        x0_fallback = [max(atm_var, 0.01), 0.0, 1.0]
        best_result = minimize(objective, x0_fallback, method="Nelder-Mead",
                               options={"maxiter": 2000})

    theta, rho, phi = best_result.x
    theta = max(theta, MIN_VARIANCE)
    phi = max(phi, MIN_VARIANCE)
    rho = max(-RHO_BOUND, min(RHO_BOUND, rho))

    # R^2
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
    Risk-neutral P(S_T > K) via Black-Scholes with SSVI-implied vol.
    Defaults to ATM (strike=spot) if no strike given.
    """
    if strike is None:
        strike = spot

    if time_to_expiry <= 0 or forward <= 0 or strike <= 0:
        return ProbabilityEstimate(
            probability_above=0.5, probability_below=0.5,
            implied_vol=0.0, r_squared=params.r_squared,
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
    """Generate synthetic IV surface for paper mode / testing."""
    rng = np.random.default_rng(42)
    moneyness = np.linspace(-0.3, 0.3, n_strikes)
    strikes = spot * np.exp(moneyness)
    implied_vols = base_vol + skew * moneyness + 0.5 * moneyness**2
    implied_vols += rng.normal(0, noise_std, n_strikes)
    implied_vols = np.maximum(implied_vols, 0.01)
    return strikes, implied_vols
