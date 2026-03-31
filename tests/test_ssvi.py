"""Tests for SSVI fitting and probability extraction."""

import numpy as np
import pytest

from ssvi import (
    SSVIParams,
    extract_probability,
    fit_ssvi,
    generate_synthetic_surface,
    ssvi_total_variance,
)


class TestSSVITotalVariance:
    def test_atm_variance(self):
        """At k=0, total variance should be close to theta."""
        k = np.array([0.0])
        w = ssvi_total_variance(k, theta=0.04, rho=0.0, phi=1.0)
        # w(0) = theta/2 * (1 + 0 + sqrt(rho^2 + 1 - rho^2)) = theta/2 * 2 = theta
        assert w[0] == pytest.approx(0.04, abs=0.001)

    def test_positive_variance(self):
        """Total variance should always be positive."""
        k = np.linspace(-1, 1, 100)
        w = ssvi_total_variance(k, theta=0.04, rho=-0.3, phi=0.5)
        assert np.all(w > 0)

    def test_symmetry_zero_rho(self):
        """With rho=0, the surface should be symmetric in k."""
        k = np.array([-0.5, 0.5])
        w = ssvi_total_variance(k, theta=0.04, rho=0.0, phi=1.0)
        assert w[0] == pytest.approx(w[1], abs=1e-10)

    def test_skew_with_negative_rho(self):
        """Negative rho should produce higher variance for k < 0 (OTM puts)."""
        k = np.array([-0.3, 0.3])
        w = ssvi_total_variance(k, theta=0.04, rho=-0.5, phi=1.0)
        # With negative rho, left wing (k<0) should have higher variance
        assert w[0] > w[1]


class TestFitSSVI:
    def test_fit_synthetic_data(self):
        """Fit SSVI to synthetic data should achieve R² > 0.80."""
        strikes, ivs = generate_synthetic_surface(
            spot=100.0, base_vol=0.3, time_to_expiry=30 / 365,
            n_strikes=15, noise_std=0.005,
        )
        params = fit_ssvi(strikes, ivs, forward=100.0, time_to_expiry=30 / 365)
        assert params.r_squared > 0.80
        assert params.theta > 0
        assert -1 < params.rho < 1
        assert params.phi > 0

    def test_fit_noisier_data(self):
        """Even with more noise, fit should be reasonable."""
        strikes, ivs = generate_synthetic_surface(
            spot=50.0, base_vol=0.5, time_to_expiry=7 / 365,
            n_strikes=11, noise_std=0.02,
        )
        params = fit_ssvi(strikes, ivs, forward=50.0, time_to_expiry=7 / 365)
        assert params.r_squared > 0.40  # More noise = lower R², but still positive
        assert params.theta > 0

    def test_fit_requires_minimum_strikes(self):
        with pytest.raises(ValueError):
            fit_ssvi(
                np.array([100.0, 110.0]),
                np.array([0.3, 0.35]),
                forward=100.0,
                time_to_expiry=1 / 12,
            )

    def test_fit_requires_positive_time(self):
        with pytest.raises(ValueError):
            fit_ssvi(
                np.array([90, 100, 110.0]),
                np.array([0.3, 0.25, 0.35]),
                forward=100.0,
                time_to_expiry=0.0,
            )


class TestExtractProbability:
    def test_atm_probability_near_half(self):
        """Probability at the money should be approximately 0.50."""
        strikes, ivs = generate_synthetic_surface(
            spot=100.0, base_vol=0.3, time_to_expiry=30 / 365,
            n_strikes=15, noise_std=0.005,
        )
        params = fit_ssvi(strikes, ivs, forward=100.0, time_to_expiry=30 / 365)
        prob = extract_probability(
            params, spot=100.0, forward=100.0, time_to_expiry=30 / 365,
        )
        assert prob.probability_above == pytest.approx(0.50, abs=0.15)

    def test_probability_bounds(self):
        """Probability should always be between 0 and 1."""
        params = SSVIParams(theta=0.04, rho=-0.2, phi=0.5, r_squared=0.9)
        prob = extract_probability(params, spot=100.0, forward=100.0, time_to_expiry=1 / 12)
        assert 0 <= prob.probability_above <= 1
        assert 0 <= prob.probability_below <= 1
        assert prob.probability_above + prob.probability_below == pytest.approx(1.0)

    def test_itm_probability_high(self):
        """Deep ITM strike should have high probability_above."""
        params = SSVIParams(theta=0.04, rho=0.0, phi=0.5, r_squared=0.9)
        prob = extract_probability(
            params, spot=100.0, forward=100.0, time_to_expiry=1 / 12,
            strike=50.0,  # Very low strike
        )
        assert prob.probability_above > 0.80

    def test_otm_probability_low(self):
        """Deep OTM strike should have low probability_above."""
        params = SSVIParams(theta=0.04, rho=0.0, phi=0.5, r_squared=0.9)
        prob = extract_probability(
            params, spot=100.0, forward=100.0, time_to_expiry=1 / 12,
            strike=200.0,  # Very high strike
        )
        assert prob.probability_above < 0.20

    def test_zero_time_to_expiry(self):
        """Zero time to expiry should return 0.50."""
        params = SSVIParams(theta=0.04, rho=0.0, phi=0.5, r_squared=0.9)
        prob = extract_probability(params, spot=100.0, forward=100.0, time_to_expiry=0.0)
        assert prob.probability_above == 0.5

    def test_implied_vol_positive(self):
        params = SSVIParams(theta=0.04, rho=0.0, phi=0.5, r_squared=0.9)
        prob = extract_probability(params, spot=100.0, forward=100.0, time_to_expiry=1 / 12)
        assert prob.implied_vol > 0

    def test_r_squared_preserved(self):
        params = SSVIParams(theta=0.04, rho=0.0, phi=0.5, r_squared=0.85)
        prob = extract_probability(params, spot=100.0, forward=100.0, time_to_expiry=1 / 12)
        assert prob.r_squared == 0.85


class TestSyntheticSurface:
    def test_correct_number_of_strikes(self):
        strikes, ivs = generate_synthetic_surface(spot=100.0, n_strikes=15)
        assert len(strikes) == 15
        assert len(ivs) == 15

    def test_strikes_centered_on_spot(self):
        strikes, _ = generate_synthetic_surface(spot=100.0, n_strikes=11)
        mid_strike = strikes[len(strikes) // 2]
        assert mid_strike == pytest.approx(100.0, rel=0.05)

    def test_positive_implied_vols(self):
        _, ivs = generate_synthetic_surface(spot=50.0)
        assert np.all(ivs > 0)

    def test_deterministic_with_seed(self):
        s1, iv1 = generate_synthetic_surface(spot=100.0)
        s2, iv2 = generate_synthetic_surface(spot=100.0)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(iv1, iv2)
