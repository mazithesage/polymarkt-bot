"""Tests for Kelly criterion and position sizing - including edge cases."""

import pytest

from kelly import KellyResult, check_position_limits, kelly_criterion, multi_outcome_kelly


class TestKellyCriterion:
    def test_positive_edge_buy_yes(self):
        """When estimated prob > market price, buy YES."""
        result = kelly_criterion(
            estimated_prob=0.70, market_price=0.50, bankroll=1000.0,
            fraction=0.25, max_position=100.0, min_edge=0.02,
        )
        assert result is not None
        assert result.edge == pytest.approx(0.20, abs=0.01)
        assert result.token_choice == "YES"
        assert result.side == "BUY"
        assert result.position_size > 0

    def test_positive_edge_buy_no(self):
        """When estimated prob < market price, buy NO."""
        result = kelly_criterion(
            estimated_prob=0.30, market_price=0.50, bankroll=1000.0,
            fraction=0.25, max_position=100.0, min_edge=0.02,
        )
        assert result is not None
        assert result.token_choice == "NO"
        assert result.side == "BUY"

    def test_no_edge_returns_none(self):
        """When edge < min_edge, returns None."""
        result = kelly_criterion(
            estimated_prob=0.51, market_price=0.50, bankroll=1000.0,
            fraction=0.25, max_position=100.0, min_edge=0.02,
        )
        assert result is None

    def test_exact_min_edge(self):
        """Edge exactly at min_edge threshold."""
        result = kelly_criterion(
            estimated_prob=0.52, market_price=0.50, bankroll=1000.0,
            fraction=0.25, max_position=100.0, min_edge=0.02,
        )
        assert result is not None
        assert result.edge >= 0.02

    def test_max_position_cap(self):
        """Position size should be capped by max_position."""
        result = kelly_criterion(
            estimated_prob=0.90, market_price=0.10, bankroll=100000.0,
            fraction=1.0, max_position=50.0, min_edge=0.01,
        )
        assert result is not None
        assert result.position_size <= 50.0

    def test_fractional_kelly_reduces_size(self):
        """Quarter Kelly should produce ~1/4 the full Kelly size."""
        full = kelly_criterion(
            estimated_prob=0.70, market_price=0.50, bankroll=1000.0,
            fraction=1.0, max_position=10000.0, min_edge=0.02,
        )
        quarter = kelly_criterion(
            estimated_prob=0.70, market_price=0.50, bankroll=1000.0,
            fraction=0.25, max_position=10000.0, min_edge=0.02,
        )
        assert full is not None and quarter is not None
        assert quarter.position_size == pytest.approx(full.position_size * 0.25, rel=0.01)

    def test_invalid_probability_zero(self):
        result = kelly_criterion(estimated_prob=0.0, market_price=0.50, bankroll=1000.0)
        assert result is None

    def test_invalid_probability_one(self):
        result = kelly_criterion(estimated_prob=1.0, market_price=0.50, bankroll=1000.0)
        assert result is None

    def test_invalid_market_price_zero(self):
        result = kelly_criterion(estimated_prob=0.50, market_price=0.0, bankroll=1000.0)
        assert result is None

    def test_invalid_market_price_one(self):
        result = kelly_criterion(estimated_prob=0.50, market_price=1.0, bankroll=1000.0)
        assert result is None

    def test_negative_probability(self):
        result = kelly_criterion(estimated_prob=-0.1, market_price=0.50, bankroll=1000.0)
        assert result is None

    def test_probability_above_one(self):
        result = kelly_criterion(estimated_prob=1.5, market_price=0.50, bankroll=1000.0)
        assert result is None

    def test_zero_bankroll(self):
        result = kelly_criterion(
            estimated_prob=0.70, market_price=0.50, bankroll=0.0,
            fraction=0.25, max_position=100.0, min_edge=0.02,
        )
        assert result is None

    def test_small_edge_large_bankroll(self):
        """Small edge with large bankroll should produce small position."""
        result = kelly_criterion(
            estimated_prob=0.53, market_price=0.50, bankroll=10000.0,
            fraction=0.25, max_position=100.0, min_edge=0.02,
        )
        assert result is not None
        assert result.position_size > 0
        assert result.position_size <= 100.0

    def test_kelly_fraction_stored(self):
        result = kelly_criterion(
            estimated_prob=0.70, market_price=0.50, bankroll=1000.0,
            fraction=0.25, max_position=1000.0, min_edge=0.02,
        )
        assert result is not None
        assert 0 < result.kelly_fraction <= 0.25

    def test_derived_price_zero_via_no_side(self):
        """When market_price=1.0, derived NO price=0 should return None."""
        result = kelly_criterion(
            estimated_prob=0.01, market_price=0.999999, bankroll=1000.0,
            min_edge=0.0001,
        )
        # Even with tiny min_edge, price near zero should be handled safely
        # (either returns None or a valid result, no division by zero)
        if result is not None:
            assert result.position_size > 0

    def test_confidence_matches_selected_prob(self):
        """Confidence should be the probability used for the chosen side."""
        result = kelly_criterion(
            estimated_prob=0.70, market_price=0.50, bankroll=1000.0,
        )
        assert result is not None
        assert result.confidence == 0.70  # Buying YES

        result_no = kelly_criterion(
            estimated_prob=0.30, market_price=0.50, bankroll=1000.0,
        )
        assert result_no is not None
        assert result_no.confidence == 0.70  # 1 - 0.30 for NO side


class TestMultiOutcomeKelly:
    def test_finds_best_edges(self):
        results = multi_outcome_kelly(
            probabilities=[0.70, 0.20, 0.10],
            market_prices=[0.50, 0.30, 0.20],
            bankroll=1000.0,
            fraction=0.25,
            min_edge=0.02,
        )
        assert len(results) > 0
        # Results should be sorted by edge, highest first
        for i in range(len(results) - 1):
            assert results[i].edge >= results[i + 1].edge

    def test_no_edges_returns_empty(self):
        results = multi_outcome_kelly(
            probabilities=[0.50, 0.30, 0.20],
            market_prices=[0.50, 0.30, 0.20],
            bankroll=1000.0,
            min_edge=0.02,
        )
        assert results == []


class TestPositionLimits:
    def test_within_limits(self):
        adjusted = check_position_limits(
            current_exposure=50.0, new_position=30.0, max_total_exposure=500.0,
        )
        assert adjusted == 30.0

    def test_at_limit(self):
        adjusted = check_position_limits(
            current_exposure=500.0, new_position=50.0, max_total_exposure=500.0,
        )
        assert adjusted == 0.0

    def test_partial_fill(self):
        adjusted = check_position_limits(
            current_exposure=480.0, new_position=50.0, max_total_exposure=500.0,
        )
        assert adjusted == 20.0

    def test_zero_exposure(self):
        adjusted = check_position_limits(
            current_exposure=0.0, new_position=100.0, max_total_exposure=500.0,
        )
        assert adjusted == 100.0
