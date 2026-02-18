"""Unit tests for calc.py — hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from calc import (
    compute_excess_returns,
    compute_half_kelly,
    compute_kelly_vector,
    compute_max_growth_rate,
    compute_returns,
    compute_sharpe,
)


class TestComputeReturns:
    def test_simple_returns(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
        returns = compute_returns(prices)
        expected = pd.DataFrame({"A": [0.1, 0.1]}, index=[1, 2])
        pd.testing.assert_frame_equal(returns, expected)

    def test_multiple_assets(self):
        prices = pd.DataFrame({"A": [100.0, 110.0], "B": [200.0, 210.0]})
        returns = compute_returns(prices)
        assert returns["A"].iloc[0] == pytest.approx(0.1)
        assert returns["B"].iloc[0] == pytest.approx(0.05)

    def test_drops_first_row(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]})
        returns = compute_returns(prices)
        assert len(returns) == 3


class TestComputeExcessReturns:
    def test_subtracts_daily_rf(self):
        returns = pd.DataFrame({"A": [0.10, 0.10]})
        excess = compute_excess_returns(returns, annual_rf=0.0252)
        daily_rf = 0.0252 / 252  # 0.0001
        assert excess["A"].iloc[0] == pytest.approx(0.10 - daily_rf)

    def test_zero_rf(self):
        returns = pd.DataFrame({"A": [0.05, -0.02]})
        excess = compute_excess_returns(returns, annual_rf=0.0)
        pd.testing.assert_frame_equal(excess, returns)


class TestComputeKellyVector:
    def test_single_asset_reduces_to_m_over_s2(self):
        """For a single asset, Kelly = mean / variance = m / s^2."""
        # Create returns with known mean and variance
        np.random.seed(42)
        n = 10000
        daily_mean = 0.0004  # annualized ~ 0.1
        daily_std = 0.01  # annualized ~ 0.1587
        r = np.random.normal(daily_mean, daily_std, n)
        excess_returns = pd.DataFrame({"A": r})

        F, M, C = compute_kelly_vector(excess_returns)
        # F should be approximately M / C (scalar case)
        expected_F = M[0] / C[0, 0]
        assert F[0] == pytest.approx(expected_F, rel=1e-6)

    def test_two_uncorrelated_assets_diagonal_equals_full(self):
        """For uncorrelated assets, diagonal-only should match full matrix."""
        np.random.seed(42)
        n = 50000
        r1 = np.random.normal(0.0004, 0.01, n)
        r2 = np.random.normal(0.0003, 0.015, n)
        excess_returns = pd.DataFrame({"A": r1, "B": r2})

        F_full, _, _ = compute_kelly_vector(excess_returns, diagonal_only=False)
        F_diag, _, _ = compute_kelly_vector(excess_returns, diagonal_only=True)

        # With large n and independent draws, these should be close
        np.testing.assert_allclose(F_full, F_diag, atol=0.5)

    def test_singular_covariance_raises(self):
        """Singular covariance matrix should raise LinAlgError."""
        # Two identical columns → singular covariance
        excess_returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.01, 0.02, 0.03]})
        with pytest.raises(np.linalg.LinAlgError):
            compute_kelly_vector(excess_returns)


class TestComputeSharpe:
    def test_sharpe_single_asset(self):
        """For single asset: S = sqrt(F * C * F) = sqrt(m^2/s^2) = |m|/s."""
        M = np.array([0.10])
        C = np.array([[0.04]])  # s = 0.2
        F = np.linalg.solve(C, M)  # 0.10 / 0.04 = 2.5
        sharpe = compute_sharpe(M, C, F)
        # S = m / s = 0.10 / 0.20 = 0.5
        assert sharpe == pytest.approx(0.5, rel=1e-6)


class TestComputeMaxGrowthRate:
    def test_known_values(self):
        """r=0.05, S=1.0 → g = 0.05 + 1.0/2 = 0.55."""
        g = compute_max_growth_rate(annual_rf=0.05, sharpe=1.0)
        assert g == pytest.approx(0.55)

    def test_zero_sharpe(self):
        """With zero Sharpe, growth rate equals risk-free rate."""
        g = compute_max_growth_rate(annual_rf=0.03, sharpe=0.0)
        assert g == pytest.approx(0.03)


class TestComputeHalfKelly:
    def test_half_of_full(self):
        F = np.array([2.0, -1.0, 0.5])
        F_half = compute_half_kelly(F)
        np.testing.assert_array_equal(F_half, np.array([1.0, -0.5, 0.25]))
