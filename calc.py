"""Pure math functions for Kelly formula calculations. No I/O."""

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple arithmetic returns: (P_t - P_{t-1}) / P_{t-1}."""
    returns = prices.pct_change().iloc[1:]
    return returns


def compute_excess_returns(returns: pd.DataFrame, annual_rf: float) -> pd.DataFrame:
    """Subtract daily risk-free rate (annual_rf / 252) from returns."""
    daily_rf = annual_rf / 252
    return returns - daily_rf


def compute_kelly_vector(
    excess_returns: pd.DataFrame, diagonal_only: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Kelly optimal leverage vector F = C⁻¹M.

    Annualizes mean (M * 252) and covariance (C * 252).

    Args:
        excess_returns: DataFrame of excess returns.
        diagonal_only: If True, zero off-diagonal covariance elements before inversion.

    Returns:
        Tuple of (F, M, C) where F is the Kelly vector, M is the annualized
        mean excess return vector, and C is the annualized covariance matrix.

    Raises:
        numpy.linalg.LinAlgError: If the covariance matrix is singular.
    """
    M = excess_returns.mean().values * 252
    C = excess_returns.cov().values * 252

    if diagonal_only:
        C = np.diag(np.diag(C))

    F = np.linalg.solve(C, M)
    return F, M, C


def compute_sharpe(M: np.ndarray, C: np.ndarray, F: np.ndarray) -> float:
    """Compute portfolio Sharpe ratio: S = sqrt(F^T C F)."""
    return float(np.sqrt(F.T @ C @ F))


def compute_max_growth_rate(annual_rf: float, sharpe: float) -> float:
    """Compute maximum growth rate: g = r_F + S^2 / 2."""
    return annual_rf + sharpe**2 / 2


def compute_half_kelly(F: np.ndarray) -> np.ndarray:
    """Compute half-Kelly leverage: F / 2."""
    return F / 2
