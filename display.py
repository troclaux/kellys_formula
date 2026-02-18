"""Formatted output and warnings."""

import sys

import numpy as np


def print_results(
    tickers: list[str],
    F: np.ndarray,
    F_half: np.ndarray,
    M: np.ndarray,
    sharpe: float,
    growth_rate: float,
    full_kelly: bool = False,
) -> None:
    """Print a formatted results table to stdout.

    Args:
        tickers: List of ticker symbols.
        F: Full Kelly leverage vector.
        F_half: Half Kelly leverage vector.
        M: Annualized mean excess return vector.
        sharpe: Portfolio Sharpe ratio.
        growth_rate: Maximum compounded growth rate.
        full_kelly: If True, recommend full Kelly; otherwise recommend half Kelly.
    """
    print("\n" + "=" * 60)
    print("Kelly Criterion Capital Allocation")
    print("=" * 60)

    header = f"{'Ticker':<10} {'Full Kelly':>12} {'Half Kelly':>12} {'Ann. Excess':>12}"
    print(header)
    print("-" * 60)

    for i, ticker in enumerate(tickers):
        print(
            f"{ticker:<10} {F[i]:>12.4f} {F_half[i]:>12.4f} {M[i]:>12.4f}"
        )

    print("-" * 60)
    recommended = F if full_kelly else F_half
    label = "Full" if full_kelly else "Half"
    print(f"Recommended allocation: {label} Kelly")
    print(f"Portfolio Sharpe Ratio: {sharpe:.4f}")
    print(f"Max Growth Rate (CAGR): {growth_rate:.4f} ({growth_rate * 100:.2f}%)")
    print("=" * 60 + "\n")


def print_warnings(
    tickers: list[str],
    F: np.ndarray,
    num_observations: int,
) -> None:
    """Print risk warnings to stderr.

    Args:
        tickers: List of ticker symbols.
        F: Full Kelly leverage vector.
        num_observations: Number of return observations used.
    """
    warnings = []

    # High leverage warning
    for i, ticker in enumerate(tickers):
        if abs(F[i]) > 1.0:
            warnings.append(
                f"  - {ticker}: Full Kelly leverage is {F[i]:.2f}x "
                f"(implies {'leveraged long' if F[i] > 0 else 'short'} position)"
            )

    if warnings:
        print("\nWARNING: High leverage detected:", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)

    if num_observations < 60:
        print(
            f"\nWARNING: Small sample size ({num_observations} observations). "
            "Estimates may be unreliable.",
            file=sys.stderr,
        )

    print(
        "\nDISCLAIMERS:",
        file=sys.stderr,
    )
    print(
        "  - Kelly criterion assumes returns are Gaussian and i.i.d. "
        "Real markets deviate significantly from these assumptions.",
        file=sys.stderr,
    )
    print(
        "  - Results require continuous rebalancing to the target allocation.",
        file=sys.stderr,
    )
    print(
        "  - Past return distributions may not persist (regime shifts).",
        file=sys.stderr,
    )
