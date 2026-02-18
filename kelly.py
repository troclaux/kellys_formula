"""CLI entry point for Kelly formula capital allocation tool."""

import argparse
import sys

import numpy as np

from calc import (
    compute_excess_returns,
    compute_half_kelly,
    compute_kelly_vector,
    compute_max_growth_rate,
    compute_returns,
    compute_sharpe,
)
from data import fetch_prices
from display import print_results, print_warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate optimal capital allocation using the Kelly criterion."
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        type=str,
        help="Ticker symbols (e.g. AAPL MSFT GOOG)",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=126,
        help="Lookback period in calendar days (default: 126, ~6 months)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Annual risk-free rate (default: 0.05)",
    )
    parser.add_argument(
        "--diagonal",
        action="store_true",
        help="Use only diagonal covariance (ignore correlations)",
    )
    parser.add_argument(
        "--full-kelly",
        action="store_true",
        help="Recommend full Kelly instead of half Kelly",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tickers = [t.upper() for t in args.tickers]

    # Fetch data
    try:
        prices = fetch_prices(tickers, lookback_days=args.lookback)
    except ValueError as e:
        print(f"Data error: {e}", file=sys.stderr)
        return 2

    # Compute
    returns = compute_returns(prices)
    excess_returns = compute_excess_returns(returns, args.risk_free_rate)

    try:
        F, M, C = compute_kelly_vector(excess_returns, diagonal_only=args.diagonal)
    except np.linalg.LinAlgError as e:
        print(f"Computation error: {e}", file=sys.stderr)
        return 2

    F_half = compute_half_kelly(F)
    sharpe = compute_sharpe(M, C, F)
    growth_rate = compute_max_growth_rate(args.risk_free_rate, sharpe)

    # Display
    print_results(
        tickers, F, F_half, M, sharpe, growth_rate, full_kelly=args.full_kelly
    )
    print_warnings(tickers, F, num_observations=len(returns))

    return 0


if __name__ == "__main__":
    sys.exit(main())
