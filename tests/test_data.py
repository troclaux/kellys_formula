"""Integration tests for data.py — requires network access."""

import pytest

from data import fetch_prices


pytestmark = pytest.mark.integration


class TestFetchPrices:
    def test_single_ticker(self):
        prices = fetch_prices(["AAPL"], lookback_days=30)
        assert "AAPL" in prices.columns
        assert len(prices) >= 2

    def test_multiple_tickers(self):
        prices = fetch_prices(["AAPL", "MSFT"], lookback_days=30)
        assert "AAPL" in prices.columns
        assert "MSFT" in prices.columns
        assert len(prices) >= 2

    def test_invalid_ticker_raises(self):
        with pytest.raises(ValueError):
            fetch_prices(["INVALIDTICKER123XYZ"], lookback_days=30)
