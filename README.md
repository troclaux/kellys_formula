# Kelly Formula Capital Allocation

CLI tool that calculates optimal capital allocation and leverage for a stock portfolio using the Kelly criterion (F = C⁻¹M), based on Ernest Chan's *Quantitative Trading*.

Given a set of stock tickers, it fetches historical prices from Yahoo Finance, computes the mean excess returns and covariance matrix, and outputs the optimal fraction of capital to allocate to each asset.

## Installation

```bash
pip install -r requirements.txt
```

For development (running tests):

```bash
pip install -r requirements-dev.txt
```

## Usage

```bash
python kelly.py TICKER [TICKER ...] [options]
```

### Passing tickers directly

```bash
python kelly.py AAPL MSFT GOOG
```

### Brazilian stocks (B3)

Use the `.SA` suffix for B3 tickers:

```bash
python kelly.py PETR4.SA VALE3.SA ITUB4.SA
```

### Reading tickers from a file

Create a `.txt` file with one ticker per line. Lines starting with `#` are comments and blank lines are ignored.

```
# tickers.txt
PETR4.SA
VALE3.SA
ITUB4.SA
WEGE3.SA
```

```bash
python kelly.py --file tickers.txt
```

You can combine both — tickers from the file and the command line are merged:

```bash
python kelly.py BBDC4.SA --file tickers.txt
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--file`, `-f` | | Path to a `.txt` file with one ticker per line |
| `--lookback` | `126` | Lookback period in calendar days (~6 months) |
| `--risk-free-rate` | `0.05` | Annual risk-free rate |
| `--diagonal` | off | Use only the diagonal of the covariance matrix (ignore correlations between assets) |
| `--full-kelly` | off | Recommend full Kelly allocation instead of the default half Kelly |

## Examples

```bash
# US stocks with default settings (half Kelly, 126-day lookback, 5% risk-free rate)
python kelly.py AAPL MSFT GOOG

# 1-year lookback with a custom risk-free rate
python kelly.py SPY QQQ --lookback 252 --risk-free-rate 0.045

# Ignore correlations and use full Kelly
python kelly.py PETR4.SA VALE3.SA --diagonal --full-kelly

# Read tickers from a file
python kelly.py -f my_portfolio.txt --lookback 252
```

## Output

The tool prints a table with per-ticker leverage values and a portfolio summary:

- **Full Kelly**: optimal leverage for each asset (F = C⁻¹M)
- **Half Kelly**: half of the full Kelly leverage (more conservative, recommended by default)
- **Ann. Excess**: annualized mean excess return for each asset
- **Portfolio Sharpe Ratio**: S = sqrt(F^T C F)
- **Max Growth Rate (CAGR)**: g = risk-free rate + S²/2

Warnings are printed to stderr, including high leverage alerts, small sample warnings, and disclaimers about the Gaussian assumption.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Argument error (missing tickers, file not found) |
| 2 | Data or computation error (invalid ticker, singular covariance) |

## Running tests

```bash
# Unit tests only (fast, no network)
pytest tests/test_calc.py -v

# All tests including integration (requires network)
pytest tests/ -v
```

## How it works

The Kelly criterion provides the theoretically optimal fraction of capital to bet on each asset to maximize long-term portfolio growth. For a portfolio of N assets:

1. Compute daily arithmetic returns from historical prices
2. Subtract the daily risk-free rate to get excess returns
3. Estimate the annualized mean excess return vector **M** and covariance matrix **C**
4. Solve **F = C⁻¹M** for the optimal leverage vector
5. Half Kelly (F/2) is recommended in practice because the Gaussian/i.i.d. assumptions are never perfectly met in real markets
