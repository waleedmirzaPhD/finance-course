"""Universe and price loading with on-disk caching.

Caching is keyed by (sorted tickers, start, end). The cache is parquet for
fast reload during repeated backtests. Cache lives under data/cache/.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed. pip install -r requirements.txt")


CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_key(tickers: list[str], start: str, end: str | None) -> str:
    payload = "|".join(sorted(tickers)) + f"::{start}::{end or 'today'}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def load_prices(tickers: list[str], start: str, end: str | None = None,
                use_cache: bool = True) -> pd.DataFrame:
    """Adjusted-close panel: index=Date, columns=Ticker.

    Forward-filled at most one business day to bridge single-day quote gaps.
    Tickers with no overlapping history are dropped with a warning.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _cache_key(tickers, start, end)
    cache_path = CACHE_DIR / f"prices_{key}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        progress=False, group_by="ticker", threads=True,
    )
    if raw.empty:
        sys.exit("yfinance returned no data.")

    if len(tickers) == 1:
        close = raw[["Close"]].copy()
        close.columns = tickers
    else:
        cols = {}
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                cols[t] = raw[t]["Close"]
        close = pd.DataFrame(cols)

    close = close.sort_index().dropna(how="all")
    close = close.ffill(limit=1)
    coverage = close.notna().mean()
    keep = coverage[coverage > 0.5].index.tolist()
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        print(f"[data] dropping tickers with <50% coverage: {', '.join(dropped)}")
    close = close[keep]

    close.to_parquet(cache_path)
    return close


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, leading NaN dropped, intermediate NaNs preserved
    so a missing day for one ticker doesn't bias the cross-section."""
    return prices.pct_change(fill_method=None).iloc[1:]
