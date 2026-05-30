# Cross-Sectional Equity L/S — Research Showcase

A self-contained quant research project: a dollar-neutral long/short equity
strategy on a fixed mega-cap US universe, built from classical
cross-sectional signals (momentum, low-volatility, trend-quality) blended
by a walk-forward Ridge regression, and optionally allocated with a
TensorFlow max-Sharpe optimizer. Backtested with realistic execution lag,
transaction costs, and slippage.

The goal is not to outperform Renaissance — it is to demonstrate the
**research workflow** a junior quant is expected to defend in an interview:

- a falsifiable hypothesis tied to known literature,
- careful avoidance of lookahead and survivorship bias,
- separation of signal research from portfolio construction,
- honest cost modeling,
- explicit performance and signal-quality diagnostics (Sharpe, IR, IC, turnover).

---

## Hypothesis

Three cross-sectional anomalies have well-documented (and persistent) excess
returns in equity markets:

| Signal       | Source                            | Direction |
|--------------|-----------------------------------|-----------|
| 12-1 momentum| Jegadeesh & Titman (1993)         | Long winners, short losers (skip last month to avoid 1-month reversal) |
| Low volatility| Ang, Hodrick, Xing, Zhang (2006) | Long low-vol, short high-vol |
| Trend quality (μ/σ, 1y)| Asness, Frazzini, Pedersen (2019, "quality minus junk" cousin) | Long high-Sharpe-trend names |

The strategy:
1. Compute each signal daily on a fixed mega-cap US universe.
2. Cross-sectionally z-score each signal.
3. Blend the three z-scores into a daily alpha forecast.
4. Build a dollar-neutral L/S portfolio: long the top decile, short the bottom decile.
5. Trade with t+1 execution and 5 bps per side + 2 bps slippage on turnover.

The blender is either equal-weighted or a **walk-forward Ridge regression**
fit on a 3-year rolling window of (lagged signal z-scores, realized 21d forward
returns), refit monthly. The Ridge approach lets the signals' relative
importance adapt to regime, while still being defensible in an interview as
a "linear, interpretable, low-capacity" model that is hard to overfit.

---

## Lookahead / leakage hygiene

The single most important property of a backtest. Three controls:

1. **All signals at date `t` use only price history up to and including `t`.** Tested in `tests/test_no_lookahead.py` by perturbing prices after `t` and asserting that no signal value at `≤ t` changes.
2. **The Ridge model at refit date `d` is trained only on rows where the target's 21-day forward return has already realized** (i.e. `t ≤ d - 21`). See `src/alpha.py`.
3. **Execution lag.** The backtester applies a configurable `delay_days = 1`: alpha computed on `close(t)` is traded at `close(t+1)`.

---

## Data and survivorship bias

Prices come from Yahoo Finance via `yfinance`, cached locally as parquet.
The universe is a fixed list of ~50 US mega-caps defined in `config.yaml`.

**This list has survivorship bias.** It is the set of names that were
large-cap *as of project creation*; a strategy "trained" on it implicitly
knows that none of them disappeared. The right institutional fix is a
point-in-time universe (Compustat / CRSP). For this showcase, the
calibration is documented as a known limitation rather than hidden.

---

## Project layout

```
quant_research/
├── README.md                  ← this file
├── config.yaml                ← universe, signals, costs, risk-free
├── requirements.txt
├── src/
│   ├── data.py                ← cached price loader
│   ├── signals.py             ← momentum, low-vol, trend-quality, cross-sectional z-score
│   ├── alpha.py               ← equal blend / walk-forward Ridge blend
│   ├── portfolio.py           ← decile portfolios; TF max-Sharpe overlay
│   ├── backtest.py            ← t+1 event loop with costs & slippage
│   ├── metrics.py             ← Sharpe, Sortino, MDD, IC, IR, turnover, yearly breakdown
│   └── plotting.py            ← equity curve, drawdown, rolling Sharpe, IC, yearly bars
├── scripts/run_backtest.py    ← entry point
├── tests/                     ← no-lookahead, signal, metrics tests (pytest)
├── data/cache/                ← parquet-cached prices (gitignored)
└── reports/                   ← PNG plots + Excel summary (gitignored)
```

---

## Running it

```bash
# one-time setup (Apple Silicon: use a native arm64 python)
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt

# full backtest, generates reports/ artifacts
./.venv/bin/python scripts/run_backtest.py

# refetch prices (skip the parquet cache)
./.venv/bin/python scripts/run_backtest.py --no-cache

# tests
./.venv/bin/pytest -q
```

Tweak `config.yaml` to switch blender (`equal` vs `ridge`), change the
weighting scheme (`equal` deciles vs `tf_max_sharpe`), or adjust costs.

---

## Outputs

`scripts/run_backtest.py` prints a summary table and writes:

- `reports/equity_curve.png` — strategy vs equal-weight long-only benchmark
- `reports/drawdown.png` — underwater chart
- `reports/rolling_sharpe.png` — 252d rolling Sharpe (regime view)
- `reports/yearly_returns.png` — yearly bar chart
- `reports/ic.png` — 63d rolling Spearman IC of the blended alpha
- `reports/backtest_report.xlsx` — Summary, Turnover, IC, Yearly, Weights sheets

Headline metrics reported:

- **Strategy block**: AnnReturn, AnnVol, Sharpe (+ Lo (2002) SE), Sortino, MaxDrawdown, CAGR, HitRate
- **Signal block**: mean IC, IC vol, IR = IC mean / IC std, t-stat
- **Cost block**: annualized turnover, max daily turnover

---

## Defending it in an interview

A few questions to expect and where the answers live:

> *"How do you know you don't have lookahead?"* → `tests/test_no_lookahead.py` perturbs future prices and asserts past signal values are unchanged. The Ridge refit pulls only fully-realized training rows (`src/alpha.py`).

> *"What is your turnover, and have you stress-tested costs?"* → Reported in the Excel `Turnover` sheet; `config.yaml` exposes `cost_bps_per_side` and `slippage_bps` so you can re-run at 2× / 5× costs.

> *"What's your IC and is it statistically significant?"* → `IC` sheet reports mean, std, IR, and t-stat across all backtest days.

> *"Is this just momentum?"* → Compare with `blend.method = equal` and with each single signal isolated (set the other signal frames to zero in `src/alpha.py:equal_blend`). The Ridge coefficients (logged from `src/alpha.py` if you instrument them) show the actual weights.

> *"What changes out-of-sample?"* → The walk-forward design means every prediction is OOS by construction; the rolling-Sharpe and per-year tables show regime stability.

> *"Why TensorFlow?"* → The TF optimizer in `src/tf_max_sharpe.py` parameterizes long-only weights via `softmax(logits)` and solves max-Sharpe by Adam — a clean demonstration of unconstrained gradient-based portfolio optimization. Set `portfolio.weighting = tf_max_sharpe` in the config to use it instead of equal-weight within deciles.

---

## Known limitations (be upfront)

- **Survivorship bias** in the fixed universe (above).
- **Daily-close fills.** No intraday liquidity model; fills are at posted close.
- **Static cost model.** Costs do not scale with order size or volatility regime.
- **No risk model.** Sector and factor exposures (beta, size, value) are not neutralized. The strategy's profits could be a leveraged compensation for known factor exposures rather than uncorrelated alpha.
- **No multiple-testing correction.** Three signals on one universe is small, but the Ridge alpha was specified before running the backtest — no parameter sweep is reported here.
- **Yahoo data quality.** Splits/dividends are auto-adjusted but corporate actions and delisted constituents are imperfect compared with institutional vendors.

A real production version would replace the universe with a point-in-time
constituent list, add a factor-risk model (e.g., Barra-style), and either
deploy a proper transaction-cost model or run on liquidity-screened names
only. The architecture here is built to accept those drop-in replacements.
