# Portfolio Management Project

## Overview

This repository now includes three complementary workflows around portfolio construction and evaluation:

- [Markowitz2-2.ipynb](./Markowitz2-2.ipynb): pedagogical notebook (core Markowitz formulas, Monte Carlo frontier, first experiments).
- [markowitz_sp500_V2.ipynb](./markowitz_sp500_V2.ipynb): modular experimental framework on S&P 500 with interchangeable estimators/objectives/optimizers.
- [backtest.py](./backtest.py): reusable walk-forward backtest engine.
- [benchmark_demo/](./benchmark_demo): interactive Streamlit demo to run, visualize, and explain benchmark simulations end-to-end.

---

## Markowitz in brief

Modern Portfolio Theory (Harry Markowitz) frames portfolio construction as a trade-off between
**expected return** and **risk**. Instead of picking assets one by one, it optimizes portfolio
weights jointly, using both each asset's volatility and how assets move together.

At a high level, if $w$ is the vector of portfolio weights, $\mu$ the vector of expected returns,
and $\Sigma$ the covariance matrix of returns:

- Expected portfolio return: $\mathbb{E}[R_p] = w^T\mu$
- Portfolio variance (risk): $\sigma_p^2 = w^T\Sigma w$
- Portfolio volatility: $\sigma_p = \sqrt{w^T\Sigma w}$

The key insight is diversification: two risky assets can reduce total risk if their covariance is
low (or negative).

Typical optimization problems used in this project are:

1. **Minimum variance for a target return**

$$
\min_w \; w^T\Sigma w
\quad \text{s.t.} \quad
w^T\mu = \mu_{\text{target}},\; \mathbf{1}^T w = 1
$$

2. **Maximum risk-adjusted return (Sharpe-style objective)**

$$
\max_w \; \frac{w^T\mu - r_f}{\sqrt{w^T\Sigma w}}
\quad \text{s.t. constraints (e.g. long-only, weight caps)}
$$

In practice, the project estimates $\mu$ and $\Sigma$ from historical data, solves one of these
optimization problems under constraints, then evaluates the result out-of-sample with
walk-forward backtesting.

---

## Notebook 1: Markowitz2-2

Main goal: build intuition and verify Markowitz mechanics on a smaller setup.

What it does:

- Downloads a subset of S&P 500 tickers.
- Computes historical return vector and covariance matrix.
- Implements closed-form optimization functions (`optimal1`, `optimal2`).
- Compares analytical frontier points with a Monte Carlo efficient frontier.

Typical use case:

- Learning/debugging the optimization logic before moving to the larger framework.

---

## Notebook 2: markowitz_sp500_V2

Main goal: run configurable, reproducible Markowitz experiments at larger scale.

What it does:

- Centralizes experiment settings in `ExperimentConfig`.
- Supports multiple universe selectors:
  - all universe
  - top N
  - random N
  - sector-stratified sample
- Supports return modes and frequencies.
- Supports multiple covariance estimators:
  - empirical
  - rolling
  - EWMA
  - Ledoit-Wolf
- Supports multiple objectives and optimizers.
- Generates summary tables and visual diagnostics.

Typical use case:

- Comparative research/benchmarking across estimators, objectives, and optimization methods.

---

## Backtesting Module (`backtest.py`)

What it does:

- Splits history into rolling train/test periods.
- Fits portfolio weights on each training window.
- Holds those weights for the next holding period.
- Repeats this through time to simulate realistic rebalancing.
- Applies transaction costs based on turnover.
- Tracks strategy NAV and an equal-weight benchmark NAV.
- Returns a history table, rebalance weight history, and summary metrics.

Main metrics include:

- Total return and CAGR
- Annualized volatility and Sharpe ratio
- Max drawdown
- Tracking error vs benchmark
- Average turnover and rebalance count

---

## Benchmark demo (Streamlit)

The benchmark demo is an interactive app that showcases how a Markowitz workflow is executed in practice, from data loading to out-of-sample evaluation.

### Core assumptions

- **Modern Portfolio Theory setup**: portfolio construction is based on expected returns $\mu$, covariance matrix $\Sigma$, and constraints.
- **Walk-forward out-of-sample protocol**: the model is estimated on a rolling training window, then evaluated on a future holding window (anti look-ahead logic).
- **Transaction costs matter**: turnover is penalized through transaction costs to avoid unrealistic gross-performance results.
- **Risk-adjusted evaluation**: diagnostics focus on risk/return trade-offs (volatility, drawdown, Sharpe, turnover), not return alone.
- **Data regime awareness**: replay presets (e.g., stress periods) help test robustness outside calm-market conditions.

### Models and methods used

- **Universe construction**: configurable asset selection (all, top-$N$, random, sector-stratified) to test concentration vs diversification effects.
- **Return estimators ($\mu$)**: sample mean and robust alternatives depending on the selected method.
- **Risk estimators ($\Sigma$)**: empirical and shrinkage/smoothed covariance estimators (for example Ledoit-Wolf/EWMA-style choices in the broader project workflows).
- **Optimization objectives**: minimum variance and risk-adjusted formulations (e.g., max-Sharpe style objective under constraints).
- **Portfolio constraints**: long-only settings, weight caps, and diversification sanity checks.

### What the demo helps you answer

1. Which estimator/objective pair is most stable out-of-sample for a given universe?
2. How sensitive are results to window length, rebalance cadence, and costs?
3. Are strong in-sample results preserved once turnover and drawdown are considered?
4. How does behavior change across different market regimes?

### How it works in practice

1. Configure assumptions from the sidebar (universe, model choices, protocol choices).
2. Click **"Lancer la simulation"** to run the backtest.
3. Inspect the decision timeline, NAV path, and risk diagnostics.
4. Compare outcomes and export a report for reproducible interpretation.

### How to run it

From the project root:

```bash
pip install -r requirements.txt -r benchmark_demo/requirements-demo.txt
cd benchmark_demo
streamlit run Cockpit.py
```

Then open the local Streamlit URL shown in the terminal.

Notes:

- By default, the demo uses live yfinance data (internet required).
- For offline or classroom/demo fallback, enable the synthetic data option in the app.

---
