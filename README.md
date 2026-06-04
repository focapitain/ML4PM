# Portfolio Management Project

## Overview

This repository contains two main Markowitz workflows and a reusable walk-forward backtesting module:

- [Markowitz2-2.ipynb](./Markowitz2-2.ipynb): pedagogical notebook (core Markowitz formulas, Monte Carlo frontier, first experiments).
- [markowitz_sp500_V2.ipynb](./markowitz_sp500_V2.ipynb): modular experimental framework on S&P 500 with interchangeable estimators/objectives/optimizers.
- [backtest.py](./backtest.py): reusable walk-forward backtest engine.

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
