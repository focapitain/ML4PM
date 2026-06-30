
# Portfolio Management Project

## Objective

This project, developed as part of the Data Science track at Télécom Paris, implements a portfolio management strategy based on **Markowitz's Modern Portfolio Theory (MPT)**. 

Within this repository, you will find:
* **A complete [report](Report__ML_based_portfolio_management.pdf)** written in LaTeX.
* **Python [scripts](Graph_script.ipynb)** used to generate the visual diagnostics and graphs for Parts 1 and 2 of the report.
* `markowitz_sp500`: [Scripts](markowitz_sp500_evaluation_V3.ipynb) to generate the graphs for Part 3 and the core implementation of the strategy.
* `benchmark_demo`: An [interactive Streamlit application](benchmark_demo) to simulate and visualize the strategy in real-time (a dedicated README inside this directory explains how to launch it).
* `legacy_code`: A [collection of scripts](legacy_code) used during the research, testing, and brainstorming phases.

---

## Overview of Key Concepts

### Markowitz in Brief

Modern Portfolio Theory (Harry Markowitz) frames portfolio construction as a trade-off between **expected return** and **risk**. Instead of selecting assets individually, it optimizes portfolio weights jointly by considering both individual asset volatility and their co-movements (covariance).

At a high level, if $w$ is the vector of portfolio weights, $\mu$ the vector of expected returns, and $\Sigma$ the covariance matrix of returns:

* **Expected portfolio return:** $\mathbb{E}[R_p] = w^T\mu$
* **Portfolio variance (risk):** $\sigma_p^2 = w^T\Sigma w$
* **Portfolio volatility:** $\sigma_p = \sqrt{w^T\Sigma w}$

The core insight is **diversification**: combining risky assets can reduce total portfolio risk if their covariance is low or negative.

#### Optimization Problems Implemented:

1. **Minimum Variance for a Target Return**

$$
\min_w \; w^T\Sigma w
\quad \text{s.t.} \quad
w^T\mu = \mu_{\text{target}},\; \mathbf{1}^T w = 1
$$

2. **Maximum Risk-Adjusted Return (Sharpe-style Objective)**

$$
\max_w \; \frac{w^T\mu - r_f}{\sqrt{w^T\Sigma w}}
\quad \text{s.t. constraints (e.g., long-only, weight caps)}
$$

In practice, the pipeline estimates $\mu$ and $\Sigma$ from historical data, solves the optimization problem under specific constraints, and evaluates performance out-of-sample using a **walk-forward backtesting** protocol.

---

## Main Modules

### 1. `markowitz_sp500_V3`

**Goal:** Run highly configurable, scalable, and reproducible Markowitz experiments.

**Features:**
* **Centralized Configuration:** Settings are easily managed via `ExperimentConfig`.
* **Universe Selectors:** Supports full universe, top-$N$, random-$N$, and sector-stratified sampling.
* **Flexible Returns:** Supports multiple return modes and frequencies.
* **Covariance Estimators:** Implements Empirical, Rolling, EWMA, and Ledoit-Wolf shrinkage estimators.
* **Optimization:** Supports various objectives and optimizers.
* **Analytics:** Generates comprehensive summary tables and visual diagnostics.

> **Use Case:** Conducting comparative research and benchmarking across different estimators, objectives, and optimization constraints.

### 2. Benchmark Demo (Streamlit App)

The benchmark demo is an interactive web application that showcases the entire Markowitz workflow in practice, from data loading to out-of-sample evaluation.

#### Core Assumptions & Logic
* **MPT Setup:** Portfolio construction relies on expected returns $\mu$, the covariance matrix $\Sigma$, and specific constraints.
* **Walk-Forward Protocol:** To avoid look-ahead bias, the model is trained on a rolling historical window and evaluated on a future holding window.
* **Transaction Costs:** Turnover is penalized to reflect realistic net performance rather than unrealistic gross returns.
* **Risk-Adjusted Evaluation:** Diagnostics focus on risk/return trade-offs (volatility, maximum drawdown, Sharpe ratio, turnover) rather than absolute returns alone.
* **Market Regimes:** Includes replay presets (e.g., historical stress periods) to test strategy robustness under volatile conditions.

#### Features & Components
* **Universe Construction:** Configurable asset selection to analyze the effects of concentration versus diversification.
* **Estimators:** Sample mean and robust alternatives for returns ($\mu$); empirical and shrinkage/smoothed methods (Ledoit-Wolf, EWMA) for risk ($\Sigma$).
* **Constraints:** Long-only constraints, weight caps, and diversification sanity checks.
* **Baselines & Benchmarks:** Every Net Asset Value (NAV) chart compares the strategy against an **Equal-Weight ($1/N$) portfolio**, the **S&P 500 Index**, and a **Random-Weights** ("zero-intelligence") portfolio.

#### Key Questions This Demo Answers:
1. Which estimator/objective pair proves most stable out-of-sample for a given universe?
2. How sensitive are the results to training window length, rebalancing cadence, and transaction fees?
3. Are strong in-sample results preserved once turnover and drawdowns are taken into account?
4. How does the portfolio behave across different market regimes?

---

## Getting Started

### How to Run the Streamlit Demo

1. **Install dependencies** from the root directory:
   ```bash
   pip install -r requirements.txt -r benchmark_demo/requirements-demo.txt

