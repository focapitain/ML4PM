from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import portfolio_lab as pl
from portfolio_lab import plotting as pl_plot
from portfolio_lab.backtest import WalkForwardBacktestConfig, plot_backtest_nav


def main() -> None:
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Offline-safe synthetic universe (40 assets, 4 sectors)
    sectors = (["Tech"] * 10 + ["Fin"] * 10 + ["Health"] * 10 + ["Energy"] * 10)
    constituents = pd.DataFrame({"sector": sectors}, index=[f"A{i:02d}" for i in range(40)])
    prices = pl.make_synthetic_prices(constituents, start="2019-01-01", end="2025-12-31", seed=7)

    cfg = pl.ExperimentConfig(
        universe=pl.StratifiedBySector(20),
        frequency="daily",
        return_mode="simple",
        cov_estimator=pl.EWMACovariance(0.94),
        mean_estimator=pl.SampleMean(),
        objective="max_sharpe",
        optimizer=pl.CVXPYOptimizer(),
        long_only=True,
        w_max=0.15,
        risk_free_annual=0.04,
    )

    # In-sample result + built-in visuals
    res = pl.run_experiment(cfg, constituents=constituents, prices=prices)

    fig, ax = plt.subplots(figsize=(11, 5))
    pl_plot.plot_weights(res, ax=ax)
    fig.savefig(out_dir / "weights.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7))
    pl_plot.plot_efficient_frontier(res, ax=ax)
    fig.savefig(out_dir / "efficient_frontier.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    pl_plot.plot_correlation(res, ax=ax)
    fig.savefig(out_dir / "correlation_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    pl_plot.plot_performance(res, ax=ax)
    fig.savefig(out_dir / "insample_performance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Walk-forward backtest + NAV visual
    bt_cfg = WalkForwardBacktestConfig(
        estimation_window=252,
        holding_period=21,
        transaction_cost_bps=10.0,
        frequency="daily",
        risk_free_annual=0.04,
    )
    wf = pl.run_walk_forward(prices, cfg, backtest_config=bt_cfg, label="Markowitz")

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_backtest_nav({"Markowitz": wf}, ax=ax)
    fig.savefig(out_dir / "oos_backtest_nav.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Drawdown visual for report-ready risk chart
    nav = wf.history["strategy_nav"].dropna()
    drawdown = nav / nav.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(drawdown.index, drawdown.values, color="firebrick", lw=1.5)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="firebrick", alpha=0.25)
    ax.set_title("Strategy Drawdown (Out-of-Sample)")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.25)
    fig.savefig(out_dir / "oos_drawdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print("Generated figures:")
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".png"):
            print(f"- figures/{name}")


if __name__ == "__main__":
    main()
