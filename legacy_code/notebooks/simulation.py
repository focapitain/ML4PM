"""
simulation.py — Quel estimateur de covariance détecte le plus vite un choc ?
============================================================================
Expérience de SIMULATION PURE (hors-ligne, vérité connue) : on génère des rendements
synthétiques stationnaires, puis on INJECTE à une date connue `t_shock` un choc de
volatilité (la vol passe de σ₀ à k·σ₀). Comme on connaît la vérité, on peut mesurer
exactement le DÉLAI DE DÉTECTION de chaque estimateur de Σ : le nombre de jours après
le choc avant que sa volatilité prévue n'atteigne la moitié du chemin vers le nouveau
niveau.

Attendu : les estimateurs réactifs (EWMA, IEWMA) détectent le choc bien plus tôt que
l'empirique ou la fenêtre glissante longue, au prix d'un peu plus de bruit.

Usage :  python notebooks/simulation.py
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import portfolio_lab as pl

PPY = 252
SEED = 42


# --------------------------------------------------------------------------- #
#  1. Données synthétiques AVEC choc de volatilité injecté (vérité connue)     #
# --------------------------------------------------------------------------- #
def make_returns_with_vol_shock(n_assets=20, T=750, t_shock=375, k=3.0,
                                sigma0=0.010, seed=SEED):
    """Modèle 1-facteur : r_{t,i} = β_i·marché_t + idiosyncratique. La volatilité (marché
    ET idio) est multipliée par `k` à partir de `t_shock`. Renvoie (returns, vol_true_ann)
    où vol_true_ann est la volatilité moyenne VRAIE annualisée par date (en escalier)."""
    rng = np.random.default_rng(seed)
    scale = np.ones(T)
    scale[t_shock:] = k                                   # choc : ×k sur la vol
    betas = rng.uniform(0.8, 1.2, n_assets)
    idio0 = sigma0 * 0.5
    market = rng.normal(0.0, sigma0, T) * scale
    idio = rng.normal(0.0, idio0, (T, n_assets)) * scale[:, None]
    rets = market[:, None] * betas[None, :] + idio
    dates = pd.bdate_range("2021-01-01", periods=T)
    cols = [f"A{i:02d}" for i in range(n_assets)]
    returns = pd.DataFrame(rets, index=dates, columns=cols)
    # Vol moyenne VRAIE par actif (annualisée), en escalier autour du choc.
    asset_var0 = (betas ** 2) * sigma0 ** 2 + idio0 ** 2
    v0 = float(np.sqrt(asset_var0).mean()) * np.sqrt(PPY)
    vol_true = pd.Series(v0, index=dates)
    vol_true.iloc[t_shock:] = v0 * k
    return returns, vol_true


# --------------------------------------------------------------------------- #
#  2. Volatilité prévue par chaque estimateur, au fil du temps                 #
# --------------------------------------------------------------------------- #
def forecast_vol_path(returns, estimator, window=120, step=3):
    """Pour chaque date t (pas `step`), estime Σ̂ sur [t-window, t] et renvoie la vol
    moyenne prévue annualisée = moyenne_i √(Σ̂_ii)·√PPY."""
    idx, vals = [], []
    for t in range(window, len(returns), step):
        win = returns.iloc[t - window:t]
        try:
            S = np.asarray(estimator.estimate(win))
            v = float(np.sqrt(np.maximum(np.diag(S), 0)).mean()) * np.sqrt(PPY)
        except Exception:
            v = np.nan
        idx.append(returns.index[t]); vals.append(v)
    return pd.Series(vals, index=idx)


def detection_lag(path, dates, t_shock_date, v0, v1, step):
    """Nb de JOURS entre le choc et le moment où la vol prévue franchit la mi-chemin
    (v0+v1)/2. NaN si jamais franchi sur l'échantillon."""
    target = 0.5 * (v0 + v1)
    after = path[path.index >= t_shock_date]
    crossed = after[after >= target]
    if crossed.empty:
        return np.nan
    return int((crossed.index[0] - t_shock_date).days)


# --------------------------------------------------------------------------- #
#  3. Expérience                                                               #
# --------------------------------------------------------------------------- #
def main():
    T, T_SHOCK, K, WINDOW, STEP = 750, 375, 3.0, 120, 3
    returns, vol_true = make_returns_with_vol_shock(T=T, t_shock=T_SHOCK, k=K)
    shock_date = returns.index[T_SHOCK]
    v0 = float(vol_true.iloc[0]); v1 = float(vol_true.iloc[-1])
    print(f"Choc de volatilité au {shock_date.date()} : {v0:.0%} -> {v1:.0%} annualisé "
          f"(×{K}). Fenêtre d'estimation = {WINDOW} j.")

    estimators = {
        "Empirique":      pl.EmpiricalCovariance(),
        "Rolling-60":     pl.RollingCovariance(window=60),
        "Ledoit-Wolf":    pl.LedoitWolfCovariance(),
        "EWMA λ=0.97":    pl.EWMACovariance(lam=0.97),
        "EWMA λ=0.94":    pl.EWMACovariance(lam=0.94),
        "IEWMA (réactif)": pl.IEWMACovariance(vol_halflife=20, corr_halflife=10),
    }

    paths, rows = {}, []
    for name, est in estimators.items():
        p = forecast_vol_path(returns, est, window=WINDOW, step=STEP)
        paths[name] = p
        lag = detection_lag(p, p.index, shock_date, v0, v1, STEP)
        rows.append({"estimateur": name, "délai détection (j)": lag,
                     "vol prévue finale": round(float(p.dropna().iloc[-1]), 3)})
    table = pd.DataFrame(rows).set_index("estimateur").sort_values("délai détection (j)")
    print("\n=== Délai de détection du choc (↓ = détecte plus vite) ===")
    print(table.to_string())
    best = table["délai détection (j)"].idxmin()
    print(f"\nLe plus rapide : {best} ({table.loc[best, 'délai détection (j)']} jours). "
          "EWMA/IEWMA réagissent avant l'empirique, qui doit « oublier » l'ancien régime.")

    # --- Figure : vol prévue par estimateur + vol vraie (escalier) + choc ---
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = {"Empirique": "#9aa0a6", "Rolling-60": "#6366F1", "Ledoit-Wolf": "#14B8A6",
              "EWMA λ=0.97": "#F59E0B", "EWMA λ=0.94": "#EF4444", "IEWMA (réactif)": "#A855F7"}
    for name, p in paths.items():
        ax.plot(p.index, p.values * 100, lw=1.8, label=name, color=colors.get(name))
    ax.step(vol_true.index, vol_true.values * 100, where="post", color="black", lw=2.2,
            ls="--", label="Volatilité VRAIE")
    ax.axvline(shock_date, color="#EF4444", lw=1.5, alpha=.7)
    ax.text(shock_date, ax.get_ylim()[1], "  choc ×%.0f" % K, color="#EF4444", va="top")
    ax.set_title("Détection d'un choc de volatilité : vol prévue par estimateur vs vol vraie")
    ax.set_ylabel("Volatilité moyenne annualisée (%)"); ax.legend(ncol=2); ax.grid(alpha=.3)
    out = os.path.join(os.path.dirname(__file__), "simulation_detection.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nFigure écrite : {out}")


if __name__ == "__main__":
    main()
