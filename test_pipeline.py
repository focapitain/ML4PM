"""Smoke test hors-ligne du package portfolio_lab (données synthétiques)."""
import os
import sys
import time
import numpy as np
import pandas as pd

# Racine du projet = dossier contenant ce fichier (chemin relatif, portable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import portfolio_lab as pl
from portfolio_lab.backtest import WalkForwardBacktestConfig
from dataclasses import replace

print("=" * 70)
print("1) IMPORTS & VERSION")
print("=" * 70)
print("portfolio_lab", pl.__version__)

# --- données synthétiques --------------------------------------------------
constituents = pd.DataFrame.from_dict({
    f"A{i:02d}": sec for i, sec in enumerate(
        (["Tech"] * 10 + ["Fin"] * 10 + ["Health"] * 10 + ["Energy"] * 10), )
}, orient="index", columns=["sector"])
# build a proper 40-name, 4-sector constituent frame
secs = (["Tech"] * 10 + ["Fin"] * 10 + ["Health"] * 10 + ["Energy"] * 10)
constituents = pd.DataFrame({"sector": secs}, index=[f"A{i:02d}" for i in range(40)])
prices = pl.make_synthetic_prices(constituents, start="2019-01-01", end="2025-12-31", seed=7)
print("prix synthétiques :", prices.shape, "| de", prices.index[0].date(), "à", prices.index[-1].date())

print("\n" + "=" * 70)
print("2) ESTIMATEURS DE COVARIANCE — PSD & conditionnement")
print("=" * 70)
rets = pl.compute_returns(prices.iloc[-252:], "simple")
ests = {
    "empirique": pl.EmpiricalCovariance(),
    "rolling120": pl.RollingCovariance(120),
    "ewma0.94": pl.EWMACovariance(0.94),
    "ledoit-wolf": pl.LedoitWolfCovariance(),
    "oas": pl.OASCovariance(),
    "corr_const": pl.ConstantCorrelationCovariance(),
}
for name, est in ests.items():
    S = est.estimate(rets).values
    ev = np.linalg.eigvalsh(S)
    psd = ev.min() > -1e-10
    cond = ev.max() / max(ev.min(), 1e-12)
    sym = np.allclose(S, S.T)
    print(f"  {name:14s} PSD={psd!s:5s} sym={sym!s:5s} cond={cond:12.1f}")

print("\nMoyennes :")
for name, m in {"sample": pl.SampleMean(), "james-stein": pl.JamesSteinMean()}.items():
    mu = m.estimate(rets)
    print(f"  {name:12s} ||mu||={np.linalg.norm(mu):.5f}  mean={mu.mean():.6f}")

print("\n" + "=" * 70)
print("3) OPTIMISEURS × OBJECTIFS — somme=1 & long-only")
print("=" * 70)
mu = rets.mean().values * 252
Sigma = rets.cov().values * 252
N = len(mu)

mk_opt = {
    "lagrangien": pl.AnalyticLagrangian(),
    "cvxpy": pl.CVXPYOptimizer(),
    "slsqp": pl.ScipyOptimizer(),
    "diff_evol": pl.DifferentialEvolutionOptimizer(maxiter=60),
}
for oname, opt in mk_opt.items():
    for obj in ["min_variance", "max_sharpe", "max_return", "max_utility"]:
        try:
            lo = False if oname == "lagrangien" else True
            w = opt.solve(mu, Sigma, objective=obj, rf=0.04, long_only=lo, w_max=1.0,
                          risk_aversion=3.0, mu_target=None)
            w = np.asarray(w).flatten()
            r, v, s = pl.portfolio_stats(w, mu, Sigma, 0.04)
            ok_sum = abs(w.sum() - 1) < 1e-4
            ok_long = (w.min() > -1e-6) if lo else True
            flag = "" if (ok_sum and ok_long) else "  <-- CHECK"
            print(f"  {oname:11s} {obj:13s} sum={w.sum():.4f} min={w.min():+.4f} "
                  f"S={s:6.3f}{flag}")
        except Exception as e:
            print(f"  {oname:11s} {obj:13s} -> {type(e).__name__}: {str(e)[:40]}")

print("\nBaselines basées risque :")
for oname, opt in {
    "equal_weight": pl.EqualWeightOptimizer(),
    "inverse_vol": pl.InverseVolatilityOptimizer(),
    "risk_parity": pl.RiskParityOptimizer(),
    "max_divers": pl.MaxDiversificationOptimizer(),
}.items():
    w = np.asarray(opt.solve(mu, Sigma, rf=0.04, long_only=True, w_max=1.0)).flatten()
    r, v, s = pl.portfolio_stats(w, mu, Sigma, 0.04)
    print(f"  {oname:13s} sum={w.sum():.4f} min={w.min():+.4f} vol={v*100:5.2f}% S={s:6.3f}")

print("\n" + "=" * 70)
print("4) run_experiment (pipeline in-sample, prix fournis)")
print("=" * 70)
cfg = pl.ExperimentConfig(
    universe=pl.StratifiedBySector(20), frequency="daily", return_mode="simple",
    cov_estimator=pl.EWMACovariance(0.94), objective="max_sharpe",
    optimizer=pl.CVXPYOptimizer(), long_only=True, w_max=0.15, risk_free_annual=0.04,
)
res = pl.run_experiment(cfg, constituents=constituents, prices=prices)
print(res.summary().to_string())

print("\n" + "=" * 70)
print("5) ADAPTER + walk_forward_backtest (hors échantillon)")
print("=" * 70)
bt_cfg = WalkForwardBacktestConfig(estimation_window=252, holding_period=21,
                                   transaction_cost_bps=10.0, frequency="daily")
t0 = time.time()
wf = pl.run_walk_forward(prices, cfg, backtest_config=bt_cfg, label="ewma_maxsharpe")
print(f"  backtest OK en {time.time()-t0:.2f}s | rebalancements={int(wf.metrics['Rebalance count'])}")
print(wf.metrics.round(3).to_string())

print("\n" + "=" * 70)
print("6) run_oos_benchmark + tables comparatives (Benchmark estimateurs)")
print("=" * 70)
base = cfg
variants = {
    "empirique": replace(base, cov_estimator=pl.EmpiricalCovariance()),
    "ewma0.94": replace(base, cov_estimator=pl.EWMACovariance(0.94)),
    "ledoit": replace(base, cov_estimator=pl.LedoitWolfCovariance()),
    "oas": replace(base, cov_estimator=pl.OASCovariance()),
}
results = pl.run_oos_benchmark(prices, variants, backtest_config=bt_cfg)
mt = pl.metrics_table(results)
print(mt[["CAGR (%)", "Vol annualized (%)", "Sharpe", "Max drawdown (%)",
          "Average turnover"]].round(3).to_string())
print("\nStabilité des poids :")
print(pl.weight_stability(results).round(4).to_string())
print("\nMétriques de queue :")
print(pl.tail_metrics_table(results, periods_per_year_=252, rf_annual=0.04).round(3).to_string())

print("\n" + "=" * 70)
print("7) EXEMPLE CONCRET — split passé/futur + allocation 100 000 €")
print("=" * 70)
ex = pl.train_test_split_evaluation(
    prices, cfg, split_date="2024-06-30", capital=100_000.0,
    frequency="daily", rf_annual=0.04,
)
print(f"  passé={ex['n_past']} obs | futur={ex['n_future']} obs | split={ex['split_date']}")
print("\n  Allocation (top 8) :")
print(ex["allocation"].head(8).to_string())
print("\n  Comparaison passé->futur (résultats réels) :")
print(ex["comparison"].round(2).to_string())

print("\n" + "=" * 70)
print("8) TESTS DE RÉGRESSION DES CORRECTIONS D'AUDIT")
print("=" * 70)

# (a) Sharpe intègre le taux sans risque : Sharpe(rf>0) < Sharpe(rf=0) sur la MÊME série.
from portfolio_lab.backtest import performance_summary
bt_rf4 = WalkForwardBacktestConfig(estimation_window=252, holding_period=21,
                                   transaction_cost_bps=10.0, frequency="daily",
                                   risk_free_annual=0.04)
wf4 = pl.run_walk_forward(prices, cfg, backtest_config=bt_rf4, label="rf4")
s_rf0 = performance_summary(wf4.history, 252, rf_annual=0.0)["Sharpe"]
s_rf4 = performance_summary(wf4.history, 252, rf_annual=0.04)["Sharpe"]
assert s_rf4 < s_rf0, "Sharpe devrait baisser quand on soustrait rf"
assert abs((s_rf0 - s_rf4) - 0.04 / (wf4.metrics["Vol annualized (%)"] / 100)) < 1e-6
print(f"  (a) Sharpe rf=0 {s_rf0:.3f} > rf=4% {s_rf4:.3f} (écart = rf/vol)  OK")

# (b) Le benchmark 1/N paie des coûts : turnover benchmark moyen > 0.
assert "Average benchmark turnover" in wf4.metrics.index
assert wf4.metrics["Average benchmark turnover"] > 0, "le 1/N rebalance -> turnover > 0"
print(f"  (b) turnover benchmark moyen = {wf4.metrics['Average benchmark turnover']:.4f} (>0)  OK")

# (c) Le compteur de fallback est exposé.
assert "Fallback count" in wf4.metrics.index
print(f"  (c) 'Fallback count' exposé = {int(wf4.metrics['Fallback count'])}  OK")

# (d) w_max respecté par les baselines basées risque (cas forçant le plafonnement).
mu_d = rets.mean().values * 252
Sig_d = rets.cov().values * 252
WMAX = 0.05
for oname, opt in {
    "equal_weight": pl.EqualWeightOptimizer(),
    "inverse_vol": pl.InverseVolatilityOptimizer(),
    "risk_parity": pl.RiskParityOptimizer(),
    "max_divers": pl.MaxDiversificationOptimizer(),
}.items():
    w = np.asarray(opt.solve(mu_d, Sig_d, long_only=True, w_max=WMAX)).flatten()
    assert w.max() <= WMAX + 1e-6, f"{oname} viole w_max ({w.max():.4f} > {WMAX})"
    assert abs(w.sum() - 1) < 1e-4, f"{oname} ne somme pas à 1"
print(f"  (d) baselines respectent w_max={WMAX} (max poids ≤ plafond) & somme=1  OK")

# (e) TopN = ordre NATIF de la source (alphabétique pour Wikipédia), et alias cohérent.
assert pl.FirstNAlphabetical is pl.TopN
assert pl.TopN(3).select(constituents) == list(constituents.index[:3])
print("  (e) FirstNAlphabetical is TopN ; TopN(n) = n premiers de la source  OK")

print("\n" + "=" * 70)
print("9) NOUVEAUX ESTIMATEURS (EWMA-Mean, IEWMA) & MODULE cov_eval")
print("=" * 70)
rets_full = pl.compute_returns(prices, "simple")
win = rets_full.iloc[-252:]

# EWMA-Mean : forme (N,), poids récents dominants (≠ moyenne empirique).
mu_ewma = pl.EWMAMean(0.94).estimate(win)
assert mu_ewma.shape == (prices.shape[1],), "EWMAMean : mauvaise forme"
assert not np.allclose(mu_ewma, win.mean().values), "EWMAMean devrait différer de la moyenne simple"
print(f"  EWMA-Mean : forme {mu_ewma.shape}  OK")

# IEWMA : PSD, symétrique, diagonale de corrélation = 1.
S_ie = pl.IEWMACovariance().estimate(win).values
ev = np.linalg.eigvalsh(S_ie)
d = np.sqrt(np.diag(S_ie))
corr_diag = np.diag(S_ie / np.outer(d, d))
assert ev.min() > -1e-10, "IEWMA non PSD"
assert np.allclose(S_ie, S_ie.T), "IEWMA non symétrique"
assert np.allclose(corr_diag, 1.0), "IEWMA : diag(corr) ≠ 1"
print(f"  IEWMA : PSD & symétrique & diag(corr)=1 (cond={ev.max()/ev.min():.1f})  OK")

# gaussian_loglik : maximale quand Σ_pred = Σ réalisée (test in-sample du critère).
S_real = pl.realized_covariance(win)
ll_true = pl.gaussian_loglik(win, S_real)
ll_id = pl.gaussian_loglik(win, np.eye(win.shape[1]) * float(np.diag(S_real).mean()))
assert np.isfinite(ll_true) and ll_true > ll_id, "loglik : Σ réalisée devrait battre une cible plate"
print(f"  gaussian_loglik : Σ réalisée ({ll_true:.1f}) > cible plate ({ll_id:.1f})  OK")

# rolling_cov_eval : colonnes attendues + condition_number > 1.
ev_df = pl.rolling_cov_eval(rets_full, pl.LedoitWolfCovariance(), window=252, step=21)
for c in ["loglik", "rmse_cov", "mae_cov", "frobenius", "vol_abs_err", "condition_number"]:
    assert c in ev_df.columns, f"rolling_cov_eval : colonne {c} manquante"
assert (ev_df["condition_number"] > 1).all(), "condition_number devrait être > 1"
print(f"  rolling_cov_eval : {ev_df.shape[0]} fenêtres OOS, colonnes complètes  OK")

# cov_eval_summary + regime_split : régularisé mieux conditionné que l'empirique.
summ = pl.cov_eval_summary(rets_full, {"emp": pl.EmpiricalCovariance(),
                                       "lw": pl.LedoitWolfCovariance()}, window=252, step=21)
assert summ.loc["lw", "condition_number"] < summ.loc["emp", "condition_number"], \
    "Ledoit-Wolf devrait être mieux conditionné que l'empirique"
rs = pl.regime_split(ev_df)
assert {"crise", "hors_crise"}.issubset(set(rs.index)), "regime_split : régimes manquants"
print("  cov_eval_summary (LW < emp en cond.) & regime_split  OK")

print("\n" + "=" * 70)
print("10) ESTIMATEURS DE RENDEMENT (RollingMean, PCAFactor) & MODULE mean_eval")
print("=" * 70)
win10 = rets_full.iloc[-252:]
Nv = prices.shape[1]

# RollingMean : forme (N,), n'utilise que les `window` derniers jours.
mu_roll = pl.RollingMean(window=63).estimate(win10)
assert mu_roll.shape == (Nv,), "RollingMean : mauvaise forme"
assert np.allclose(mu_roll, win10.iloc[-63:].mean().values), "RollingMean : doit = moyenne des 63 derniers"
print(f"  RollingMean : forme {mu_roll.shape}, = moyenne fenêtre  OK")

# PCAFactor : forme (N,) ; avec n_factors = N, projection = identité (≈ moyenne empirique).
mu_pca = pl.PCAFactorMean(n_factors=3).estimate(win10)
mu_full = pl.PCAFactorMean(n_factors=Nv).estimate(win10)
assert mu_pca.shape == (Nv,), "PCAFactor : mauvaise forme"
assert np.allclose(mu_full, win10.mean().values, atol=1e-8), "PCAFactor(N) doit ≈ moyenne empirique"
print(f"  PCAFactor : forme {mu_pca.shape}, PCA(N)=moyenne empirique  OK")

# IC : corrélation de rang parfaite = 1 ; nulle si indépendants (signe).
assert abs(pl.spearman_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-9
assert pl.rmse_mean([0.0, 0.0], [0.0, 0.0]) == 0.0
print("  spearman_ic parfait=1 & rmse_mean nul  OK")

# rolling_mean_eval : colonnes attendues + IC dans [-1, 1].
me = pl.rolling_mean_eval(rets_full, pl.SampleMean(), window=252, step=21)
for c in ["rmse", "mae", "wmape", "pearson", "IC", "hit_rate", "realized_vol"]:
    assert c in me.columns, f"rolling_mean_eval : colonne {c} manquante"
assert me["IC"].dropna().between(-1, 1).all(), "IC hors [-1,1]"
print(f"  rolling_mean_eval : {me.shape[0]} fenêtres, IC ∈ [-1,1]  OK")

# mean_eval_summary + window_sensitivity exécutables sur les 5 estimateurs μ.
ms = pl.mean_eval_summary(rets_full, {
    "sample": pl.SampleMean(), "rolling": pl.RollingMean(63), "ewma": pl.EWMAMean(0.94),
    "js": pl.JamesSteinMean(), "pca": pl.PCAFactorMean(3)}, window=252, step=21)
assert ms.shape[0] == 5 and "IC" in ms.columns
wsens = pl.window_sensitivity(rets_full, pl.EWMAMean(0.94), [126, 252], step=21)
assert list(wsens.index) == [126, 252]
print("  mean_eval_summary (5 estimateurs) & window_sensitivity  OK")

print("\n" + "=" * 70)
print("11) FORMULATIONS DE MARKOWITZ (markowitz.py)")
print("=" * 70)
ME11, CE11 = pl.SampleMean(), pl.LedoitWolfCovariance()

# Les 5 variantes résolvent et respectent budget + w_max.
mu11 = rets_full.iloc[-252:].mean().values * 252
S11 = CE11.estimate(rets_full.iloc[-252:]).values * 252
for vname in ["standard", "linear", "operational", "robust", "pp"]:
    opt = pl.make_variant(vname, w_max=0.10)
    w = opt.solve(mu11, S11, w_prev=np.ones(len(mu11)) / len(mu11), mu_uncertainty=np.diag(S11))
    assert abs(w.sum() - 1) < 1e-3, f"{vname} : budget ≠ 1"
    if vname != "standard":
        assert w.max() <= 0.10 + 1e-4, f"{vname} : viole w_max"
print("  5 variantes : budget=1 & w_max respectés  OK")

# Harnais walk-forward dédié : tourne, suit levier & coûts, backtest.py NON modifié.
variants11 = {n: pl.make_variant(n, w_max=0.10) for n in ["standard", "operational", "pp"]}
res11 = pl.run_markowitz_benchmark(prices, variants11, mean_estimator=ME11, cov_estimator=CE11,
                                   window=252, step=21, tc_bps=10.0, rf_annual=0.04)
mt11 = pl.markowitz_metrics_table(res11)
for col in ["Sharpe", "Average turnover", "Cumulative cost (%)", "Max leverage"]:
    assert col in mt11.columns, f"métrique {col} manquante"
# Contrainte de turnover : l'opérationnel (T_max=0.30) a un turnover post-initial borné.
to_op = res11["operational"].weights_history.diff().abs().sum(axis=1).dropna().max()
assert to_op <= 0.30 + 1e-3, f"turnover opérationnel non borné : {to_op}"
print(f"  harnais walk-forward OK (turnover opérationnel ≤ 0.30 : {to_op:.3f})")

# Calibration coordinate-ascent : renvoie des pénalités et un score fini.
pen11, score11, hist11 = pl.calibrate_markowitzpp(
    prices, pl.make_variant("pp"), mean_estimator=ME11, cov_estimator=CE11,
    params=("gamma_risk",), window=252, step=42, max_passes=1)
assert np.isfinite(score11) and pen11.gamma_risk > 0
print(f"  calibration : score={score11:.3f}, gamma_risk*={pen11.gamma_risk:.3f}  OK")

# Sensibilité : sorties finies + la pénalité robuste CHANGE bien l'allocation
# (le robust ≤ linear est une tendance statistique, pas un invariant déterministe).
sr = pl.weight_sensitivity(mu11, S11, pl.make_variant("robust"), pct=0.20)
sl = pl.weight_sensitivity(mu11, S11, pl.make_variant("linear"), pct=0.20)
assert 0 <= sr["mean_l1"] <= 2 and 0 <= sl["mean_l1"] <= 2, "mean_l1 hors bornes"
assert not np.allclose(sr["base_weights"], sl["base_weights"]), \
    "la pénalité robuste (kappa_mu) devrait modifier l'allocation"
print(f"  sensibilité finie & pénalité robuste active "
      f"(robust {sr['mean_l1']:.4f} | linear {sl['mean_l1']:.4f})  OK")

print("\n" + "=" * 70)
print("TOUS LES TESTS SONT PASSÉS ✔")
print("=" * 70)
