"""Estimateur de μ basé sur ml_features.py (Ridge + Random Forest, moyenne pondérée).

Branche ml_features.py (racine, anti-leakage déjà garanti par compute_features/
build_ml_dataset) sur l'interface MeanEstimator de estimators.py. Le module racine
ml_features.py n'est pas modifié.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import ml_features

from .estimators import MeanEstimator


class MLMeanEstimator(MeanEstimator):
    """μ̂ prédit par un ensemble Ridge + Random Forest sur les features de ml_features.py.

    `prices` (historique complet) et `sector_map` sont passés au constructeur (comme
    BlackLittermanMean reçoit mcaps/views) car l'interface estimate(returns) ne reçoit que
    les rendements. `estimate()` ne regarde jamais au-delà de la fenêtre `returns` reçue :
    même garantie anti-look-ahead que les autres MeanEstimator (cf. _iter_windows dans
    cov_eval.py / backtest.py, qui ne passent que train_returns = iloc[start-window:start]).

    estimation_window=200 (et non 126 envisagé initialement) : dans build_ml_dataset, la
    fenêtre interne passée à compute_features() a une taille CONSTANTE (estimation_window+1
    lignes) à chaque itération — une feature dont le calcul nécessite plus d'historique que
    cette taille est donc NaN pour 100 % des lignes d'entraînement, pas seulement les
    premières. Avec estimation_window=126, mom_6m (149j requis) et ma_dist_200 (201j requis)
    seraient NaN→0 à l'entraînement alors qu'elles seraient valides (non nulles) lors de la
    prédiction finale, qui utilise la fenêtre externe complète (252j dans return.ipynb) :
    le modèle apprendrait un poids ≈0 sur ces colonnes puis recevrait soudain des valeurs
    réelles en inférence — feature inerte, incohérence train/inférence. estimation_window=200
    élimine ce décalage pour ma_dist_200 (201j ≤ 200+1) ; mom_12m (252j requis) reste NaN dans
    les deux cas car aucune fenêtre ≤ 252j (la fenêtre externe maximale de ce notebook) ne
    peut le satisfaire — limite structurelle de ml_features.py vis-à-vis de ce notebook, pas
    un artefact de ce choix de paramètre.
    """

    name = "ml_estimator"

    def __init__(self, prices: pd.DataFrame, sector_map: pd.Series | None = None,
                 horizon: int = 21, estimation_window: int = 200,
                 # train_stride=5 retenu plutôt que 10/15 : sur l'univers de return.ipynb
                 # (~82 titres, fenêtre externe 252j), stride=5 produit 574 paires (X,y)
                 # d'entraînement, stride=10 en produit 328 et stride=15 seulement 246. Sous
                 # ~300-400 paires, on ne peut plus distinguer « le ML n'apporte rien sur ces
                 # données » de « le modèle est sous-entraîné » — ce qui invaliderait la
                 # comparaison avec les autres estimateurs de μ. Coût : stride=5 ajoute
                 # ~27 min à return.ipynb (vs ~16 min pour stride=10) ; ce coût est accepté
                 # car l'objectif du notebook est une comparaison honnête entre estimateurs,
                 # pas la minimisation du temps d'exécution.
                 train_stride: int = 5,
                 rf_n_estimators: int = 50, ridge_alpha: float = 1.0,
                 ridge_weight: float = 0.5, random_state: int = 42):
        self.prices = prices
        self.sector_map = sector_map
        self.horizon = horizon
        self.estimation_window = estimation_window
        self.train_stride = train_stride
        self.rf_n_estimators = rf_n_estimators
        self.ridge_alpha = ridge_alpha
        self.ridge_weight = ridge_weight
        self.random_state = random_state
        self.fallback_log: list[tuple[int, bool]] = []  # (len(returns), a_basculé_sur_SampleMean)

    def estimate(self, returns: pd.DataFrame) -> np.ndarray:
        cols = returns.columns
        px = self.prices.reindex(returns.index)[cols]
        n = len(returns)

        # Repli sur SampleMean (plutôt que laisser l'exception remonter) si la fenêtre
        # `returns` reçue est trop courte pour produire au moins une paire (features, label) :
        # return.ipynb appelle ce même estimateur avec window=126 dans le test de sensibilité
        # (WINDOWS=[126, 252, 300]), où 126 < estimation_window(200)+horizon(21)=221. Une
        # exception non gérée y interromprait tout le run au lieu d'un seul point de mesure.
        # fallback_log trace chaque occurrence pour que le notebook puisse l'afficher
        # explicitement (étoile + annotation sur le graphe) plutôt que de la laisser invisible
        # dans un résultat qui ressemblerait, à tort, à une mesure de performance du modèle ML.
        if n < self.estimation_window + self.horizon + 1:
            self.fallback_log.append((n, True))
            return returns.mean().values
        self.fallback_log.append((n, False))

        # Boucle walk-forward maison (même logique que ml_features.build_ml_dataset :
        # features sur px_window/ret_window se terminant à t_idx, label = rendement
        # px[t_idx -> t_idx+horizon], jamais au-delà de la fenêtre `returns` reçue) mais
        # avec un pas `train_stride` sur t_idx — build_ml_dataset calcule compute_features
        # pour CHAQUE date avant de pouvoir sous-échantillonner, ce qui annulerait le
        # bénéfice de train_stride sur le coût de calcul.
        X_rows, y_rows = [], []
        for t_idx in range(self.estimation_window, n - self.horizon, self.train_stride):
            px_window = px.iloc[t_idx - self.estimation_window: t_idx + 1]
            ret_window = returns.iloc[t_idx - self.estimation_window: t_idx]
            try:
                feats = ml_features.compute_features(px_window, ret_window, sector_map=self.sector_map)
            except Exception:
                continue
            fwd_ret = (px.iloc[t_idx + self.horizon] / px.iloc[t_idx] - 1.0).reindex(feats.index)
            valid = fwd_ret.notna()
            if valid.any():
                X_rows.append(feats[valid])
                y_rows.append(fwd_ret[valid])

        if not X_rows:
            return returns.mean().values
        X, y = pd.concat(X_rows, axis=0), pd.concat(y_rows, axis=0)

        ridge = Ridge(alpha=self.ridge_alpha).fit(X.values, y.values)
        rf = RandomForestRegressor(n_estimators=self.rf_n_estimators,
                                    random_state=self.random_state, n_jobs=-1).fit(X.values, y.values)

        feats_last = ml_features.compute_features(px, returns, sector_map=self.sector_map)
        pred = (self.ridge_weight * ridge.predict(feats_last.values)
                + (1 - self.ridge_weight) * rf.predict(feats_last.values))
        mu = pd.Series(pred / self.horizon, index=feats_last.index).reindex(cols)
        mu = mu.fillna(returns.mean())
        return mu.values


__all__ = ["MLMeanEstimator"]
