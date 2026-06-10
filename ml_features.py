"""
ml_features.py — Feature Engineering pour ML4PM
=================================================
Module standalone de construction de features cross-sectionnelles.
Compatible avec l'interface de backtest.py : toutes les features
calculées sur la fenêtre [start - window : start] uniquement.

Règles anti-leakage garanties :
  - Chaque feature à la date t utilise uniquement des données ≤ t-1
    (shift(1) systématique sur les prix).
  - Les z-scores cross-sectionnels sont calculés date par date,
    sans information future.
  - Les scalers ne sont jamais fittés sur des données hors fenêtre.

Usage principal :
    features_df = compute_features(prices, returns, sector_map)
    # → DataFrame [dates × tickers, n_features] sur la fenêtre fournie

Usage dans MLMeanEstimator (ml_model.py) :
    X_last = compute_features(prices, returns, sector_map).iloc[-1]
    # → vecteur de features sur le dernier état disponible (sans leakage)

Intégration dans le framework via MLMeanEstimator (ml_model.py) :
    MLMeanEstimator hérite de MeanEstimator et appelle compute_features()
    à l'intérieur d'estimate(train_returns). Aucune modification du notebook
    ni de backtest.py n'est nécessaire.

    Branche 1 : Features → (PCA optionnel) → Ridge
    Branche 2 : Features originales → Random Forest
    Ensemble  : moyenne pondérée Ridge + RF
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

TRADING_DAYS_YEAR: int = 252

# Fenêtres par défaut (en jours de trading)
MOMENTUM_WINDOWS: dict[str, int] = {
    "mom_1m":  21,
    "mom_3m":  63,
    "mom_6m":  126,
    "mom_12m": 252,
}
SKIP_DAYS: int = 21          # skip-1-month pour le momentum (évite le mean reversion court terme)
VOL_WINDOWS: dict[str, int] = {"vol_21d": 21, "vol_63d": 63}
MA_WINDOWS:  dict[str, int] = {"ma_50":  50,  "ma_200": 200}
RSI_PERIOD:  int = 14
HIGH_52W:    int = 252


# ---------------------------------------------------------------------------
# Fonctions primitives (une seule feature, sur toute la fenêtre de prix)
# ---------------------------------------------------------------------------

def _momentum(prices: pd.DataFrame, window: int, skip: int = SKIP_DAYS) -> pd.Series:
    """
    Rendement sur `window` jours avec skip de `skip` jours (skip-1-month).

    Timeline à la dernière date t de la fenêtre :
        retourne prices[t - skip] / prices[t - skip - window] - 1
    Utilise uniquement des prix ≤ t-1 après décalage dans compute_features.

    Le skip évite le biais de mean-reversion à très court terme documenté
    dans Jegadeesh & Titman (1993) : on exclut le dernier mois du calcul.
    """
    # On décale d'abord d'1 jour pour ne jamais utiliser le prix du jour t
    px = prices.shift(1)
    end   = px.shift(skip)          # prix à t-1-skip
    start = px.shift(skip + window) # prix à t-1-skip-window
    mom = (end / start - 1.0).iloc[-1]
    return mom


def _realized_vol(returns: pd.DataFrame, window: int) -> pd.Series:
    """
    Volatilité réalisée annualisée sur les `window` derniers jours.

    Utilise returns.shift(1) pour ne jamais inclure le rendement du jour t.
    """
    r = returns.shift(1).iloc[-window:]
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR)


def _downside_vol(returns: pd.DataFrame, window: int = 63) -> pd.Series:
    """
    Semi-déviation (volatilité des rendements négatifs uniquement), annualisée.
    Proxy du risque de perte réel — utilisé dans le ratio de Sortino.
    """
    r = returns.shift(1).iloc[-window:]
    neg = r.clip(upper=0.0)
    return neg.std(ddof=1) * np.sqrt(TRADING_DAYS_YEAR)


def _distance_to_ma(prices: pd.DataFrame, window: int) -> pd.Series:
    """
    Distance au-dessus de la moyenne mobile : (P_t-1 - MA_window) / MA_window.

    Mesure d'extension / retour vers la moyenne.
    P_t-1 = dernier prix connu (shift(1) déjà appliqué en amont).
    """
    px = prices.shift(1)
    ma = px.rolling(window, min_periods=window // 2).mean()
    last_px = px.iloc[-1]
    last_ma = ma.iloc[-1]
    return (last_px - last_ma) / last_ma.replace(0, np.nan)


def _distance_to_52w_high(prices: pd.DataFrame) -> pd.Series:
    """
    Distance au plus haut 52 semaines : (P_t-1 - P_52w_high) / P_52w_high.

    George & Hwang (2004) : l'un des meilleurs prédicteurs de rendements futurs
    parmi les features techniques, même supérieur au momentum standard.
    Négatif = en dessous du plus haut (drawdown depuis le sommet).
    """
    px = prices.shift(1)
    high = px.rolling(HIGH_52W, min_periods=HIGH_52W // 2).max()
    last_px = px.iloc[-1]
    last_high = high.iloc[-1]
    return (last_px - last_high) / last_high.replace(0, np.nan)


def _rolling_drawdown(prices: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Drawdown glissant : (P_t-1 - max(P sur `window` jours)) / max.

    Signal ambigu : mean reversion ou continuation (momentum). À utiliser
    prudemment — principalement comme feature de risque.
    """
    px = prices.shift(1)
    roll_max = px.rolling(window, min_periods=window // 4).max()
    last_px = px.iloc[-1]
    last_max = roll_max.iloc[-1]
    return (last_px - last_max) / last_max.replace(0, np.nan)


def _rsi(prices: pd.DataFrame, period: int = RSI_PERIOD) -> pd.Series:
    """
    Relative Strength Index sur `period` jours.

    Calcul standard Wilder : moyenne des gains / (moyenne des gains + moyenne des pertes).
    Normalisé [0, 100]. Shift(1) sur les prix pour garantir le non-leakage.
    En usage cross-sectionnel : feature de momentum normalisée,
    plus utile après z-score que comme valeur brute.
    """
    px = prices.shift(1)
    delta = px.diff()
    gain = delta.clip(lower=0.0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0.0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.iloc[-1]


def _relative_volume(prices: pd.DataFrame, volume: Optional[pd.DataFrame] = None,
                     window: int = 21) -> Optional[pd.Series]:
    """
    Volume relatif : volume_t-1 / moyenne(volume sur `window` jours).

    Retourne None si les données de volume ne sont pas disponibles.
    Disponible via yfinance avec auto_adjust=True (colonne 'Volume').
    En interaction avec le momentum : les ruptures sur fort volume sont plus persistantes.
    """
    if volume is None:
        return None
    vol = volume.shift(1)
    avg_vol = vol.rolling(window, min_periods=window // 2).mean()
    last_vol = vol.iloc[-1]
    last_avg = avg_vol.iloc[-1]
    return (last_vol / last_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Normalisation cross-sectionnelle
# ---------------------------------------------------------------------------

def cross_section_zscore(series: pd.Series) -> pd.Series:
    """
    Z-score cross-sectionnel : (x_i - mean(x)) / std(x) sur l'univers à une date donnée.

    Opération centrale du feature engineering :
    - Rend les features comparables entre actifs (supprime le niveau absolu)
    - Robuste aux régimes de marché (normalisé par la dispersion du jour)
    - Directement exploitable par un modèle cross-sectionnel

    Gestion des outliers : winsorisation à ±3 sigma après normalisation.
    """
    mu  = series.mean()
    std = series.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    z = (series - mu) / std
    return z.clip(-3.0, 3.0)


def sector_zscore(series: pd.Series, sector_map: pd.Series) -> pd.Series:
    """
    Z-score intra-sectoriel : normalise chaque feature au sein du même secteur GICS.

    Raffinement significatif par rapport au z-score global :
    un titre tech à fort momentum est comparé aux autres titres tech, pas aux utilities.
    Neutralise les biais sectoriels — standard dans les modèles de factor investing professionnels.

    `sector_map` : pd.Series indexée par ticker, valeurs = secteur GICS
                   (issue de get_sp500_constituents() dans markowitz_sp500_V2.ipynb)

    Les tickers sans secteur connu reçoivent un z-score global.
    """
    result = pd.Series(np.nan, index=series.index)
    common = series.index.intersection(sector_map.index)
    sectors = sector_map.loc[common].unique()

    for sector in sectors:
        tickers_in_sector = sector_map.index[sector_map == sector]
        sub = series.loc[series.index.intersection(tickers_in_sector)].dropna()
        if len(sub) < 2:
            # Secteur trop petit : fallback z-score global
            result.loc[sub.index] = cross_section_zscore(sub)
            continue
        mu  = sub.mean()
        std = sub.std(ddof=1)
        if std == 0 or np.isnan(std):
            result.loc[sub.index] = 0.0
        else:
            z = (sub - mu) / std
            result.loc[sub.index] = z.clip(-3.0, 3.0)

    # Tickers sans secteur connu → z-score global
    no_sector = series.index.difference(common)
    if len(no_sector) > 0:
        result.loc[no_sector] = cross_section_zscore(series.loc[no_sector])

    return result


def rank_normalize(series: pd.Series) -> pd.Series:
    """
    Normalisation par rang [0, 1] — alternative aux z-scores, plus robuste aux outliers.

    Recommandée pour les features asymétriques comme le momentum.
    rank / (n - 1) → [0, 1] avec 0 = plus faible, 1 = plus fort.
    """
    r = series.rank(method="average", na_option="keep")
    n_valid = r.notna().sum()
    if n_valid <= 1:
        return pd.Series(0.5, index=series.index)
    return (r - 1) / (n_valid - 1)


# ---------------------------------------------------------------------------
# Fonction principale : compute_features
# ---------------------------------------------------------------------------

def compute_features(
    prices:     pd.DataFrame,
    returns:    pd.DataFrame,
    sector_map: Optional[pd.Series] = None,
    volume:     Optional[pd.DataFrame] = None,
    normalize:  str = "sector_zscore",
) -> pd.DataFrame:
    """
    Calcule le feature set complet pour une fenêtre de données.

    Garantie anti-leakage : toutes les features utilisent shift(1) sur les prix/rendements,
    donc à la dernière date t de la fenêtre, aucune feature n'utilise de données > t-1.
    Cette fonction est conçue pour être appelée à l'intérieur de optimize_weights_fn,
    uniquement sur train_returns (fenêtre [start - window : start] de backtest.py).

    Paramètres
    ----------
    prices : pd.DataFrame [T × N]
        Prix de clôture ajustés (dividendes/splits). Issu de download_prices() et clean_prices().
        Doit inclure suffisamment d'historique : au moins HIGH_52W + 1 jours (252 + 1 = 253 min).

    returns : pd.DataFrame [T × N]
        Rendements simples (pct_change). Issu de compute_returns(prices, mode='simple').
        Doit être aligné avec prices sur les mêmes colonnes.

    sector_map : pd.Series, optional
        Index = tickers, valeurs = secteur GICS.
        Issu de get_sp500_constituents()['sector'].
        Si None : z-score global (cross_section_zscore) utilisé pour toutes les features.

    volume : pd.DataFrame, optional
        Volume de trading quotidien [T × N]. Issu de yfinance avec auto_adjust=True.
        Si None : features de volume ignorées.

    normalize : str
        Mode de normalisation cross-sectionnelle :
        - 'sector_zscore'  : z-score intra-sectoriel (recommandé, Architecture 2)
        - 'global_zscore'  : z-score global sur tout l'univers
        - 'rank'           : normalisation par rang [0, 1]

    Retourne
    --------
    pd.DataFrame [N_actifs × N_features]
        Un vecteur de features par actif, indexé par ticker.
        Toutes les features sont normalisées cross-sectionnellement.
        Valeurs manquantes remplies par 0 (neutre après normalisation).

    Features incluses (15–17 selon disponibilité du volume)
    --------------------------------------------------------
    Momentum (4) :
        mom_1m, mom_3m, mom_6m, mom_12m
        Tous avec skip-1-month (SKIP_DAYS = 21) sauf mom_1m (= skip only).
        Justification : Jegadeesh & Titman (1993), facteur le plus répliqué.

    Volatilité (3) :
        vol_21d, vol_63d : volatilité réalisée annualisée
        vol_diff         : vol_21d - vol_63d (compression/expansion de volatilité)

    Trend (2) :
        ma_dist_50  : distance à la MA50
        ma_dist_200 : distance à la MA200
        Justification : "golden cross" / "death cross" comme indicateurs de régime.

    Drawdown (1) :
        dist_52w_high : distance au plus haut 52 semaines
        Justification : George & Hwang (2004).

    Oscillateurs (1) :
        rsi_14 : RSI 14 jours (valeur brute avant normalisation cross-sectionnelle)

    Rolling drawdown (1) :
        rolling_dd : drawdown glissant 252 jours

    Volume (2, si disponible) :
        rel_volume     : volume relatif vs moyenne 21 jours
        rel_volume_zscore : z-score du volume relatif (spike detector)
    """
    # --- Alignement des colonnes ---
    common_cols = prices.columns.intersection(returns.columns)
    prices  = prices[common_cols]
    returns = returns[common_cols]
    if volume is not None:
        common_vol = common_cols.intersection(volume.columns)
        volume = volume[common_vol]

    tickers = list(common_cols)
    features: dict[str, pd.Series] = {}

    # --- 1. Momentum (avec skip-1-month) ---
    # mom_1m : rendement sur 21j, skip=21j (donc on saute le dernier mois)
    # mom_3m, 6m, 12m : fenêtres plus larges, même skip
    for name, window in MOMENTUM_WINDOWS.items():
        if len(prices) >= window + SKIP_DAYS + 2:
            features[name] = _momentum(prices, window=window, skip=SKIP_DAYS)
        else:
            features[name] = pd.Series(np.nan, index=tickers)

    # --- 2. Volatilité réalisée ---
    for name, window in VOL_WINDOWS.items():
        if len(returns) >= window + 1:
            features[name] = _realized_vol(returns, window=window)
        else:
            features[name] = pd.Series(np.nan, index=tickers)

    # Différentiel de volatilité : signal de compression/expansion
    if "vol_21d" in features and "vol_63d" in features:
        features["vol_diff"] = features["vol_21d"] - features["vol_63d"]

    # --- 3. Trend : distances aux moyennes mobiles ---
    for name, window in MA_WINDOWS.items():
        ma_feat_name = f"ma_dist_{window}"
        if len(prices) >= window + 1:
            features[ma_feat_name] = _distance_to_ma(prices, window=window)
        else:
            features[ma_feat_name] = pd.Series(np.nan, index=tickers)

    # --- 4. Distance au plus haut 52 semaines ---
    if len(prices) >= HIGH_52W // 2:
        features["dist_52w_high"] = _distance_to_52w_high(prices)
    else:
        features["dist_52w_high"] = pd.Series(np.nan, index=tickers)

    # --- 5. RSI 14 jours ---
    if len(prices) >= RSI_PERIOD + 2:
        features["rsi_14"] = _rsi(prices, period=RSI_PERIOD)
    else:
        features["rsi_14"] = pd.Series(np.nan, index=tickers)

    # --- 6. Rolling drawdown ---
    dd_window = min(252, len(prices) - 1)
    if dd_window >= 20:
        features["rolling_dd"] = _rolling_drawdown(prices, window=dd_window)
    else:
        features["rolling_dd"] = pd.Series(np.nan, index=tickers)

    # --- 7. Volume (optionnel) ---
    if volume is not None and len(volume) >= 22:
        rv = _relative_volume(volume, window=21)
        if rv is not None:
            rv = rv.reindex(tickers)
            features["rel_volume"] = rv
            # Z-score du volume relatif : détecteur de spike
            rv_filled = rv.fillna(rv.median())
            mu_rv  = rv_filled.mean()
            std_rv = rv_filled.std(ddof=1)
            if std_rv > 0:
                features["rel_volume_zscore"] = (rv_filled - mu_rv) / std_rv
            else:
                features["rel_volume_zscore"] = pd.Series(0.0, index=tickers)

    # --- Assemblage en DataFrame ---
    feat_df = pd.DataFrame(features, index=tickers)

    # --- Normalisation cross-sectionnelle ---
    # Choix du normalisateur selon le paramètre `normalize`
    if normalize == "sector_zscore" and sector_map is not None:
        norm_fn = lambda s: sector_zscore(s, sector_map)
    elif normalize == "rank":
        norm_fn = rank_normalize
    else:
        norm_fn = cross_section_zscore

    normalized = feat_df.apply(norm_fn, axis=0)

    # Remplissage des NaN résiduels par 0 (signal neutre après normalisation)
    normalized = normalized.fillna(0.0)

    return normalized


# ---------------------------------------------------------------------------
# Fonction utilitaire : build_ml_dataset
# ---------------------------------------------------------------------------

def build_ml_dataset(
    prices:     pd.DataFrame,
    returns:    pd.DataFrame,
    sector_map: Optional[pd.Series] = None,
    volume:     Optional[pd.DataFrame] = None,
    horizon:    int = 21,
    estimation_window: int = 252,
    normalize:  str = "sector_zscore",
    target_mode: str = "simple",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construit le dataset (X, y) complet pour l'entraînement ML hors walk-forward.

    Utilisé dans ml_backtest_nb.ipynb pour l'analyse IC par feature avant backtest.

    Paramètres
    ----------
    prices, returns, sector_map, volume, normalize : cf. compute_features.

    horizon : int
        Horizon de prédiction en jours (21 = mensuel, par défaut).

    estimation_window : int
        Nombre de jours minimum d'historique pour calculer les features.

    target_mode : str
        'simple' : rendement futur brut sur `horizon` jours.
        'zscore' : rendement futur cross-sectionnellement normalisé (z-score).

    Retourne
    --------
    X : pd.DataFrame [N_observations × N_features]
        Features pour chaque (date, ticker), index = MultiIndex(date, ticker).

    y : pd.Series [N_observations]
        Labels futurs (rendement sur horizon jours), index aligné avec X.
        y_{i,t} = prix[t + horizon, i] / prix[t, i] - 1

    Note sur le leakage
    -------------------
    Les labels y_{i,t} utilisent les prix en t+horizon.
    Les features à t utilisent uniquement des prix ≤ t-1.
    Aucune fenêtre ne se chevauche : garantie anti-leakage par construction.
    """
    all_dates = returns.index
    if len(all_dates) < estimation_window + horizon:
        raise ValueError(
            f"Pas assez d'observations ({len(all_dates)}) pour "
            f"estimation_window={estimation_window} + horizon={horizon}."
        )

    X_rows, y_rows, index_rows = [], [], []

    for t_idx in range(estimation_window, len(all_dates) - horizon):
        t_date = all_dates[t_idx]

        # Fenêtre d'estimation : [t - window, t[ — strictement passé
        px_window  = prices.iloc[t_idx - estimation_window : t_idx + 1]
        ret_window = returns.iloc[t_idx - estimation_window : t_idx]
        if volume is not None:
            vol_window = volume.iloc[t_idx - estimation_window : t_idx + 1]
        else:
            vol_window = None

        # Features sur la fenêtre (garantie : shift(1) interne, pas de t+h)
        try:
            feats = compute_features(
                px_window, ret_window,
                sector_map=sector_map,
                volume=vol_window,
                normalize=normalize,
            )
        except Exception:
            continue

        # Labels : rendement de t à t+horizon
        t_plus_h = t_idx + horizon
        px_t      = prices.iloc[t_idx]
        px_future = prices.iloc[t_plus_h]
        fwd_ret = (px_future / px_t - 1.0).reindex(feats.index)

        if target_mode == "zscore":
            fwd_ret = cross_section_zscore(fwd_ret.dropna()).reindex(feats.index).fillna(0.0)

        # Empilement
        for ticker in feats.index:
            if ticker not in fwd_ret.index or np.isnan(fwd_ret.loc[ticker]):
                continue
            X_rows.append(feats.loc[ticker].values)
            y_rows.append(fwd_ret.loc[ticker])
            index_rows.append((t_date, ticker))

    if not X_rows:
        raise ValueError("Dataset vide — vérifiez les données et les paramètres.")

    mi = pd.MultiIndex.from_tuples(index_rows, names=["date", "ticker"])
    X  = pd.DataFrame(X_rows, index=mi, columns=feats.columns)
    y  = pd.Series(y_rows,   index=mi, name=f"fwd_ret_{horizon}d")
    return X, y


# ---------------------------------------------------------------------------
# Fonctions de sanité / tests
# ---------------------------------------------------------------------------

def check_no_leakage(
    prices:  pd.DataFrame,
    returns: pd.DataFrame,
    sector_map: Optional[pd.Series] = None,
    tol: float = 1e-8,
) -> bool:
    """
    Test de sanité : vérifie que les features calculées à la date t
    ne dépendent PAS du prix du jour t lui-même.

    Principe (correct) : calculer les features sur une fenêtre [0:T],
    puis recalculer sur la même fenêtre après avoir remplacé le dernier
    prix (= prix du jour t, qui NE DOIT PAS être utilisé par shift(1))
    par une valeur absurde (999). Si les features sont identiques →
    shift(1) fonctionne, pas de look-ahead bias.

    Ce test est plus rigoureux que de comparer deux fenêtres de longueur
    différente (ce qui comparerait deux dates différentes, pas le même t).

    Retourne True si le test passe, lève une AssertionError sinon.
    """
    if len(prices) < 60:
        print("[SKIP] check_no_leakage : fenêtre trop courte pour le test.")
        return True

    window = min(60, len(prices) - 2)
    px_base  = prices.iloc[:window].copy()
    ret_base = returns.iloc[:window - 1].copy()

    # Features sur la fenêtre de base
    f_base = compute_features(px_base, ret_base, sector_map=sector_map)

    # Modifier le DERNIER prix (= jour t) : s'il était utilisé → différence détectable
    px_modified = px_base.copy()
    px_modified.iloc[-1] = 999.0           # valeur absurde pour maximiser la diff
    ret_modified = px_modified.pct_change().dropna()
    f_modified = compute_features(px_modified, ret_modified, sector_map=sector_map)

    # Les features doivent être identiques (shift(1) les protège du dernier prix)
    common = f_base.index.intersection(f_modified.index)
    diff = (f_base.loc[common] - f_modified.loc[common]).abs().max().max()

    if diff > tol:
        raise AssertionError(
            f"LEAKAGE DÉTECTÉ : modification du prix du jour t change les features "
            f"de {diff:.6f} (tolérance={tol}). "
            f"Vérifier que shift(1) est bien appliqué dans toutes les primitives."
        )
    print(f"[OK] check_no_leakage : diff max = {diff:.2e} ≤ {tol} — "
          f"les features ne dépendent pas du prix du jour t.")
    return True


def check_zscore_properties(features_df: pd.DataFrame, tol: float = 0.15) -> bool:
    """
    Vérifie que les z-scores cross-sectionnels ont bien :
    - une moyenne ≈ 0 (|mean| < tol)
    - un écart-type ≈ 1 (|std - 1| < tol)

    Tolérance assouplie à 0.15 car le winsorising à ±3σ et les NaN → 0
    peuvent décaler légèrement les moments.
    """
    passed = True
    for col in features_df.columns:
        s = features_df[col].replace(0.0, np.nan).dropna()
        if len(s) < 5:
            continue
        m  = s.mean()
        sd = s.std(ddof=1)
        if abs(m) > tol:
            print(f"[WARN] {col} : mean = {m:.4f} (attendu ≈ 0, tol={tol})")
            passed = False
        if abs(sd - 1.0) > tol + 0.15:   # std peut s'éloigner après winsorising
            print(f"[WARN] {col} : std = {sd:.4f} (attendu ≈ 1)")
            passed = False
    if passed:
        print("[OK] check_zscore_properties : tous les z-scores bien centrés/réduits.")
    return passed


def feature_ic_analysis(
    prices:     pd.DataFrame,
    returns:    pd.DataFrame,
    sector_map: Optional[pd.Series] = None,
    horizon:    int = 21,
    estimation_window: int = 252,
) -> pd.DataFrame:
    """
    Calcule le Spearman IC moyen par feature sur l'ensemble des données fournies.

    Usage : analyse exploratoire dans ml_backtest_nb.ipynb (Section 3).
    Si aucune feature n'a IC > 2%, le modèle ML n'aura aucun signal à exploiter.

    Retourne un DataFrame trié par |IC| décroissant :
        IC_mean : IC Spearman moyen dans le temps
        IC_std  : écart-type de l'IC (instabilité du signal)
        ICIR    : IC_mean / IC_std (ratio signal/bruit, analogue du Sharpe)
        t_stat  : IC_mean / (IC_std / sqrt(T)) (significativité statistique)
    """
    from scipy.stats import spearmanr

    X, y = build_ml_dataset(
        prices, returns,
        sector_map=sector_map,
        horizon=horizon,
        estimation_window=estimation_window,
        target_mode="simple",
    )

    dates = X.index.get_level_values("date").unique()
    ic_per_date: dict[str, list[float]] = {col: [] for col in X.columns}

    for date in dates:
        X_t = X.loc[date]
        y_t = y.loc[date]
        if len(y_t) < 10:
            continue
        for col in X.columns:
            valid = X_t[col].notna() & y_t.notna()
            if valid.sum() < 10:
                ic_per_date[col].append(np.nan)
                continue
            ic, _ = spearmanr(X_t.loc[valid, col], y_t.loc[valid])
            ic_per_date[col].append(ic if np.isfinite(ic) else np.nan)

    rows = []
    for col, ics in ic_per_date.items():
        ic_arr = np.array([v for v in ics if not np.isnan(v)])
        if len(ic_arr) == 0:
            continue
        ic_mean = ic_arr.mean()
        ic_std  = ic_arr.std(ddof=1) if len(ic_arr) > 1 else np.nan
        icir    = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
        t_stat  = ic_mean / (ic_std / np.sqrt(len(ic_arr))) if ic_std and ic_std > 0 else np.nan
        rows.append({
            "feature": col,
            "IC_mean":  round(ic_mean * 100, 3),  # en %
            "IC_std":   round(ic_std  * 100, 3) if not np.isnan(ic_std) else np.nan,
            "ICIR":     round(icir, 3)            if not np.isnan(icir) else np.nan,
            "t_stat":   round(t_stat, 2)          if not np.isnan(t_stat) else np.nan,
            "n_periods": len(ic_arr),
        })

    result = pd.DataFrame(rows).set_index("feature").sort_values("IC_mean", key=abs, ascending=False)
    return result


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Fonction principale
    "compute_features",
    # Construction dataset ML
    "build_ml_dataset",
    # Normalisation cross-sectionnelle
    "cross_section_zscore",
    "sector_zscore",
    "rank_normalize",
    # Tests de sanité
    "check_no_leakage",
    "check_zscore_properties",
    # Analyse exploratoire
    "feature_ic_analysis",
    # Constantes exposées
    "TRADING_DAYS_YEAR",
    "MOMENTUM_WINDOWS",
    "SKIP_DAYS",
    "VOL_WINDOWS",
    "MA_WINDOWS",
    "RSI_PERIOD",
    "HIGH_52W",
]
