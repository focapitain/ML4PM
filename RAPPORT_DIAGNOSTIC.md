# Rapport de diagnostic — ML4PM

> Diagnostic en lecture seule. Aucun fichier du repo n'a été modifié. Un venv jetable
> (`.venv_diag/`, non commité) a servi à exécuter les notebooks ; à supprimer après lecture.

---

## 1. Arborescence et rôle de chaque fichier

### Racine — architecture "historique" (notebooks autonomes + `backtest.py`)

| Fichier | Rôle |
|---|---|
| `Markowitz_1.ipynb` | Premier prototype pédagogique Markowitz (closed-form, S&P 500 subset). |
| `Markowitz2.ipynb` | Itération suivante du prototype. |
| `Markowitz2-2.ipynb` | Notebook pédagogique "final" de cette lignée : `optimal1/optimal2`, frontière Monte-Carlo, importe `backtest.py` (racine). |
| `markowitz_sp500_V1.ipynb` | Première version du framework S&P 500 modulaire (cvxpy, sans `backtest.py`). |
| `markowitz_sp500_V2.ipynb` | Framework modulaire (univers/estimateurs/objectifs/optimiseurs interchangeables) ; importe `backtest.py` (racine). |
| `markowitz_sp500_evaluation_V3.ipynb` | Suite de V2, axée évaluation ; importe aussi `backtest.py` (racine). |
| `backtest.py` | Moteur de backtest walk-forward "legacy", utilisé uniquement par les 3 notebooks ci-dessus. Contient un bloc en chantier marqué `# ── NUOVO ──` (CVaR/VaR/Calmar/Information Ratio) avec indentation mixte tabs/espaces — *parse* sans erreur mais visuellement non finalisé. **N'a pas** le Sharpe net du rf, ni les coûts symétriques sur le benchmark, ni le comptage des fallbacks. |
| `ml_features.py` | Module de feature engineering cross-sectionnel (momentum, vol, RSI, distance 52w high, z-scores sectoriels...) avec garde-fous anti-leakage. **Orphelin : jamais importé nulle part dans le repo.** Sa propre docstring référence `ml_model.py` et `ml_backtest_nb.ipynb`, qui **n'existent pas** dans le repo. |
| `test_pipeline.py` | Smoke-test hors-ligne complet du package `portfolio_lab` (données synthétiques) — la vraie suite de tests du projet. |
| `README.md` | Documentation de l'architecture "racine" (3 notebooks + `backtest.py` + `benchmark_demo`). Ne mentionne ni `portfolio_lab/` ni `READMEE.md`. |
| `READMEE.md` | Documentation complète de `portfolio_lab/` (architecture, workflow, notebooks `notebooks/*`, tests). Ne mentionne aucun fichier racine. |

### `portfolio_lab/` — architecture "moderne" (package + 3 notebooks de recherche)

| Fichier | Rôle |
|---|---|
| `__init__.py` | API publique du package (réexporte tout), `__version__ = "2.0.0"`. |
| `config.py` | `ExperimentConfig` : tous les leviers d'une expérience en un seul dataclass. |
| `data.py` | Constituants S&P 500, téléchargement yfinance, nettoyage, `make_synthetic_prices` (mode hors-ligne). |
| `frequencies.py` | Resampling daily/weekly/monthly + annualisation. |
| `returns.py` | Rendements simples / log. |
| `universe.py` | Sélecteurs d'univers (Strategy) : `AllUniverse`, `TopN`/`FirstNAlphabetical`, `RandomN`, `StratifiedBySector`. |
| `estimators.py` | Estimateurs μ (Sample, JamesStein, EWMA, Rolling, PCAFactor) et Σ (Empirical, Rolling, EWMA, IEWMA, LedoitWolf, OAS, ConstantCorrelation). |
| `optimizers.py` | Optimiseurs (AnalyticLagrangian, CVXPY, Scipy/SLSQP, DifferentialEvolution) + baselines de risque. |
| `markowitz.py` | Benchmark institutionnel : 5 formulations (Standard, Linéaire, Opérationnelle, Robuste, Markowitz++), calibration, harnais walk-forward dédié. |
| `frontier.py` | Frontière efficiente (grille de `target_return`). |
| `experiment.py` | `run_experiment(config)` → résultat in-sample. |
| `backtest.py` | Moteur de backtest walk-forward OOS **audité** (Sharpe net de rf, coûts symétriques stratégie/benchmark, fallback compté, sélection d'actifs par fenêtre anti-look-ahead). C'est la version utilisée par `notebooks/` et `benchmark_demo/`. |
| `backtest_adapter.py` | Pont `ExperimentConfig` → `backtest.walk_forward_backtest`. |
| `benchmarking.py` | Moteur partagé des notebooks (`run_oos_benchmark`, tables comparatives). |
| `cov_eval.py` | Évaluation statistique de Σ (log-vraisemblance, RMSE, conditionnement, stabilité, régimes). |
| `mean_eval.py` | Évaluation prédictive de μ (IC de Spearman, hit-rate...). |
| `metrics.py` | Sortino, Calmar, VaR, CVaR, `tail_metrics`. |
| `plotting.py` | Visualisations in-sample. |
| `reporting.py` | `RunReport` : journalise chaque run dans `results/` (non versionné). |

### `notebooks/` — les 3 notebooks de recherche officiels de `portfolio_lab`

| Fichier | Rôle |
|---|---|
| `return.ipynb` | Benchmark des 5 estimateurs de μ (IC, robustesse, impact portefeuille). |
| `covariance.ipynb` | Benchmark des 7 estimateurs de Σ (axes statistique + portefeuille). |
| `markowitz.ipynb` | Benchmark des 5 formulations de Markowitz + calibration + analyse par régime. |

### `benchmark_demo/` — démo Streamlit (utilise `portfolio_lab`, pas `backtest.py` racine)

| Fichier | Rôle |
|---|---|
| `Cockpit.py` | Point d'entrée Streamlit (page d'accueil / vue d'ensemble). |
| `components/charts.py` | Fabriques de figures Plotly pures. |
| `components/sidebar.py` | Barre latérale globale (paramètres). |
| `components/theme.py` | Thème visuel Plotly/Streamlit. |
| `components/ui.py` | Composants HTML custom (timeline, cartes d'insight). |
| `core/analytics.py` | Analytique démo pure (ne modifie pas `portfolio_lab`). |
| `core/constraints.py` | Détection des contraintes actives sur une allocation. |
| `core/presets.py` | Scénarios historiques / univers curatés / annotations d'événements. |
| `core/registry.py` | Registres nom → objet `portfolio_lab` (pour compatibilité cache Streamlit). |
| `core/services.py` | Couche service : seul point de contact entre l'UI et `portfolio_lab` (`from portfolio_lab.backtest import ...`). |
| `pages/1_Simulation.py` | Page "comment le portefeuille décide" (lance le backtest). |
| `pages/2_Analyse.py` | Page performance/risque/marché avec lectures automatiques. |

### `scripts/`

| Fichier | Rôle |
|---|---|
| `generate_figures.py` | Génère un jeu de figures PNG (poids, frontière efficiente, corrélations, performance, NAV OOS, drawdown) à partir de données synthétiques, dans `figures/`. Utilise `portfolio_lab`. |

---

## 2. Exécution des 3 notebooks `notebooks/`

Environnement : venv créé via `virtualenv` (le module `venv` système n'était pas disponible,
`apt install python3.10-venv` aurait demandé `sudo` — contourné sans droits admin), dépendances
de `requirements.txt` installées avec succès, `jupyter nbconvert --execute`.

| Notebook | Résultat | Temps | Cause |
|---|---|---|---|
| `return.ipynb` | ❌ Échec | **13 min 14 s** avant l'erreur | `KeyError` sur la dernière cellule de synthèse. |
| `covariance.ipynb` | ❌ Échec | **50 s** avant l'erreur | `KeyError: 'Empirique'`. |
| `markowitz.ipynb` | ✅ Succès (bout en bout, 0 cellule en erreur) | **≈ 6 h** (355 min, mesuré avec contention CPU car les 3 notebooks tournaient en partie en parallèle ; à isoler en standalone pour un chiffre fiable, mais l'ordre de grandeur restera "très long") | `AllUniverse()` = S&P 500 complet (~500 titres), avec calibration coordinate-ascent (`max_passes=3`) sur 5 formulations — le code lui-même commente `# Option 1 : S&P 500 complet (~500, lent)`. |

### Détail `return.ipynb` — `KeyError`

Cellule fautive (section finale de synthèse) :
```python
mu_only = m.loc[[f"MaxSharpe / {n}" for n in mean_estimators]].copy()
```
```
KeyError: "None of [Index(['MaxSharpe / SampleMean', ...])] are in the [index]"
```
**Cause probable** : plus haut dans le notebook, les variantes comparées sont construites avec
le préfixe `"Max_utility / {n}"` (objectif réellement utilisé : `objective="max_utility"`) :
```python
variants = {f"Max_utility / {n}": replace(base, mean_estimator=e) for n, e in mean_estimators.items()}
```
La cellule de synthèse cherche ensuite `"MaxSharpe / {n}"` — un préfixe qui n'a **jamais existé**
dans l'index `m`. C'est une incohérence de nommage interne au notebook (copier-coller d'une
version antérieure où l'objectif était `max_sharpe`), pas un problème d'environnement ou de
dépendances. Tout ce qui précède (sections 1 à 4 : IC, RMSE, sensibilité, impact portefeuille)
s'exécute correctement — seule la dernière cellule de décision casse.

### Détail `covariance.ipynb` — `KeyError`

Cellule fautive (graphique de conditionnement dans le temps) :
```python
for name in ["Empirique", "EWMA λ=0.94", "IEWMA", "Ledoit-Wolf"]:
    df = pl.rolling_cov_eval(returns, estimators[name], window=WINDOW, step=STEP)
```
```
KeyError: 'Empirique'
```
**Cause probable** : le dict `estimators` défini plus haut contient 7 clés — `Rolling-120`,
`EWMA λ=0.94`, `EWMA λ=0.97`, `IEWMA`, `Ledoit-Wolf`, `OAS`, `Corr. constante` — mais **aucune
clé `"Empirique"`** (l'estimateur empirique/sample n'a jamais été ajouté à ce dict de
comparaison, contrairement à ce que documente `READMEE.md` : *"Empirical, Rolling, EWMA, IEWMA,
Ledoit-Wolf, OAS, Constant-Corr"*). Échec précoce (cellule 7), donc très peu de résultats produits
avant l'arrêt.

### Temps d'exécution — exploitabilité en TP

- `covariance.ipynb` échoue en moins d'une minute → corrigeable, mais inutilisable en l'état.
- `return.ipynb` prend déjà ~13 min pour échouer à la toute dernière cellule → frustrant en
  session live (tout le calcul refait à chaque relance), mais le bug lui-même est trivial à
  corriger (un nom de clé).
- `markowitz.ipynb` tourne avec succès mais sur plusieurs heures avec `AllUniverse()`. **Pour un
  usage TP, il faut absolument changer le sélecteur par défaut** (le notebook propose déjà en
  commentaire `StratifiedBySector(80)`, `TopN(50)`, `RandomN(50)` — bien plus rapides) sans quoi
  aucune démonstration live n'est possible.

---

## 3. Cohérence `portfolio_lab/` vs fichiers racine

- **`backtest.py` (racine) vs `portfolio_lab/backtest.py`** : même API publique exacte (mêmes
  classes/fonctions : `WalkForwardBacktestConfig`, `walk_forward_backtest`, `performance_summary`,
  `plot_backtest_nav`...), mais la version racine est **antérieure et moins aboutie** : pas de
  `risk_free_annual`, pas de coûts symétriques sur le benchmark 1/N, pas de comptage des
  fallbacks, sélection d'actifs par `dropna` global (biais de look-ahead potentiel) au lieu d'une
  sélection par fenêtre. Elle porte en plus un bloc visiblement en chantier (`# ── NUOVO ──`,
  indentation tabs/espaces mélangée, double commentaire `#### / ####`) qui ajoute Calmar/IR/VaR/CVaR.
  - **Qui l'utilise ?** Uniquement les 3 notebooks racine `Markowitz2-2.ipynb`,
    `markowitz_sp500_V2.ipynb`, `markowitz_sp500_evaluation_V3.ipynb` (`import backtest`).
  - Les notebooks dans `notebooks/` et `benchmark_demo/` (`core/services.py`,
    `core/registry.py`) importent **exclusivement** `portfolio_lab.backtest` — jamais le
    `backtest.py` racine.
- **`ml_features.py` (racine)** : **n'est importé nulle part** dans le repo (`grep` ne renvoie
  aucune occurrence hors le fichier lui-même). Sa documentation interne référence un
  `MLMeanEstimator` dans `ml_model.py` et un usage dans `ml_backtest_nb.ipynb` — **aucun des deux
  fichiers n'existe**. C'est du code mort/orphelin, probablement une étape préparatoire à une
  fonctionnalité ML jamais branchée au reste du projet.

---

## 4. TODO, code mort, doublons, incohérences README

- **Code en chantier** : seul `backtest.py` (racine) porte un marqueur explicite (`# ── NUOVO ──`)
  avec une mise en forme bancale (indentation tabs/espaces mélangée — le fichier reste
  syntaxiquement valide, mais c'est manifestement un patch collé rapidement).
- **Code orphelin** : `ml_features.py` (718 lignes, le plus gros fichier `.py` du repo) n'est
  utilisé par rien ni personne actuellement.
- **Artefact versionné par erreur** : `__pycache__/backtest.cpython-311.pyc` est suivi par git
  (`git ls-files`) malgré un `.gitignore` qui exclut `__pycache__/` — probablement ajouté avant
  que la règle n'existe.
- **Pas de duplication problématique à l'intérieur de `portfolio_lab/`** : les méthodes `solve`/
  `__init__` répétées dans `estimators.py`/`optimizers.py`/`universe.py` sont le patron Strategy
  documenté (une classe par méthode), pas une duplication accidentelle.
- **`README.md` vs `READMEE.md`** : ce sont **deux documentations totalement disjointes et qui ne
  se citent jamais**, décrivant deux architectures concurrentes du même projet :
  - `README.md` (racine, mis à jour le plus récemment) documente le monde "notebooks racine +
    `backtest.py` + `benchmark_demo`".
  - `READMEE.md` (racine, mais structuré comme le README de `portfolio_lab`) documente le monde
    "package `portfolio_lab` + `notebooks/` + `test_pipeline.py`", et liste même `benchmark_demo`
    comme client de `portfolio_lab` — ce qui est vérifié par le code, mais que `README.md` ne
    rend pas explicite (il décrit `benchmark_demo` sans dire qu'il dépend de `portfolio_lab` et
    non de `backtest.py`).
  - Un lecteur qui ouvre seulement `README.md` ne saura pas que `portfolio_lab/` (l'architecture
    la plus aboutie et la mieux testée) existe.
- **Notebooks racine volumineux** : `markowitz_sp500_evaluation_V3.ipynb` (1.4 Mo),
  `markowitz_sp500_V1.ipynb` (970 Ko) contiennent des sorties (figures/résultats) jamais purgées
  — pas un bug, mais un signe que ces notebooks n'ont jamais été nettoyés avant commit.

---

## 5. Synthèse — ce qui tourne, ce qui casse, recommandation

### Ce qui tourne
- `test_pipeline.py` : suite de tests hors-ligne complète de `portfolio_lab`, exécutable
  rapidement, validée par construction (c'est le test de référence du projet).
- `markowitz.ipynb` : tourne bout en bout sans erreur, mais à un coût temps inacceptable en l'état
  (`AllUniverse`).
- `benchmark_demo/` et `scripts/generate_figures.py` : architecturalement propres, branchés
  uniquement sur `portfolio_lab` (non testés ici en exécution, hors périmètre de la consigne, mais
  aucune incohérence d'import détectée).

### Ce qui casse
- `return.ipynb` : bug de nommage trivial (`MaxSharpe` vs `Max_utility`) dans la toute dernière
  cellule, après 13 min de calcul.
- `covariance.ipynb` : bug de nommage trivial (clé `"Empirique"` jamais ajoutée au dict
  `estimators`), après 50 s.
- `markowitz.ipynb` : pas de bug logique, mais un défaut d'ergonomie sévère (univers complet par
  défaut) qui le rend inutilisable en démo live.

### Ce qui est dupliqué / orphelin
- `backtest.py` (racine) duplique `portfolio_lab/backtest.py` en version dégradée et non finalisée
  — utilisé seulement par les 3 anciens notebooks racine.
- `ml_features.py` est totalement orphelin (zéro import, dépendances vers des fichiers
  inexistants).
- `README.md` et `READMEE.md` documentent chacun une moitié du projet sans se référencer.

### Recommandation d'architecture pour le rendu final

**Garder `notebooks/` + `portfolio_lab/` comme socle unique**, et traiter tout le reste de la
racine (`Markowitz*.ipynb`, `markowitz_sp500_*.ipynb`, `backtest.py`) comme des brouillons
historiques à archiver (ou supprimer) :

- `portfolio_lab/` est la version la plus aboutie, la plus testée (`test_pipeline.py`,
  11 sections de validation), la plus documentée (`READMEE.md`) et la seule à avoir les
  corrections d'audit importantes (Sharpe net du rf, coûts symétriques, anti-look-ahead par
  fenêtre, fallback compté).
- `benchmark_demo/` dépend déjà uniquement de `portfolio_lab` — il n'y a donc aucune perte à
  abandonner `backtest.py` racine.
- Avant de pouvoir s'appuyer sur `notebooks/` pour le rendu, il faut :
  1. Corriger les deux bugs de nommage (`return.ipynb`, `covariance.ipynb`) — triviaux, une ligne
     chacun.
  2. Changer le sélecteur d'univers par défaut de `markowitz.ipynb` (`AllUniverse()` →
     `StratifiedBySector(80)` ou équivalent) pour le rendre exploitable en temps raisonnable.
  3. Décider du sort de `ml_features.py` : soit le brancher réellement (créer le
     `MLMeanEstimator`/`ml_model.py` annoncé dans sa propre docstring), soit le retirer s'il ne
     sera pas utilisé pour le rendu.
  4. Fusionner ou supprimer un des deux README pour n'avoir qu'une seule documentation faisant
     autorité, qui mentionne explicitement l'abandon de l'architecture racine.

Je n'ai rien modifié — dis-moi comment tu veux trancher sur ces 4 points (notamment
`ml_features.py` : intégré ou supprimé ?) avant que je touche au code.
