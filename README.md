# Projet M2 DE - Optimisation du Retour sur Investissement Marketing

Ce projet implemente un systeme intelligent multi-modeles pour predire les ventes, comparer des modeles de regression et de classification, et recommander une allocation budgetaire qui maximise le ROI marketing.

## 1) Objectif

A partir du dataset `Dummy Data HSS.csv` (TV, Radio, Social Media, Influencer, Sales), la solution couvre:

- Regression: prediction de `Sales`
- Classification: prediction d'une classe de ventes (`Low`, `Medium`, `High`)
- Optimisation: recommandation de repartition budgetaire sous contrainte
- Dashboard decisionnel: simulation interactive avec Streamlit

## 2) Structure du projet

- `data/raw/Dummy Data HSS.csv`: dataset source
- `src/eda.py` : analyse exploratoire des données et génération automatique des graphiques EDA
- `src/data_utils.py`: chargement, nettoyage, features (`ROI`, `SalesClass`), split
- `src/train.py`: entrainement et comparaison des modeles (ML + MLP)
- `src/optimize_roi.py`: optimisation du budget marketing
- `app.py`: dashboard Streamlit
- `run_project.py`: pipeline complet (train + optimisation)
- `tests/smoke_test.py`: test de fumee minimal
- `reports/`: metriques, meilleurs modeles, optimisation, rapport
- `models/`: modeles entraines (`.joblib`)

## 3) Modeles compares

### Regression (obligatoire)

- `LinearRegression`
- `RandomForestRegressor`
- `GradientBoostingRegressor`
- `MLPRegressor` (Deep Learning)

### Classification (obligatoire)

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `MLPClassifier` (Deep Learning)

## 4) Resultats obtenus

Fichiers de sortie:

- `reports/eda/sales_distribution.png`
- `reports/eda/TV_distribution.png`
- `reports/eda/Radio_distribution.png`
- `reports/eda/Social Media_distribution.png`
- `reports/eda/TV_vs_sales.png`
- `reports/eda/Radio_vs_sales.png`
- `reports/eda/Social Media_vs_sales.png`
- `reports/eda/correlation_matrix.png`
- `reports/eda/sales_by_influencer.png`
- `reports/eda/outliers.png`
- `reports/regression_metrics.csv`
- `reports/classification_metrics.csv`
- `reports/best_models.json`
- `reports/budget_optimization.json`

Meilleurs modeles observes:

- Regression: `RandomForestRegressor` (`R2` ~ 0.9945)
- Classification: `RandomForestClassifier` (`F1_macro` ~ 0.9869)

## 5) Installation et execution

1. Installer les dependances:

```bash
python -m pip install -r requirements.txt
```

2. Lancer le pipeline complet:

```bash
python run_project.py
```

3. Lancer le test minimal:

```bash
python tests/smoke_test.py
```

4. Lancer le dashboard:

```bash
streamlit run app.py
```

## 6) Notes methodologiques

- EDA : analyse des distributions, corrélations, relations budgets/ventes, influence du type d’influenceur et détection visuelle des outliers
- Valeurs manquantes: imputation par mediane sur variables numeriques
- Encodage categoriel: `OneHotEncoder` pour `Influencer`
- Evaluation regression: MAE, RMSE, R2 + validation croisee
- Evaluation classification: Accuracy, F1_macro + validation croisee
- Optimisation ROI: contrainte somme des ratios = 1 et bornes canal

## 7) Limites et ameliorations

- Ajouter SHAP pour une interpretabilite plus fine
- Integrer recherche d'hyperparametres (GridSearchCV/Optuna)
- Ajouter une API REST (FastAPI) en option
- Renforcer les tests unitaires et d'integration

## 8) Livrables

- Code source fonctionnel
- Analyse exploratoire générée dans `reports/eda/`
- Rapport: `reports/final_report.md`
- Dashboard Streamlit
- Artefacts modeles + metriques dans `models/` et `reports/`

