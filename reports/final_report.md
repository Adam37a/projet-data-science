# Rapport Final - Projet 3

## Optimisation du Retour sur Investissement Marketing

## 1. Contexte et problematique

Dans ce projet, la problematique traitee est:

**Comment predire les ventes et optimiser l'allocation budgetaire marketing (TV, Radio, Social Media, Influencer) afin de maximiser le ROI?**

Le systeme developpe repond a trois objectifs:

- analyser les données marketing via une EDA
- predire `Sales` (regression),
- classer les campagnes en niveaux de performance (`SalesClass`: Low, Medium, High),
- recommander une repartition budgetaire orientee ROI.

## 2. Donnees, préparation et EDA

Dataset: `data/raw/Dummy Data HSS.csv`

- 4572 lignes
- Variables: `TV`, `Radio`, `Social Media`, `Influencer`, `Sales`

### 2.1 Analyse exploratoire des données

Une analyse exploratoire a été ajoutée dans `src/eda.py`

L'objectif est de comprendre les distributions, identifier les relations entre budgets et ventes, analyser les corrélations,
observer l’impact du type d’influenceur,
et détecter visuellement les valeurs extrêmes avant modélisation.

Elle génère automatiquement des graphiques dans : 
```text 
reports/eda/ 
```
Graphiques générés :

- distribution des ventes : sales_distribution.png
- distributions des budgets : TV_distribution.png, Radio_distribution.png, Social Media_distribution.png
- relations budget / ventes : TV_vs_sales.png, Radio_vs_sales.png, Social Media_vs_sales.png
matrice de corrélation : correlation_matrix.png
- ventes selon le type d’influenceur : sales_by_influencer.png
- détection visuelle des outliers : outliers.png

Traitements appliques (`src/data_utils.py`):

- conversion numerique des colonnes quantitatives,
- imputation des valeurs manquantes par la mediane,
- creation de `TotalBudget` et `ROI`,
- discretisation de `Sales` en `SalesClass` pour la classification.

## 3. Methodologie de modelisation

Pipeline (`src/train.py`):

1. Split train/test (80/20)
2. Preprocessing avec `StandardScaler` (numeriques) + `OneHotEncoder` (categorielle)
3. Entrainement de plusieurs modeles
4. Evaluation quantitative et comparaison
5. Selection et sauvegarde du meilleur modele

Modeles testes:

- Regression: `LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, `MLPRegressor`
- Classification: `LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `MLPClassifier`

## 4. Resultats quantitatifs

### 4.1 Regression

| Modele | MAE | RMSE | R2 | CV_R2 |
|---|---:|---:|---:|---:|
| RandomForestRegressor | 2.9375 | 6.7931 | 0.9945 | 0.9945 |
| GradientBoostingRegressor | 2.8392 | 7.0110 | 0.9941 | 0.9940 |
| LinearRegression | 2.8645 | 8.1130 | 0.9921 | 0.9936 |
| MLPRegressor | 3.5787 | 8.5053 | 0.9913 | 0.9928 |

Modele retenu: `RandomForestRegressor`.

### 4.2 Classification

| Modele | Accuracy | F1_macro | CV_F1_macro |
|---|---:|---:|---:|
| RandomForestClassifier | 0.9869 | 0.9869 | 0.9819 |
| GradientBoostingClassifier | 0.9825 | 0.9825 | 0.9797 |
| MLPClassifier | 0.9814 | 0.9814 | 0.9777 |
| LogisticRegression | 0.9781 | 0.9782 | 0.9771 |

Modele retenu: `RandomForestClassifier`.

## 5. Optimisation ROI

Une optimisation sous contraintes est implemente dans `src/optimize_roi.py` pour repartir un budget total entre `TV`, `Radio`, `Social Media` avec:

- somme des ratios = 1,
- bornes min/max par canal,
- maximisation du ROI predit.

Exemple de sortie (`reports/budget_optimization.json`):

- budget total: 120
- influenceur: Mega
- ROI predit: 1.1908

## 6. Dashboard interactif

Le dashboard `app.py` (Streamlit) permet:

- simulation d'un scenario budgetaire,
- prediction de ventes en temps reel,
- estimation ROI,
- visualisation de la comparaison des modeles,
- affichage d'une recommandation budgetaire.

## 7. Explicabilite des modeles (section demandee)

### 7.1 Quand applique-t-on les techniques d'explicabilite?

Les techniques d'explicabilite s'appliquent **apres l'entrainement du modele**, pendant la phase d'evaluation et d'interpretation des predictions.

Pourquoi:

- elles expliquent les decisions d'un modele deja appris,
- elles repondent a: **"Pourquoi le modele a produit cette prediction?"**
- sans modele entraine et sans predictions, il n'y a rien a interpreter.

### 7.2 Tableau de synthese

| Technique | Quand l'utiliser | Niveau |
|---|---|---|
| `feature_importances_` | Juste apres entrainement (modeles arbres) | Basique |
| Permutation Importance | Apres evaluation des performances | Recommande |
| SHAP | Sur le modele final selectionne | Avance |

### 7.3 Feature Importance (importance globale)

Objectif: identifier les variables les plus influentes sur le comportement global du modele (vision macro).

Exemple metier:

- si `Social Media` a une importance elevee, ce canal influence fortement la prediction des ventes a l'echelle globale.

Approches:

- importance native des arbres (`model.feature_importances_`) pour `RandomForest` ou `GradientBoosting`,
- mesure basee sur reduction d'impurete/variance.

### 7.4 Permutation Importance (recommandee)

Objectif: mesurer l'impact reel d'une variable sur la performance, de maniere agnostique au modele.

Principe:

1. mesurer la performance initiale,
2. permuter une variable,
3. re-evaluer la performance,
4. observer la chute de performance.

Interpretation:

- plus la performance se degrade, plus la variable est importante.

### 7.5 SHAP (explicabilite locale et globale)

SHAP permet d'expliquer:

- **localement**: pourquoi une prediction individuelle est haute ou basse,
- **globalement**: quelles variables dominent le modele,
- le sens de l'effet (positif ou negatif) de chaque variable.

Exemple local:

- pour une campagne donnee, SHAP peut montrer que `TV` pousse la prediction vers le haut, alors qu'un `Radio` tres eleve apporte un gain marginal plus faible.

Exemple global:

- la combinaison `Social Media + Influencer Macro` peut apparaitre comme favorable a la performance globale.

## 8. Interpretation metier du modele retenu

Le couple performance/stabilite des modeles arbres en fait le meilleur compromis pour ce dataset.

Interpretation business attendue:

- prioriser les canaux avec impact fort et stable,
- surveiller les zones de rendement marginal decroissant,
- justifier les recommandations budgetaires a partir des explications globales et locales.

## 9. Limites et perspectives

- variables metier limitees (pas de saisonnalite, concurrence, segmentation),
- optimisation encore simple au regard de contraintes reelles (plafonds contractuels, priorites produit),
- prochaine etape: integrer explicitement les graphes d'importance et SHAP dans le dashboard.

## 10. Conclusion

Le projet fournit une chaine complete: donnees -> modeles -> evaluation -> optimisation -> dashboard.

L'ajout d'une demarche d'explicabilite (Feature Importance, Permutation Importance, SHAP) permet de passer d'un modele performant a un outil decisionnel justifiable pour des profils metier (CMO, finance, direction commerciale).

