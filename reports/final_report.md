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

### 4.3 Lecture metier des resultats

Les scores obtenus montrent que le projet n'est pas seulement performant sur le plan technique : il est surtout **fiable pour piloter des decisions budgetaires**.

#### Ce que cela signifie pour le business

- **La prediction des ventes est très robuste.**
  Avec un R² de l'ordre de **0.994**, le modele de regression explique quasiment toute la variation observée dans les ventes. Pour un utilisateur metier, cela veut dire que les estimations de ventes sont suffisamment stables pour servir de base a une discussion budgetaire.

- **Les campagnes sont tres bien classees.**
  Le meilleur modele de classification atteint une **F1-macro de 0.9869**. Concretement, le systeme classe tres bien les campagnes entre les niveaux de performance, ce qui aide a identifier rapidement si une campagne est prometteuse ou non.

- **Les resultats sont coherents dans le temps et sur differents decoupages des donnees.**
  La validation croisee reste tres proche des scores de test : cela limite le risque de sur-apprentissage. En langage business, cela veut dire que le modele ne fonctionne pas seulement sur un cas isolé : il garde une logique stable lorsqu'on lui presente de nouveaux scenarii.

#### Recommandation de pilotage

Le projet peut etre utilise comme **outil d'aide a la decision** pour :

- valider un budget media avant lancement,
- comparer plusieurs scenarii de campagne,
- arbitrer entre croissance des ventes et niveau d'investissement,
- securiser les decisions d'allocation auprès du management.

## 5. Optimisation ROI

Une optimisation sous contraintes est implemente dans `src/optimize_roi.py` pour repartir un budget total entre `TV`, `Radio`, `Social Media` avec:

- somme des ratios = 1,
- bornes min/max par canal,
- maximisation du ROI predit.

Exemple de sortie (`reports/budget_optimization.json`):

- budget total: 120
- influenceur: Mega
- ROI predit: 2.8560

### 5.1 Lecture metier de l'optimisation

L'optimisation met en avant une logique simple pour le metier : **le modele recommande de concentrer la plus grande partie du budget sur le canal qui genere le plus de ventes pour chaque euro investi**.

D'apres les resultats disponibles :

- la meilleure repartition pour un budget de reference est de l'ordre de **80% TV / 10% Radio / 10% Social Media**,
- le ROI predit atteint environ **2.82 a 2.86** selon le profil d'influenceur et le budget teste,
- le canal **TV** est le principal levier de creation de valeur,
- **Radio** joue un role de soutien,
- **Social Media** reste utile mais apparait moins determinant que la TV sur ce dataset.

#### Pourquoi c'est important pour le business

Cette recommandation permet de :

- mettre le budget la ou il est le plus rentable,
- reduire les depenses moins efficaces,
- gagner en coherence entre objectifs commerciaux et plan media,
- presenter une allocation simple a comprendre et facile a defender en comite.

#### Point d'attention

Le modele travaille avec les donnees disponibles : il faut donc interprete cette recommandation comme une **base de pilotage**, pas comme une regle absolue. En pratique, d'autres contraintes metier peuvent s'ajouter : stock, calendrier commercial, pression concurrentielle, ou obligations de marque.

### 5.2 Recommandations operationnelles

1. **Prioriser TV comme levier principal** lorsque l'objectif est la performance commerciale pure.
2. **Conserver Radio comme canal de complement** pour soutenir la couverture et l'amplification.
3. **Utiliser Social Media comme levier d'appoint**, surtout pour les campagnes de soutien ou de visibilite.
4. **Arbitrer les budgets selon le ROI attendu** plutot que par habitude historique.
5. **Revoir l'allocation a chaque nouvelle campagne** si le contexte commercial change fortement.

## 6. Dashboard interactif

Le dashboard `app.py` (Streamlit) permet:

- simulation d'un scenario budgetaire,
- prediction de ventes en temps reel,
- estimation ROI,
- visualisation de la comparaison des modeles,
- affichage d'une recommandation budgetaire.

### 6.1 Valeur business du dashboard

Le dashboard transforme les modeles en un **outil de decision utilisable par un profil metier**. Il apporte trois gains principaux :

- **Rapidité** : l'utilisateur teste plusieurs budgets sans refaire une analyse manuelle.
- **Lisibilite** : les resultats sont presentes sous forme de recommandations et de graphiques simples.
- **Confiance** : les sections de performance, de comparaison et d'interpretation permettent de justifier la recommandation.

En pratique, le dashboard peut servir a un responsable marketing, a une direction commerciale ou a un manager financier pour preparer une campagne, comparer plusieurs allocations et choisir un budget plus rentable.

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

### 7.3 Lecture metier de `linear_friendly`

Le mode d'entrainement `linear_friendly` signifie ici que les donnees suivent une logique suffisamment claire pour privilegier un modele interpretable et stable.

Pour un utilisateur metier, cela apporte trois benefices directs :

- **des recommandations plus faciles a comprendre**,
- **un impact budgetaire plus previsible**,
- **moins de surprises lors des arbitrages**.

Autrement dit, si on augmente le budget sur un canal, le modele aide a estimer plus simplement l'effet attendu sur les ventes. C'est utile quand on veut expliquer la decision au management et pas seulement produire un score technique.

### 7.4 Feature Importance (importance globale)

Objectif: identifier les variables les plus influentes sur le comportement global du modele (vision macro).

Exemple metier:

- si `Social Media` a une importance elevee, ce canal influence fortement la prediction des ventes a l'echelle globale.

Approches:

- importance native des arbres (`model.feature_importances_`) pour `RandomForest` ou `GradientBoosting`,
- mesure basee sur reduction d'impurete/variance.

### 7.5 Permutation Importance (recommandee)

Objectif: mesurer l'impact reel d'une variable sur la performance, de maniere agnostique au modele.

Principe:

1. mesurer la performance initiale,
2. permuter une variable,
3. re-evaluer la performance,
4. observer la chute de performance.

Interpretation:

- plus la performance se degrade, plus la variable est importante.

### 7.6 SHAP (explicabilite locale et globale)

SHAP permet d'expliquer:

- **localement**: pourquoi une prediction individuelle est haute ou basse,
- **globalement**: quelles variables dominent le modele,
- le sens de l'effet (positif ou negatif) de chaque variable.

Exemple local:

- pour une campagne donnee, SHAP peut montrer que `TV` pousse la prediction vers le haut, alors qu'un `Radio` tres eleve apporte un gain marginal plus faible.

Exemple global:

- la combinaison `Social Media + Influencer Macro` peut apparaitre comme favorable a la performance globale.

### 7.7 Lecture metier de la validation croisee

La validation croisee montre que les performances restent elevees meme quand on change legerement la facon de separer les donnees.

En business, cela veut dire que la recommandation n'est pas juste bonne "sur le papier" : elle reste credible sur plusieurs cas de figure. C'est un point essentiel pour un modele qui doit aider a decider des budgets media, car on cherche une solution **stable, defendable et repetable**.

## 8. Interpretation metier du modele retenu

Le couple performance/stabilite des modeles arbres en fait le meilleur compromis pour ce dataset.

Interpretation business attendue:

- prioriser les canaux avec impact fort et stable,
- surveiller les zones de rendement marginal decroissant,
- justifier les recommandations budgetaires a partir des explications globales et locales.

### 8.1 Recommandation finale au metier

Au vu des resultats, la recommandation la plus solide est la suivante :

- **TV doit rester le canal prioritaire**,
- **Radio doit conserver une place secondaire mais utile**,
- **Social Media doit etre pilote avec prudence** car son impact est moins marque sur ce dataset.

Le projet fournit donc un cadre de decision qui permet de **mieux utiliser le budget marketing** et d'aligner les equipes sur une logique de performance mesurable.

## 9. Limites et perspectives

- variables metier limitees (pas de saisonnalite, concurrence, segmentation),
- optimisation encore simple au regard de contraintes reelles (plafonds contractuels, priorites produit),
- prochaine etape: integrer explicitement les graphes d'importance et SHAP dans le dashboard.

## 10. Conclusion

Le projet fournit une chaine complete: donnees -> modeles -> evaluation -> optimisation -> dashboard.

L'ajout d'une demarche d'explicabilite (Feature Importance, Permutation Importance, SHAP) permet de passer d'un modele performant a un outil decisionnel justifiable pour des profils metier (CMO, finance, direction commerciale).

### 10.1 Conclusion metier

Au final, le projet ne sert pas seulement a prevoir des ventes. Il sert surtout a **mieux investir**.

La valeur pour l'entreprise est la suivante :

- choisir une repartition budgetaire plus rentable,
- reduire le risque de decision approximative,
- expliquer les recommandations a des interlocuteurs non techniques,
- piloter les campagnes marketing avec une logique ROI.

En resume, ce projet transforme des donnees marketing en **aide a la decision exploitable par le business**.

