from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from src.optimize_roi import optimize_budget


ROOT = Path(__file__).resolve().parent

REG_MODEL_PATH = ROOT / "models" / "best_regression_model.joblib"
CLF_MODEL_PATH = ROOT / "models" / "best_classification_model.joblib"

REG_METRICS_PATH = ROOT / "reports" / "regression_metrics.csv"
CLF_METRICS_PATH = ROOT / "reports" / "classification_metrics.csv"
BEST_MODELS_PATH = ROOT / "reports" / "best_models.json"
CORR_PATH = ROOT / "reports" / "correlation_with_sales.csv"
CV_PATH = ROOT / "reports" / "cv_summary.csv"
POLICY_PATH = ROOT / "reports" / "model_selection_policy.json"


@st.cache_resource
def load_regression_model():
    if not REG_MODEL_PATH.exists():
        return None
    return joblib.load(REG_MODEL_PATH)


@st.cache_resource
def load_classification_model():
    if not CLF_MODEL_PATH.exists():
        return None
    return joblib.load(CLF_MODEL_PATH)


def safe_get(data: dict, key: str, default=None):
    return data[key] if key in data else default


st.set_page_config(page_title="Marketing ROI Dashboard", layout="wide")
st.title("Marketing ROI Optimization Dashboard")

reg_model = load_regression_model()
clf_model = load_classification_model()

if reg_model is None:
    st.warning("Regression model not found. Run `python src/train.py` first.")
    st.stop()


@st.cache_data(show_spinner=False)
def run_dynamic_optimization(total_budget: float, influencer: str):
    return optimize_budget(
        model=reg_model,
        total_budget=total_budget,
        influencer=influencer,
        random_trials=120,
        grid_step=0.05,
    )


st.sidebar.header("Campaign Inputs")

tv = st.sidebar.slider("TV budget", min_value=0.0, max_value=300.0, value=80.0, step=1.0)
radio = st.sidebar.slider("Radio budget", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
social = st.sidebar.slider("Social Media budget", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
influencer = st.sidebar.selectbox("Influencer tier", ["Mega", "Macro", "Micro", "Nano"])

input_df = pd.DataFrame(
    [{"TV": tv, "Radio": radio, "Social Media": social, "Influencer": influencer}]
)

pred_sales = float(reg_model.predict(input_df)[0])
total_budget = tv + radio + social
pred_roi = pred_sales / max(total_budget, 1e-9)

pred_class = None
if clf_model is not None:
    pred_class = clf_model.predict(input_df)[0]


col1, col2, col3, col4 = st.columns(4)

col1.metric("Predicted Sales", f"{pred_sales:,.2f}")
col2.metric("Total Budget", f"{total_budget:,.2f}")
col3.metric("Estimated ROI", f"{pred_roi:.3f}")

if pred_class is not None:
    col4.metric("Campaign Class", str(pred_class))
else:
    col4.metric("Campaign Class", "N/A")


tab_roi, tab_perf, tab_compare, tab_analysis = st.tabs(
    [
        "Simulation ROI",
        "Performances des modèles",
        "Comparaison des modèles",
        "Analyse & interprétabilité",
    ]
)


with tab_roi:
    st.subheader("Scénario sélectionné par l'utilisateur")
    st.dataframe(input_df, use_container_width=True)

    st.info(
        "Cette section permet à un responsable marketing de simuler un budget de campagne, "
        "d’estimer les ventes attendues, le ROI et la classe de performance de la campagne."
    )

    st.subheader("Prédiction du scénario")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Ventes prédites", f"{pred_sales:.2f}")
    col_b.metric("ROI estimé", f"{pred_roi:.3f}")
    col_c.metric("Classe prédite", str(pred_class) if pred_class is not None else "N/A")

    scenario_df = pd.DataFrame(
        {
            "Canal": ["TV", "Radio", "Social Media"],
            "Budget": [tv, radio, social],
        }
    )

    st.subheader("Répartition actuelle du budget")
    st.bar_chart(scenario_df.set_index("Canal"))

    if total_budget <= 0:
        st.warning("Veuillez saisir un budget total supérieur à 0 pour lancer l’optimisation.")
    else:
        with st.spinner("Optimisation dynamique de la répartition budgétaire..."):
            rec = run_dynamic_optimization(total_budget, influencer)

        rec_df = pd.DataFrame(
            {
                "Canal": list(rec["recommended_budget"].keys()),
                "Budget recommandé": list(rec["recommended_budget"].values()),
                "Ratio recommandé": [
                    rec["recommended_ratio"][k]
                    for k in rec["recommended_budget"].keys()
                ],
            }
        )

        st.subheader("Répartition budgétaire recommandée")

        col_a, col_b = st.columns(2)
        col_a.metric("Ventes prédites après optimisation", f"{rec['predicted_sales']:.2f}")
        col_b.metric("ROI prédit après optimisation", f"{rec['predicted_roi']:.3f}")

        st.dataframe(rec_df, use_container_width=True)

        st.subheader("Budget recommandé par canal")
        st.bar_chart(rec_df.set_index("Canal")[["Budget recommandé"]])

        st.subheader("Ratio recommandé par canal")
        st.bar_chart(rec_df.set_index("Canal")[["Ratio recommandé"]])

        best_channel_row = rec_df.sort_values("Ratio recommandé", ascending=False).iloc[0]
        best_channel = best_channel_row["Canal"]
        best_ratio = best_channel_row["Ratio recommandé"]

        st.subheader("Recommandation business")
        st.success(
            f"L’allocation optimisée recommande de prioriser **{best_channel}**, "
            f"avec **{best_ratio:.0%}** du budget marketing total."
        )

        st.write(
            "Selon le modèle entraîné, cette répartition devrait maximiser le ROI prédit "
            "sous les contraintes budgétaires définies."
        )

        if "marginal_impact" in rec:
            st.subheader("Analyse de l’impact marginal")

            impact_rows = []
            for channel, values in rec["marginal_impact"].items():
                increase = values["delta_sales_if_ratio_increases"]
                decrease = values["delta_sales_if_ratio_decreases"]

                impact_rows.append(
                    {
                        "Canal": channel,
                        "Variation des ventes si la part augmente": (
                            round(increase, 4)
                            if increase is not None
                            else "Non applicable - limite atteinte"
                        ),
                        "Variation des ventes si la part diminue": (
                            round(decrease, 4)
                            if decrease is not None
                            else "Non applicable - limite atteinte"
                        ),
                    }
                )

            impact_df = pd.DataFrame(impact_rows)
            st.dataframe(impact_df, use_container_width=True)

            st.info(
                "L’impact marginal estime comment les ventes prédites évoluent lorsqu’une "
                "part du budget est déplacée d’un canal vers un autre, tout en gardant le "
                "même budget total."
            )

        if "search_diagnostics" in rec:
            diagnostics = rec["search_diagnostics"]
            st.subheader("Résumé de la méthode d’optimisation")

            st.write(
                f"L’optimisation a testé **{diagnostics['candidate_points_tested']}** "
                f"combinaisons budgétaires avec la méthode : "
                f"**{diagnostics['method']}**."
            )


with tab_perf:
    st.subheader("Performance des modèles")

    reg_metrics_df = None
    clf_metrics_df = None

    if REG_METRICS_PATH.exists():
        reg_metrics_df = pd.read_csv(REG_METRICS_PATH)

    if CLF_METRICS_PATH.exists():
        clf_metrics_df = pd.read_csv(CLF_METRICS_PATH)

    st.markdown("### Régression")
    if reg_metrics_df is not None and not reg_metrics_df.empty:
        st.dataframe(reg_metrics_df, use_container_width=True)

        if "R2" in reg_metrics_df.columns:
            st.write("Comparaison des modèles selon le R²")
            st.bar_chart(reg_metrics_df[["Model", "R2"]].set_index("Model"))

        if "RMSE" in reg_metrics_df.columns:
            st.write("Comparaison des modèles selon le RMSE")
            st.bar_chart(reg_metrics_df[["Model", "RMSE"]].set_index("Model"))

        if {"Model", "R2", "RMSE"}.issubset(reg_metrics_df.columns):
            best_reg_row = reg_metrics_df.sort_values(
                by=["R2", "RMSE"],
                ascending=[False, True],
            ).iloc[0]

            st.success(
                f"En régression, le meilleur modèle observé est **{best_reg_row['Model']}**. "
                f"Il obtient un R² de **{best_reg_row['R2']:.4f}** et un RMSE de "
                f"**{best_reg_row['RMSE']:.4f}**."
            )
    else:
        st.info("Le fichier de métriques de régression est manquant ou vide.")

    st.markdown("### Classification")
    if clf_metrics_df is not None and not clf_metrics_df.empty:
        st.dataframe(clf_metrics_df, use_container_width=True)

        if "F1_macro" in clf_metrics_df.columns:
            st.write("Comparaison des modèles selon le F1-macro")
            st.bar_chart(clf_metrics_df[["Model", "F1_macro"]].set_index("Model"))

        if {"Model", "Accuracy", "F1_macro"}.issubset(clf_metrics_df.columns):
            best_clf_row = clf_metrics_df.sort_values(
                by=["F1_macro", "Accuracy"],
                ascending=[False, False],
            ).iloc[0]

            st.success(
                f"En classification, le meilleur modèle observé est **{best_clf_row['Model']}**. "
                f"Il obtient une Accuracy de **{best_clf_row['Accuracy']:.4f}** et un F1-macro "
                f"de **{best_clf_row['F1_macro']:.4f}**."
            )
    else:
        st.info("Le fichier de métriques de classification est manquant ou vide.")

    st.markdown("---")
    st.subheader("Informations complémentaires")

    if BEST_MODELS_PATH.exists():
        with BEST_MODELS_PATH.open("r", encoding="utf-8") as f:
            best_models = json.load(f)

        st.success(
            f"Meilleur modèle de régression : **{best_models['best_regression_model']}** "
            f"avec R² = **{best_models['best_regression_r2']:.4f}** "
            f"et CV R² = **{best_models['best_regression_cv_r2']:.4f}**."
        )

        st.success(
            f"Meilleur modèle de classification : **{best_models['best_classification_model']}** "
            f"avec F1-macro = **{best_models['best_classification_f1_macro']:.4f}** "
            f"et CV F1-macro = **{best_models['best_classification_cv_f1_macro']:.4f}**."
        )

        st.write(
            "Les modèles finaux sont sélectionnés à partir des performances sur le jeu de test "
            "et des scores de validation croisée afin de limiter le risque de choisir un modèle "
            "performant uniquement sur un découpage spécifique."
        )

    if CV_PATH.exists():
        cv_df = pd.read_csv(CV_PATH)
        st.write("Synthèse de la validation croisée")
        st.dataframe(cv_df, use_container_width=True)


with tab_compare:
    st.subheader("Comparaison des résultats des modèles")

    st.info(
        "Cet onglet compare les modèles de régression et de classification. "
        "Les métriques ne sont pas directement comparables entre tâches, "
        "mais permettent de classer les modèles à l’intérieur de chaque famille."
    )

    reg_metrics_df = pd.read_csv(REG_METRICS_PATH) if REG_METRICS_PATH.exists() else None
    clf_metrics_df = pd.read_csv(CLF_METRICS_PATH) if CLF_METRICS_PATH.exists() else None

    if reg_metrics_df is None and clf_metrics_df is None:
        st.warning("Aucun fichier de métriques trouvé. Lance `python src/train.py`.")
    else:
        comparison_rows = []

        if reg_metrics_df is not None:
            reg_ranked = reg_metrics_df.sort_values(
                by=["R2", "RMSE"],
                ascending=[False, True],
            ).copy()
            reg_ranked["Rang"] = range(1, len(reg_ranked) + 1)
            reg_ranked["Tâche"] = "Régression"
            reg_ranked["Score principal"] = reg_ranked["R2"]
            reg_ranked["Métrique principale"] = "R2"

            st.write("Classement - Régression")
            st.dataframe(
                reg_ranked[["Rang", "Model", "R2", "RMSE"]],
                use_container_width=True,
            )
            st.bar_chart(reg_ranked[["Model", "R2"]].set_index("Model"))

            comparison_rows.extend(
                reg_ranked[
                    ["Tâche", "Model", "Métrique principale", "Score principal"]
                ].to_dict("records")
            )

        if clf_metrics_df is not None:
            clf_ranked = clf_metrics_df.sort_values(
                by=["F1_macro", "Accuracy"],
                ascending=[False, False],
            ).copy()
            clf_ranked["Rang"] = range(1, len(clf_ranked) + 1)
            clf_ranked["Tâche"] = "Classification"
            clf_ranked["Score principal"] = clf_ranked["F1_macro"]
            clf_ranked["Métrique principale"] = "F1_macro"

            st.write("Classement - Classification")
            st.dataframe(
                clf_ranked[["Rang", "Model", "F1_macro", "Accuracy"]],
                use_container_width=True,
            )
            st.bar_chart(clf_ranked[["Model", "F1_macro"]].set_index("Model"))

            comparison_rows.extend(
                clf_ranked[
                    ["Tâche", "Model", "Métrique principale", "Score principal"]
                ].to_dict("records")
            )

        if comparison_rows:
            st.subheader("Vue consolidée")
            consolidated_df = pd.DataFrame(comparison_rows)
            consolidated_df["Modèle"] = consolidated_df["Tâche"] + " - " + consolidated_df["Model"]

            st.dataframe(
                consolidated_df[
                    ["Tâche", "Model", "Métrique principale", "Score principal"]
                ],
                use_container_width=True,
            )
            st.bar_chart(consolidated_df[["Modèle", "Score principal"]].set_index("Modèle"))


with tab_analysis:
    st.subheader("Corrélation avec les ventes")

    if CORR_PATH.exists():
        corr_df = pd.read_csv(CORR_PATH)
        st.dataframe(corr_df, use_container_width=True)

        chart_df = corr_df[["Feature", "CorrelationWithSales"]].set_index("Feature")
        st.bar_chart(chart_df)

        if "AbsCorrelation" in corr_df.columns:
            top_feature = corr_df.sort_values("AbsCorrelation", ascending=False).iloc[0]

            st.success(
                f"La variable la plus influente est **{top_feature['Feature']}**, "
                f"avec une corrélation de **{top_feature['CorrelationWithSales']:.3f}** "
                f"avec les ventes."
            )

            st.write(
                "Cela signifie que cette variable présente la relation la plus forte avec les ventes "
                "dans le dataset. D’un point de vue métier, elle doit donc être suivie de près "
                "lors de la définition de la stratégie budgétaire marketing."
            )
    else:
        st.info("Le fichier de corrélation est manquant. Lance `python src/train.py`.")

    st.subheader("Synthèse de la validation croisée")

    if CV_PATH.exists():
        cv_df = pd.read_csv(CV_PATH)
        st.dataframe(cv_df, use_container_width=True)

        if {"Model", "CVScore"}.issubset(cv_df.columns):
            st.bar_chart(cv_df[["Model", "CVScore"]].set_index("Model"))
    else:
        st.info("Le fichier de synthèse CV est manquant.")

    st.subheader("Stratégie d’entraînement")

    if POLICY_PATH.exists():
        with POLICY_PATH.open("r", encoding="utf-8") as f:
            policy = json.load(f)

        training_mode = safe_get(policy, "training_mode", "unknown")
        max_corr = safe_get(policy, "max_abs_numeric_correlation", None)

        if max_corr is not None:
            st.info(
                f"La stratégie d’entraînement a choisi une approche **{training_mode}** "
                f"car la corrélation numérique la plus forte avec Sales est de **{max_corr:.3f}**."
            )
        else:
            st.info(
                f"La stratégie d’entraînement a choisi une approche **{training_mode}**."
            )

        if "top_correlated_features" in policy:
            st.write("Variables les plus corrélées utilisées pour guider la stratégie de modélisation :")

            top_features_df = pd.DataFrame(policy["top_correlated_features"])
            st.dataframe(top_features_df, use_container_width=True)
    else:
        st.info("Le fichier de politique d'entraînement est manquant.")