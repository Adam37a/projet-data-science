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

# Title & Logo
col_title, col_spacer = st.columns([1, 4])
with col_title:
    st.markdown("## 🎯 Marketing ROI Optimizer")

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
        random_trials=40,
        grid_step=0.1,
    )


# ============================================================================
# SIDEBAR: Campaign Inputs (Utilisateur)
# ============================================================================
st.sidebar.header("📊 Paramètres de la Campagne")

tv = st.sidebar.slider("TV budget", min_value=0.0, max_value=300.0, value=80.0, step=1.0)
radio = st.sidebar.slider("Radio budget", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
social = st.sidebar.slider("Social Media budget", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
influencer = st.sidebar.selectbox("Tier d'influenceur", ["Mega", "Macro", "Micro", "Nano"])

input_df = pd.DataFrame(
    [{"TV": tv, "Radio": radio, "Social Media": social, "Influencer": influencer}]
)

pred_sales = float(reg_model.predict(input_df)[0])
total_budget = tv + radio + social
pred_roi = pred_sales / max(total_budget, 1e-9)

pred_class = None
if clf_model is not None:
    pred_class = clf_model.predict(input_df)[0]

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Prédictions Actuelles")

col1, col2 = st.sidebar.columns(2)
col1.metric("Ventes prédites", f"{pred_sales:,.0f}")
col2.metric("ROI estimé", f"{pred_roi:.3f}")

col3, col4 = st.sidebar.columns(2)
col3.metric("Budget total", f"{total_budget:,.0f}")
if pred_class is not None:
    col4.metric("Classe", str(pred_class))


# ============================================================================
# MAIN TABS: Section Admin vs Section Utilisateur
# ============================================================================
section = st.radio(
    "Sélectionnez votre espace :",
    ["👤 Utilisateur", "🔧 Admin"],
    horizontal=True,
    label_visibility="collapsed"
)

if section == "👤 Utilisateur":
    tab_roi, tab_recommend, tab_analysis = st.tabs(
        [
            "🎮 Simulation ROI",
            "💼 Modèle Business Optimal",
            "📊 Analyse & Interprétabilité",
        ]
    )
    
    # ========== TAB 1: SIMULATION ROI ==========
    with tab_roi:
        st.subheader("Simulation de Campagne Marketing")
        
        st.info(
            "Ajustez les budgets des différents canaux marketing via la barre latérale pour "
            "simuler l'impact sur les ventes et le ROI de votre campagne."
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 📌 Paramètres Saisis")
            scenario_df = pd.DataFrame(
                {
                    "Canal": ["TV", "Radio", "Social Media", "**Total**"],
                    "Budget": [tv, radio, social, total_budget],
                    "% du total": [
                        f"{tv/total_budget*100:.1f}%" if total_budget > 0 else "0%",
                        f"{radio/total_budget*100:.1f}%" if total_budget > 0 else "0%",
                        f"{social/total_budget*100:.1f}%" if total_budget > 0 else "0%",
                        "100%"
                    ]
                }
            )
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        
        with col_b:
            st.markdown("### 💰 Prédictions du Scénario")
            col_x, col_y, col_z = st.columns(3)
            col_x.metric("Ventes prédites", f"{pred_sales:,.0f}")
            col_y.metric("ROI estimé", f"{pred_roi:.3f}")
            col_z.metric("Tier influenceur", influencer)
        
        st.markdown("---")
        
        # Graphiques de répartition
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 📊 Répartition Actuelle (Budget)")
            chart_data = pd.DataFrame(
                {"Canal": ["TV", "Radio", "Social Media"], "Budget": [tv, radio, social]}
            )
            st.bar_chart(chart_data.set_index("Canal"))
        
        with col_chart2:
            st.markdown("### 📊 Répartition Actuelle (% Ratio)")
            if total_budget > 0:
                chart_ratio = pd.DataFrame(
                    {
                        "Canal": ["TV", "Radio", "Social Media"],
                        "Ratio": [tv/total_budget, radio/total_budget, social/total_budget]
                    }
                )
                st.bar_chart(chart_ratio.set_index("Canal"))
            else:
                st.warning("Budget total = 0. Impossible de calculer les ratios.")
    
    # ========== TAB 2: MODÈLE BUSINESS OPTIMAL ==========
    with tab_recommend:
        st.subheader("🎯 Allocation Budgétaire Recommandée")
        
        st.markdown(
            "Cette section présente l'allocation budgétaire optimale calculée pour **maximiser "
            "les ventes** en fonction du budget total et du tier d'influenceur sélectionnés."
        )
        
        if total_budget <= 0:
            st.warning("⚠️ Veuillez saisir un budget total supérieur à 0 pour afficher les recommandations.")
        else:
            with st.spinner("⏳ Calcul de l'allocation optimale..."):
                rec = run_dynamic_optimization(total_budget, influencer)
            
            # === Résultats Clés ===
            st.markdown("### 📈 Résultats de l'Optimisation")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric(
                "💵 Ventes Prédites (Optimisées)",
                f"{rec['predicted_sales']:,.0f}",
                delta=f"{rec['predicted_sales'] - pred_sales:+,.0f} vs simulation actuelle"
            )
            col_res2.metric(
                "📊 ROI Optimisé",
                f"{rec['predicted_roi']:.3f}",
                delta=f"{rec['predicted_roi'] - pred_roi:+.3f} vs ROI actuel"
            )
            col_res3.metric(
                "🎯 Budget Total",
                f"{rec['total_budget']:,.0f}"
            )
            
            st.markdown("---")
            
            # === Tableau de Répartition ===
            st.markdown("### 💼 Répartition Budgétaire Recommandée")
            
            rec_budget_raw = list(rec["recommended_budget"].values())
            rec_df = pd.DataFrame(
                {
                    "Canal": list(rec["recommended_budget"].keys()),
                    "Budget Recommandé": rec_budget_raw,
                    "Ratio Recommandé": [
                        rec["recommended_ratio"][k]
                        for k in rec["recommended_budget"].keys()
                    ],
                    "Budget Actuel": [tv, radio, social],
                }
            )
            
            rec_df["Différence"] = rec_df["Budget Recommandé"] - rec_df["Budget Actuel"]
            rec_df_display = rec_df.copy()
            rec_df_display["Budget Recommandé"] = rec_df_display["Budget Recommandé"].apply(lambda x: f"{x:,.0f}")
            rec_df_display["Budget Actuel"] = rec_df_display["Budget Actuel"].apply(lambda x: f"{x:,.0f}")
            rec_df_display["Différence"] = rec_df_display["Différence"].apply(lambda x: f"{x:+,.0f}")
            rec_df_display["Ratio Recommandé"] = rec_df_display["Ratio Recommandé"].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(rec_df_display, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # === Visualisations ===
            st.markdown("### 📊 Visualisations")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("**Répartition Recommandée (Budget)**")
                rec_budget_chart = pd.DataFrame(
                    {
                        "Canal": list(rec["recommended_budget"].keys()),
                        "Budget": rec_budget_raw,
                    }
                )
                st.bar_chart(rec_budget_chart.set_index("Canal"))
            
            with col_viz2:
                st.markdown("**Répartition Recommandée (Ratio %)**")
                rec_ratio_chart = pd.DataFrame(
                    {
                        "Canal": list(rec["recommended_ratio"].keys()),
                        "Ratio": list(rec["recommended_ratio"].values()),
                    }
                )
                st.bar_chart(rec_ratio_chart.set_index("Canal"))
            
            st.markdown("---")
            
            # === Justification Business ===
            st.markdown("### 💡 Justification Business")
            
            best_channel_row = max(
                rec["recommended_ratio"].items(),
                key=lambda x: x[1]
            )
            best_channel = best_channel_row[0]
            best_ratio = best_channel_row[1]
            
            justification_text = f"""
            **Recommandation Clé :** Allouer **{best_ratio:.0%}** du budget marketing au canal **{best_channel}**.
            
            **Justification :**
            - Le modèle d'optimisation analyse l'impact marginal de chaque canal sur les ventes
            - Cette allocation maximise le retour sur investissement (ROI) prédit
            - Le ROI représente le ratio entre les ventes générées et le budget investi
            - En concentrant les efforts sur {best_channel}, vous maximisez le chiffre d'affaires 
              pour chaque euro dépensé
            
            **Impact Financier Estimé :**
            - **Ventes supplémentaires :** {rec['predicted_sales'] - pred_sales:+,.0f}
            - **Amélioration du ROI :** {(rec['predicted_roi'] - pred_roi) / pred_roi * 100:+.1f}%
            """
            
            st.success(justification_text)
            
            st.markdown("---")
            
            # === Analyse Marginal (Impact) ===
            if "marginal_impact" in rec:
                st.markdown("### 🔍 Analyse de Sensibilité (Impact Marginal)")
                
                st.markdown(
                    "Cette analyse montre comment les ventes évoluent si vous déplacez "
                    "une petite part du budget d'un canal vers un autre."
                )
                
                impact_rows = []
                for channel, values in rec["marginal_impact"].items():
                    increase = values["delta_sales_if_ratio_increases"]
                    decrease = values["delta_sales_if_ratio_decreases"]
                    
                    impact_rows.append({
                        "Canal": channel,
                        "Si la part augmente": (
                            f"+{increase:,.0f} ventes" if increase is not None else "Limite atteinte"
                        ),
                        "Si la part diminue": (
                            f"{decrease:,.0f} ventes" if decrease is not None else "Limite atteinte"
                        ),
                    })
                
                impact_df = pd.DataFrame(impact_rows)
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
                
                st.info(
                    "**Interprétation :** Les valeurs positives signifient qu'augmenter l'investissement "
                    "dans ce canal augmente les ventes. Les valeurs négatives signifient le contraire. "
                    "Plus la valeur est grande (en valeur absolue), plus l'impact est important."
                )
            
            st.markdown("---")
            
            # === Diagnostique de la Méthode ===
            if "search_diagnostics" in rec:
                diagnostics = rec["search_diagnostics"]
                st.markdown("### ⚙️ Méthodologie d'Optimisation")
                
                st.markdown(
                    f"""
                    **Nombre de scénarios testés :** {diagnostics['candidate_points_tested']} allocations budgétaires
                    
                    **Méthode utilisée :** {diagnostics['method']}
                    
                    **Meilleure solution trouvée par :** {diagnostics['best_solution_source']}
                    
                    Cette approche multi-start garantit de trouver un optimum robuste en testant 
                    plusieurs points de départ et en les raffinant avec des techniques d'optimisation 
                    numérique avancées.
                    """
                )
    
    # ========== TAB 3: ANALYSE & INTERPRÉTABILITÉ ==========
    with tab_analysis:
        st.subheader("📊 Analyse & Interprétabilité")
        
        st.markdown(
            "Explorez les facteurs qui influencent les prédictions du modèle et comprendre "
            "les relations entre les variables de marketing."
        )
        
        # === Corrélation avec les ventes ===
        st.markdown("### 📈 Influence des Canaux sur les Ventes")
        
        if CORR_PATH.exists():
            corr_df = pd.read_csv(CORR_PATH)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
            
            chart_df = corr_df[["Feature", "CorrelationWithSales"]].set_index("Feature")
            st.bar_chart(chart_df)
            
            if "AbsCorrelation" in corr_df.columns:
                top_feature = corr_df.sort_values("AbsCorrelation", ascending=False).iloc[0]
                
                st.success(
                    f"**Variable la plus influente :** {top_feature['Feature']} "
                    f"(corrélation : {top_feature['CorrelationWithSales']:.3f})\n\n"
                    f"Cela signifie que {top_feature['Feature']} est le canal qui impacte "
                    f"le plus les ventes. Suivre et optimiser ce canal devrait être une priorité "
                    f"stratégique dans votre stratégie marketing."
                )
        else:
            st.info("Le fichier d'analyse de corrélation n'est pas disponible.")
        
        st.markdown("---")
        
        # === Validation Croisée ===
        st.markdown("### ✅ Fiabilité des Prédictions")
        
        if CV_PATH.exists():
            cv_df = pd.read_csv(CV_PATH)
            st.dataframe(cv_df, use_container_width=True, hide_index=True)
            
            if "CVScore" in cv_df.columns:
                st.bar_chart(cv_df[["Model", "CVScore"]].set_index("Model"))
                
                avg_cv = cv_df["CVScore"].mean()
                st.info(
                    f"**Score moyen de validation croisée :** {avg_cv:.3f}\n\n"
                    f"La validation croisée teste le modèle sur plusieurs découpages des données. "
                    f"Un score élevé indique que le modèle généralise bien et que ses prédictions "
                    f"sont fiables sur de nouvelles données."
                )
        else:
            st.info("Le fichier de validation croisée n'est pas disponible.")
        
        st.markdown("---")
        
        # === Stratégie d'Entraînement ===
        st.markdown("### 🎓 Stratégie d'Entraînement")
        
        if POLICY_PATH.exists():
            with POLICY_PATH.open("r", encoding="utf-8") as f:
                policy = json.load(f)
            
            training_mode = safe_get(policy, "training_mode", "unknown")
            max_corr = safe_get(policy, "max_abs_numeric_correlation", None)
            
            st.markdown(
                f"**Mode d'entraînement :** {training_mode}\n\n"
                f"Le modèle a été sélectionné en fonction de l'analyse préalable des données. "
                f"Si la corrélation était forte, un modèle plus simple a été privilégié. "
                f"Sinon, un modèle plus complexe a été utilisé pour capturer les relations non-linéaires."
            )
            
            if "top_correlated_features" in policy:
                st.markdown("**Variables clés utilisées pour la modélisation :**")
                top_features_df = pd.DataFrame(policy["top_correlated_features"])
                st.dataframe(top_features_df, use_container_width=True, hide_index=True)
        else:
            st.info("Le fichier de politique d'entraînement n'est pas disponible.")

# ============================================================================
# SECTION ADMIN: Onglets Admin
# ============================================================================
else:  # section == "🔧 Admin"
    tab_perf, tab_compare = st.tabs(
        [
            "🏆 Performance des Modèles",
            "📈 Comparaison des Modèles",
        ]
    )
    
    # ========== TAB 1: PERFORMANCE DES MODÈLES ==========
    with tab_perf:
        st.subheader("🏆 Performance des Modèles")
        
        st.markdown(
            "Vue détaillée des performances des modèles de régression et classification "
            "sur le jeu de test et en validation croisée."
        )
        
        # === RÉGRESSION ===
        st.markdown("### 📊 Modèles de Régression")
        
        reg_metrics_df = None
        if REG_METRICS_PATH.exists():
            reg_metrics_df = pd.read_csv(REG_METRICS_PATH)
        
        if reg_metrics_df is not None and not reg_metrics_df.empty:
            st.dataframe(reg_metrics_df, use_container_width=True, hide_index=True)
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                if "R2" in reg_metrics_df.columns:
                    st.markdown("**R² (Coefficient de Détermination)**")
                    r2_chart = reg_metrics_df[["Model", "R2"]].set_index("Model")
                    st.bar_chart(r2_chart)
            
            with col_r2:
                if "RMSE" in reg_metrics_df.columns:
                    st.markdown("**RMSE (Erreur Quadratique Moyenne)**")
                    rmse_chart = reg_metrics_df[["Model", "RMSE"]].set_index("Model")
                    st.bar_chart(rmse_chart)
            
            if {"Model", "R2", "RMSE"}.issubset(reg_metrics_df.columns):
                best_reg_row = reg_metrics_df.sort_values(
                    by=["R2", "RMSE"],
                    ascending=[False, True],
                ).iloc[0]
                
                st.success(
                    f"**Meilleur modèle en régression :** {best_reg_row['Model']}\n\n"
                    f"- R² : {best_reg_row['R2']:.4f} (explique {best_reg_row['R2']*100:.1f}% de la variance)\n"
                    f"- RMSE : {best_reg_row['RMSE']:.4f} (erreur moyenne de prédiction)"
                )
        else:
            st.info("❌ Fichier de métriques de régression non disponible.")
        
        st.markdown("---")
        
        # === CLASSIFICATION ===
        st.markdown("### 🎯 Modèles de Classification")
        
        clf_metrics_df = None
        if CLF_METRICS_PATH.exists():
            clf_metrics_df = pd.read_csv(CLF_METRICS_PATH)
        
        if clf_metrics_df is not None and not clf_metrics_df.empty:
            st.dataframe(clf_metrics_df, use_container_width=True, hide_index=True)
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                if "F1_macro" in clf_metrics_df.columns:
                    st.markdown("**F1-Macro Score**")
                    f1_chart = clf_metrics_df[["Model", "F1_macro"]].set_index("Model")
                    st.bar_chart(f1_chart)
            
            with col_c2:
                if "Accuracy" in clf_metrics_df.columns:
                    st.markdown("**Accuracy**")
                    acc_chart = clf_metrics_df[["Model", "Accuracy"]].set_index("Model")
                    st.bar_chart(acc_chart)
            
            if {"Model", "Accuracy", "F1_macro"}.issubset(clf_metrics_df.columns):
                best_clf_row = clf_metrics_df.sort_values(
                    by=["F1_macro", "Accuracy"],
                    ascending=[False, False],
                ).iloc[0]
                
                st.success(
                    f"**Meilleur modèle en classification :** {best_clf_row['Model']}\n\n"
                    f"- Accuracy : {best_clf_row['Accuracy']:.4f} ({best_clf_row['Accuracy']*100:.1f}%)\n"
                    f"- F1-Macro : {best_clf_row['F1_macro']:.4f}"
                )
        else:
            st.info("❌ Fichier de métriques de classification non disponible.")
        
        st.markdown("---")
        
        # === RÉSUMÉ BEST MODELS ===
        st.markdown("### 🌟 Modèles Sélectionnés (Finaux)")
        
        if BEST_MODELS_PATH.exists():
            with BEST_MODELS_PATH.open("r", encoding="utf-8") as f:
                best_models = json.load(f)
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.success(
                    f"**🔴 Régression**\n\n"
                    f"Modèle : {best_models['best_regression_model']}\n"
                    f"R² Test : {best_models['best_regression_r2']:.4f}\n"
                    f"R² CV : {best_models['best_regression_cv_r2']:.4f}"
                )
            
            with col_m2:
                st.success(
                    f"**🟢 Classification**\n\n"
                    f"Modèle : {best_models['best_classification_model']}\n"
                    f"F1-Macro : {best_models['best_classification_f1_macro']:.4f}\n"
                    f"F1-Macro CV : {best_models['best_classification_cv_f1_macro']:.4f}"
                )
    
    # ========== TAB 2: COMPARAISON DES MODÈLES ==========
    with tab_compare:
        st.subheader("📈 Comparaison des Modèles")
        
        st.markdown(
            "Comparaison détaillée des performances de tous les modèles testés, "
            "triés par performance descendante."
        )
        
        reg_metrics_df = pd.read_csv(REG_METRICS_PATH) if REG_METRICS_PATH.exists() else None
        clf_metrics_df = pd.read_csv(CLF_METRICS_PATH) if CLF_METRICS_PATH.exists() else None
        
        if reg_metrics_df is None and clf_metrics_df is None:
            st.warning("❌ Aucun fichier de métriques trouvé.")
        else:
            # === RÉGRESSION ===
            if reg_metrics_df is not None and not reg_metrics_df.empty:
                st.markdown("### 📊 Classement - Régression")
                
                reg_ranked = reg_metrics_df.sort_values(
                    by=["R2", "RMSE"],
                    ascending=[False, True],
                ).copy()
                reg_ranked.insert(0, "Rang", range(1, len(reg_ranked) + 1))
                
                st.dataframe(
                    reg_ranked[["Rang", "Model", "R2", "RMSE"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.bar_chart(reg_ranked[["Model", "R2"]].set_index("Model"))
                
                st.markdown("---")
            
            # === CLASSIFICATION ===
            if clf_metrics_df is not None and not clf_metrics_df.empty:
                st.markdown("### 🎯 Classement - Classification")
                
                clf_ranked = clf_metrics_df.sort_values(
                    by=["F1_macro", "Accuracy"],
                    ascending=[False, False],
                ).copy()
                clf_ranked.insert(0, "Rang", range(1, len(clf_ranked) + 1))
                
                st.dataframe(
                    clf_ranked[["Rang", "Model", "F1_macro", "Accuracy"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                st.bar_chart(clf_ranked[["Model", "F1_macro"]].set_index("Model"))

