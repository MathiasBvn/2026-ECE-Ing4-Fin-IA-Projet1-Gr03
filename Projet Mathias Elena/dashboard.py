import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import time

from src.core.config import MarketConfig, InvestmentConfig, SolverConfig
from src.core.model import InvestmentMDP
from src.solvers.dp_solver import DPSolver
from src.solvers.ortools_solver import ORToolsSolver
from src.simulation.engine import SimulationEngine
from src.utils.plotting import ASSET_COLORS

# Configuration de la page
st.set_page_config(page_title="Wealth Planner AI", page_icon="💰", layout="wide")

st.title("💰 Wealth Planner AI : Optimisation d'Investissement")
st.markdown("""
Cette application utilise la **Programmation Dynamique** et l'**Optimisation Linéaire** pour concevoir votre stratégie d'investissement optimale sur mesure.
""")

# --- SIDEBAR : PARAMÈTRES GÉNÉRAUX ---
st.sidebar.header("👤 Profil & Paramètres")

initial_wealth = st.sidebar.number_input("Capital Initial (k€)", min_value=0.0, value=200.0, step=10.0)
monthly_savings = st.sidebar.number_input("Épargne Mensuelle (k€)", min_value=0.0, value=1.0, step=0.1)
current_age = st.sidebar.slider("Âge Actuel", 18, 80, 35)
retirement_age = st.sidebar.slider("Âge de Retraite (Horizon)", current_age + 5, 100, 65)
horizon = retirement_age - current_age

risk_profile = st.sidebar.selectbox(
    "Profil de Risque",
    ["Prudent", "Équilibré", "Dynamique"],
    index=1
)

# Ajustement de l'aversion au risque selon le profil
risk_aversion_map = {
    "Prudent": 4.0,
    "Équilibré": 2.0,
    "Dynamique": 1.0
}
risk_aversion = risk_aversion_map[risk_profile]

# --- ZONE PRINCIPALE : PLAN DE VIE ---
st.header("📅 Votre Plan de Vie")
st.subheader("Événements de cash-flow (Sorties de capital)")

# Initialisation des événements par défaut
if 'events_df' not in st.session_state:
    st.session_state.events_df = pd.DataFrame([
        {"Nom": "Achat Voiture", "Année": 5, "Montant (k€)": 20.0},
        {"Nom": "Apport Immobilier", "Année": 12, "Montant (k€)": 80.0},
        {"Nom": "Études Enfants", "Année": 20, "Montant (k€)": 30.0}
    ])

edited_events = st.data_editor(
    st.session_state.events_df,
    num_rows="dynamic",
    column_config={
        "Année": st.column_config.NumberColumn(min_value=1, max_value=horizon),
        "Montant (k€)": st.column_config.NumberColumn(min_value=0.0)
    },
    key="events_editor"
)

# --- CALCULS ---
if st.button("🚀 Calculer la Stratégie Optimale", type="primary"):
    with st.spinner("Calcul de la stratégie de Bellman en cours..."):
        # 1. Préparation de la configuration
        market_cfg = MarketConfig()
        
        # Transformation des événements
        life_events = {}
        event_names = {}
        for _, row in edited_events.iterrows():
            year = int(row["Année"])
            amount = float(row["Montant (k€)"])
            life_events[year] = life_events.get(year, 0) + amount
            event_names[year] = row["Nom"]

        invest_cfg = InvestmentConfig(
            initial_wealth=initial_wealth,
            horizon=horizon,
            monthly_savings=monthly_savings,
            life_events=life_events,
            event_names=event_names,
            risk_aversion=risk_aversion
        )
        
        solver_cfg = SolverConfig(
            wealth_grid_size=40,
            max_wealth=initial_wealth * 5 + monthly_savings * 12 * horizon
        )
        
        # 2. Initialisation du MDP et Moteur
        mdp = InvestmentMDP(market_cfg, invest_cfg)
        sim_engine = SimulationEngine(mdp)
        
        # 3. Résolution DP
        dp_solver = DPSolver(mdp, solver_cfg)
        dp_solver.solve()
        
        # 4. Simulation
        results_dp = sim_engine.run_simulation(dp_solver, n_trajectories=100)
        results_dp['solver'] = 'DP'
        
        # 5. Résolution OR-Tools (pour comparaison)
        ort_solver = ORToolsSolver(mdp, solver_cfg)
        results_ort = sim_engine.run_simulation(ort_solver, n_trajectories=100)
        results_ort['solver'] = 'OR-Tools'
        
        all_results = pd.concat([results_dp, results_ort])
        
        # --- AFFICHAGE DES RÉSULTATS ---
        st.success("Calcul terminé !")
        
        # KPIs
        final_wealth_dp = results_dp[results_dp['time'] == horizon]['wealth']
        mean_final_wealth = final_wealth_dp.mean()
        prob_success = (final_wealth_dp > 0).mean() * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Richesse Finale Moyenne", f"{mean_final_wealth:.1f} k€")
        col2.metric("Probabilité de Succès", f"{prob_success:.1f} %")
        col3.metric("Âge de Fin", f"{retirement_age} ans")
        
        # --- GRAPHIQUES INTERACTIFS (PLOTLY) ---
        tab1, tab2, tab3 = st.tabs(["📈 Richesse", "📊 Allocation", "⚖️ Comparaison"])
        
        with tab1:
            st.subheader("Convergence de la Richesse (DP)")
            stats = results_dp.groupby('time')['wealth'].agg(['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)])
            stats.columns = ['mean', 'std', 'p5', 'p95']
            
            fig_wealth = go.Figure()
            
            # Zone d'ombre p5-p95
            fig_wealth.add_trace(go.Scatter(
                x=stats.index, y=stats['p95'],
                mode='lines', line=dict(width=0),
                showlegend=False, name='p95'
            ))
            fig_wealth.add_trace(go.Scatter(
                x=stats.index, y=stats['p5'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(31, 119, 180, 0.1)',
                showlegend=False, name='p5'
            ))
            
            # Ligne moyenne
            fig_wealth.add_trace(go.Scatter(
                x=stats.index, y=stats['mean'],
                mode='lines', line=dict(color='#1f77b4', width=4),
                name='Richesse Moyenne'
            ))
            
            # Événements
            for year, amount in life_events.items():
                fig_wealth.add_vline(x=year, line_dash="dash", line_color="red", opacity=0.5)
                fig_wealth.add_annotation(x=year, y=mean_final_wealth, text=f"-{amount}k€", showarrow=True, arrowhead=1)

            fig_wealth.update_layout(
                title="Évolution de la Richesse au cours du temps",
                xaxis_title="Années",
                yaxis_title="Valeur (k€)",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig_wealth, use_container_width=True)

        with tab2:
            st.subheader("Composition du Portefeuille (DP)")
            alloc_cols = [c for c in results_dp.columns if c.startswith('alloc_')]
            avg_alloc = results_dp.groupby('time')[alloc_cols].mean()
            
            fig_alloc = go.Figure()
            for col in alloc_cols:
                asset_name = col.replace('alloc_', '').capitalize()
                fig_alloc.add_trace(go.Scatter(
                    x=avg_alloc.index, y=avg_alloc[col],
                    mode='lines',
                    stackgroup='one',
                    name=asset_name,
                    line=dict(color=ASSET_COLORS.get(asset_name, None))
                ))
            
            fig_alloc.update_layout(
                title="Allocation d'Actifs Optimale",
                xaxis_title="Années",
                yaxis_title="Poids (%)",
                yaxis=dict(range=[0, 1]),
                template="plotly_white"
            )
            st.plotly_chart(fig_alloc, use_container_width=True)
            
        with tab3:
            st.subheader("Comparaison DP vs OR-Tools")
            fig_comp = px.violin(
                all_results[all_results['time'] == horizon],
                x="solver", y="wealth", color="solver",
                box=True, points="all",
                title="Distribution de la Richesse Finale"
            )
            st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("Configurez vos paramètres dans la barre latérale et cliquez sur 'Calculer' pour voir votre stratégie.")

# Footer
st.markdown("---")
st.caption("Développé par Roo Code Expert - 2026")
