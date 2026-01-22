import os
import time
import pandas as pd
from src.core.config import MarketConfig, InvestmentConfig, SolverConfig
from src.core.model import InvestmentMDP
from src.solvers.dp_solver import DPSolver
try:
    from src.solvers.rl_solver import RLSolver, TORCH_AVAILABLE
except (ImportError, OSError):
    TORCH_AVAILABLE = False
from src.solvers.ortools_solver import ORToolsSolver
from src.simulation.engine import SimulationEngine
from src.utils.plotting import plot_results_professional
from src.analysis.stress_analysis import run_stress_analysis
from src.analysis.plot_robustness import load_data, create_robustness_comparison


def benchmark_solver(name, solver, mdp, sim_engine, n_trajs=200):
    """
    Évalue un solveur et retourne les résultats de simulation.
    """
    print(f"\n--- Évaluation du solveur : {name} ---")
    start_time = time.perf_counter()
    solver.solve()
    solve_time = time.perf_counter() - start_time
    print(f"Temps de résolution : {solve_time:.4f}s")
    
    results_df = sim_engine.run_simulation(solver, n_trajectories=n_trajs)
    results_df['solver'] = name
    
    final_wealths = results_df[results_df['time'] == mdp.i_cfg.horizon]['wealth']
    mean_wealth = final_wealths.mean()
    
    # Calcul du Sharpe Ratio (simplifié)
    returns = final_wealths / mdp.i_cfg.initial_wealth - 1
    sharpe = returns.mean() / (returns.std() + 1e-8)
    
    print(f"Richesse finale moyenne : {mean_wealth:.2f}")
    print(f"Sharpe Ratio : {sharpe:.2f}")
    
    return results_df, solve_time, mean_wealth, sharpe


def main():
    # =======================================================
    # ÉTAPE 1 : Simulation Normale
    # =======================================================
    print("=" * 70)
    print("ÉTAPE 1 : Simulation Normale - Exécution des 3 solveurs")
    print("=" * 70)
    
    # 1. Configuration
    market_cfg = MarketConfig()
    invest_cfg = InvestmentConfig()
    solver_cfg = SolverConfig()
    
    # 2. Initialisation du MDP
    mdp = InvestmentMDP(market_cfg, invest_cfg)
    sim_engine = SimulationEngine(mdp)
    
    # Définition du dossier output
    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Exécution des solveurs
    all_results = []
    summary_data = []
    
    solvers = [
        ("DP", DPSolver(mdp, solver_cfg)),
        ("OR-Tools", ORToolsSolver(mdp, solver_cfg))
    ]
    
    if TORCH_AVAILABLE:
        solvers.append(("RL", RLSolver(mdp, solver_cfg)))
    else:
        print("RL désactivé : Stable-Baselines3 ou PyTorch non disponible.")
    
    for name, solver in solvers:
        res_df, s_time, m_wealth, sharpe = benchmark_solver(name, solver, mdp, sim_engine)
        all_results.append(res_df)
        summary_data.append({
            "Solver": name,
            "Time (s)": s_time,
            "Mean Wealth": m_wealth,
            "Sharpe Ratio": sharpe
        })
    
    # Stockage des résultats dans un dictionnaire
    results = {
        'DP': None,
        'OR-Tools': None,
        'RL': None
    }
    for name, solver in solvers:
        for res_df in all_results:
            if not res_df.empty and res_df['solver'].iloc[0] == name:
                results[name] = res_df
                break
    
    # Sauvegarde des résultats en CSV
    comparison_df = pd.concat(all_results)
    comparison_df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    
    print(f"\n--- Résumé des résultats (simulation normale) ---")
    print(summary_df.to_string(index=False))
    
    # =======================================================
    # ÉTAPE 2 : Visualisation "Pro"
    # =======================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Génération des graphiques professionnels")
    print("=" * 70)
    
    plot_results_professional(
        comparison_df,
        mdp.i_cfg.horizon,
        output_dir=output_dir,
        target_wealth=mdp.i_cfg.target_wealth,
        inflation_rate=mdp.i_cfg.inflation_rate,
        life_events=mdp.i_cfg.life_events,
        event_names=mdp.i_cfg.event_names
    )
    print(f"Graphiques professionnels générés dans : {output_dir}")
    
    # =======================================================
    # ÉTAPE 3 : Stress Test
    # =======================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Stress Test - Simulation de crise")
    print("=" * 70)
    
    run_stress_analysis(output_dir=output_dir, n_trajectories=200)
    
    # =======================================================
    # ÉTAPE 4 : Conclusion (Robustesse)
    # =======================================================
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Conclusion - Graphique de robustesse")
    print("=" * 70)
    
    normal_df, stress_df = load_data()
    create_robustness_comparison(normal_df, stress_df)
    
    print("\n" + "=" * 70)
    print("TERMINÉ - Pipeline complet exécuté avec succès")
    print("=" * 70)
    print(f"\nTous les résultats sont dans le dossier : {output_dir}")
    print("Fichiers générés :")
    print("  - dp_wealth_prof.png, or-tools_wealth_prof.png, rl_wealth_prof.png")
    print("  - comparison_results.csv, summary.csv")
    print("  - stress_results.csv, stress_summary.csv")
    print("  - robustness_comparison.png")


if __name__ == "__main__":
    main()
