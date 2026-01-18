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

def benchmark_solver(name, solver, mdp, sim_engine, n_trajs=200):
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
    # 1. Configuration
    market_cfg = MarketConfig()
    invest_cfg = InvestmentConfig(horizon=10)
    solver_cfg = SolverConfig(wealth_grid_size=30, total_timesteps=20000)
    
    # 2. Initialisation du MDP
    mdp = InvestmentMDP(market_cfg, invest_cfg)
    sim_engine = SimulationEngine(mdp)
    
    if not os.path.exists("output"):
        os.makedirs("output")
        
    # 3. Benchmarks
    all_results = []
    solvers = [
        ("DP", DPSolver(mdp, solver_cfg)),
        ("OR-Tools", ORToolsSolver(mdp, solver_cfg))
    ]
    
    if TORCH_AVAILABLE:
        solvers.append(("RL", RLSolver(mdp, solver_cfg)))
    else:
        print("Note : Le solveur RL sera ignoré car PyTorch n'est pas opérationnel sur ce système.")
    
    summary_data = []
    for name, solver in solvers:
        res_df, s_time, m_wealth, sharpe = benchmark_solver(name, solver, mdp, sim_engine)
        all_results.append(res_df)
        summary_data.append({
            "Solver": name,
            "Time (s)": s_time,
            "Mean Wealth": m_wealth,
            "Sharpe Ratio": sharpe
        })
    
    # 4. Comparaison globale et Graphiques Professionnels
    print("\n--- Génération des graphiques professionnels ---")
    comparison_df = pd.concat(all_results)
    plot_results_professional(comparison_df, mdp.i_cfg.horizon)
    
    # Sauvegarde des données pour analyse ultérieure
    comparison_df.to_csv("output/comparison_results.csv", index=False)
    
    # Affichage du tableau récapitulatif
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Tableau Récapitulatif ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("output/summary.csv", index=False)
    
    print("\nTerminé. Tous les résultats sont dans le dossier 'output/'.")

if __name__ == "__main__":
    main()
