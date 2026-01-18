import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.core.model import InvestmentMDP
from src.solvers.base import BaseSolver

class SimulationEngine:
    """
    Moteur de simulation Monte Carlo pour évaluer les politiques d'investissement.
    """
    
    def __init__(self, mdp: InvestmentMDP):
        self.mdp = mdp

    def run_simulation(self, solver: BaseSolver, n_trajectories: int = 100) -> pd.DataFrame:
        """
        Exécute n simulations et retourne les résultats sous forme de DataFrame.
        """
        results = []
        T = self.mdp.i_cfg.horizon
        
        for i in range(n_trajectories):
            wealth = self.mdp.i_cfg.initial_wealth
            wealth_history = [wealth]
            allocation_history = []
            previous_weights = None
            
            for t in range(T):
                # Obtenir l'allocation du solveur
                weights = solver.get_policy(wealth, t)
                allocation_history.append(weights)
                
                # Générer un rendement aléatoire pour cette étape
                returns_sample = self.mdp.generate_returns_sample(1)[0]
                
                # Transition avec frais de transaction
                wealth = self.mdp.transition(wealth, t, weights, returns_sample, previous_weights)
                wealth_history.append(wealth)
                previous_weights = weights
                
                if wealth <= 0:
                    # Faillite : on remplit le reste avec 0
                    for _ in range(t + 1, T):
                        wealth_history.append(0.0)
                        allocation_history.append(np.zeros_like(weights))
                    break
            
            # Stockage des résultats de la trajectoire
            for t, (w, a) in enumerate(zip(wealth_history, allocation_history + [np.nan])):
                res_dict = {
                    "trajectory": i,
                    "time": t,
                    "wealth": w
                }
                if isinstance(a, np.ndarray):
                    for j, asset in enumerate(self.mdp.m_cfg.assets):
                        res_dict[f"alloc_{asset.lower()}"] = a[j]
                else:
                    for asset in self.mdp.m_cfg.assets:
                        res_dict[f"alloc_{asset.lower()}"] = np.nan
                results.append(res_dict)
                
        return pd.DataFrame(results)

    def calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule des statistiques agrégées par période."""
        stats = df.groupby("time")["wealth"].agg([
            "mean", "std", "min", "max",
            lambda x: np.percentile(x, 5),
            lambda x: np.percentile(x, 50),
            lambda x: np.percentile(x, 95)
        ]).rename(columns={
            "<lambda_0>": "p5",
            "<lambda_1>": "median",
            "<lambda_2>": "p95"
        })
        return stats
