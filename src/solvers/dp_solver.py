import numpy as np
from scipy.interpolate import interp1d
from src.core.model import InvestmentMDP
from src.core.config import SolverConfig
from src.solvers.base import BaseSolver
from typing import List

class DPSolver(BaseSolver):
    """
    Solveur par Programmation Dynamique (Induction Arrière).
    
    Équation de Bellman :
    V_t(W) = max_{a} E[ V_{t+1}( (W - C_t) * sum(w_i(1+r_i)) ) ]
    """
    
    def __init__(self, mdp: InvestmentMDP, solver_cfg: SolverConfig):
        self.mdp = mdp
        self.cfg = solver_cfg
        
        # Grille de richesse
        self.wealth_grid = np.linspace(self.cfg.min_wealth, self.cfg.max_wealth, self.cfg.wealth_grid_size)
        
        # Grille d'actions (simplifiée : allocations discrètes)
        # On génère des combinaisons de poids qui somment à 1
        self.action_grid = self._generate_action_grid(step=0.1)
        
        # Tableaux pour stocker la fonction de valeur et la politique
        # V[temps, index_richesse]
        self.V = np.zeros((self.mdp.i_cfg.horizon + 1, self.cfg.wealth_grid_size))
        # Policy[temps, index_richesse] -> index_action
        self.policy = np.zeros((self.mdp.i_cfg.horizon, self.cfg.wealth_grid_size), dtype=int)

    def _generate_action_grid(self, step: float) -> np.ndarray:
        """
        Génère une grille d'allocations simplifiée par classes de risque
        pour éviter l'explosion combinatoire (6 actifs).
        Classes :
        1. Sécurisé (Cash, Obligations)
        2. Modéré (Or, SCPI)
        3. Dynamique (Actions, Crypto)
        """
        actions = []
        for w_safe in np.arange(0, 1 + step, step):
            for w_mod in np.arange(0, 1 - w_safe + step, step):
                w_dyn = 1.0 - w_safe - w_mod
                if w_dyn >= -1e-7:
                    # Répartition 50/50 au sein de chaque classe
                    # Ordre : Actions, Obligations, Cash, Or, Crypto, SCPI
                    weights = np.zeros(6)
                    weights[0] = w_dyn * 0.8  # Actions (80% du dynamique)
                    weights[4] = w_dyn * 0.2  # Crypto (20% du dynamique)
                    weights[1] = w_safe * 0.7 # Obligations
                    weights[2] = w_safe * 0.3 # Cash
                    weights[3] = w_mod * 0.4  # Or
                    weights[5] = w_mod * 0.6  # SCPI
                    actions.append(weights)
        return np.array(actions)

    def solve(self) -> None:
        """Exécute l'induction arrière."""
        T = self.mdp.i_cfg.horizon
        
        # Condition terminale : V_T(W) = U(W)
        for i, w in enumerate(self.wealth_grid):
            self.V[T, i] = self.mdp.utility_function(w)
            
        # Échantillons de rendements pour l'approximation de l'espérance (Monte Carlo local)
        n_samples = 50
        returns_samples = self.mdp.generate_returns_sample(n_samples)
        
        # Induction arrière
        for t in range(T - 1, -1, -1):
            print(f"Résolution période {t}...")
            # Interpolateur pour V_{t+1}
            v_next_interp = interp1d(self.wealth_grid, self.V[t+1, :], 
                                     kind='linear', fill_value="extrapolate")
            
            for i, w in enumerate(self.wealth_grid):
                best_val = -1e20
                best_action_idx = 0
                
                cash_flow = self.mdp.i_cfg.life_events.get(t, 0.0)
                available_wealth = max(0.0, w - cash_flow)
                
                if available_wealth <= 0:
                    self.V[t, i] = self.mdp.utility_function(0)
                    self.policy[t, i] = 0
                    continue

                # Tester chaque action
                for a_idx, action in enumerate(self.action_grid):
                    # Calcul de l'espérance de V_{t+1}
                    # W_{t+1} pour chaque échantillon de rendement
                    next_wealths = available_wealth * np.sum(action * (1 + returns_samples), axis=1)
                    # On s'assure que la richesse ne descend pas sous le min de la grille
                    next_wealths = np.maximum(self.cfg.min_wealth, next_wealths)
                    
                    expected_val = np.mean(v_next_interp(next_wealths))
                    
                    if expected_val > best_val:
                        best_val = expected_val
                        best_action_idx = a_idx
                
                self.V[t, i] = best_val
                self.policy[t, i] = best_action_idx

    def get_policy(self, wealth: float, time: int) -> np.ndarray:
        """Retourne l'allocation optimale par interpolation de la politique apprise."""
        # Si on est à la fin de l'horizon, on garde l'allocation de la dernière période calculée
        effective_time = min(time, self.mdp.i_cfg.horizon - 1)
        
        if effective_time < 0:
            return self.action_grid[0]
            
        # Trouver l'action la plus proche dans la grille de richesse
        idx = np.abs(self.wealth_grid - wealth).argmin()
        action_idx = self.policy[effective_time, idx]
        return self.action_grid[action_idx]
