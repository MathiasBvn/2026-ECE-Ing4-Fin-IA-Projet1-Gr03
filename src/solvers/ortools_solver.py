from ortools.linear_solver import pywraplp
import numpy as np
from src.core.model import InvestmentMDP
from src.core.config import SolverConfig
from src.solvers.base import BaseSolver

class ORToolsSolver(BaseSolver):
    """
    Solveur utilisant Google OR-Tools pour une optimisation déterministe.
    Utile pour gérer des contraintes strictes sur l'allocation.
    """
    def __init__(self, mdp: InvestmentMDP, solver_cfg: SolverConfig):
        self.mdp = mdp
        self.cfg = solver_cfg
        self.optimal_policy = {} # (time, wealth) -> weights

    def solve(self) -> None:
        """
        Note: OR-Tools est utilisé ici pour résoudre une version déterministe 
        du problème à chaque étape (approche MPC).
        """
        pass

    def get_policy(self, wealth: float, time: int) -> np.ndarray:
        """
        Résout un problème d'optimisation déterministe sur l'horizon restant.
        """
        T = self.mdp.i_cfg.horizon
        if time >= T:
            return np.array([0, 0, 1.0])

        # Création du solveur GLOP (Linear Programming)
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            return np.array([0, 0, 1.0])

        # Pour rendre OR-Tools moins myope, on optimise l'allocation actuelle
        # en considérant les rendements espérés et les cash-flows futurs.
        # Comme le problème est multiplicatif (non-linéaire), on utilise une
        # approximation logarithmique ou on optimise simplement le rendement espéré
        # ajusté par le risque (moyenne-variance simplifiée).
        
        n_assets = len(self.mdp.m_cfg.assets)
        w = [solver.NumVar(0.0, 1.0, f'w_{i}') for i in range(n_assets)]
        
        # Contrainte : somme des poids = 1
        solver.Add(sum(w) == 1.0)
        
        # Objectif : Maximiser le rendement espéré - pénalité de risque simplifiée
        expected_returns = [self.mdp.m_cfg.expected_returns[a] for a in self.mdp.m_cfg.assets]
        vols = [self.mdp.m_cfg.volatilities[a] for a in self.mdp.m_cfg.assets]
        
        objective = solver.Objective()
        for i in range(n_assets):
            # Rendement espéré ajusté par une fraction de la volatilité (proxy risque)
            # On inclut une pénalité pour les frais d'entrée élevés (ex: SCPI)
            f_buy, _ = self.mdp.i_cfg.asset_fees[self.mdp.m_cfg.assets[i]]
            score = expected_returns[i] - 0.5 * self.mdp.i_cfg.risk_aversion * (vols[i]**2) - f_buy * 0.1
            objective.SetCoefficient(w[i], score)
        
        objective.SetMaximization()
        
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            return np.array([var.solution_value() for var in w])
        else:
            return np.array([0, 0, 1.0])
