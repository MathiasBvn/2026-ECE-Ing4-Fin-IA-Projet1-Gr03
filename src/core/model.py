import numpy as np
from typing import Tuple, Dict, List
from src.core.config import MarketConfig, InvestmentConfig

class InvestmentMDP:
    """
    Modélisation du Markov Decision Process pour la planification d'investissement.
    
    Équation de transition de la richesse :
    W_{t+1} = (W_t - C_t) * sum(w_i * (1 + r_i))
    
    où :
    - W_t est la richesse au temps t
    - C_t est le cash-flow sortant au temps t
    - w_i sont les poids d'allocation
    - r_i sont les rendements stochastiques des actifs
    """
    
    def __init__(self, market_cfg: MarketConfig, invest_cfg: InvestmentConfig):
        self.m_cfg = market_cfg
        self.i_cfg = invest_cfg
        
        # Préparation des paramètres de rendement
        self.means = np.array([self.m_cfg.expected_returns[a] for a in self.m_cfg.assets])
        self.vols = np.array([self.m_cfg.volatilities[a] for a in self.m_cfg.assets])
        self.corr = np.array(self.m_cfg.correlations)
        
        # Matrice de covariance : Sigma = diag(vols) * Corr * diag(vols)
        self.cov = np.diag(self.vols) @ self.corr @ np.diag(self.vols)
        
    def utility_function(self, wealth: float) -> float:
        """
        Fonction d'utilité CRRA (Constant Relative Risk Aversion).
        U(W) = (W^(1-gamma) - 1) / (1-gamma) si gamma != 1 sinon log(W)
        """
        gamma = self.i_cfg.risk_aversion
        if wealth <= 0:
            return -1e10  # Pénalité forte pour la faillite
        
        if gamma == 1.0:
            return np.log(wealth)
        else:
            return (np.power(wealth, 1 - gamma) - 1) / (1 - gamma)

    def transition(self, wealth: float, time: int, weights: np.ndarray,
                   returns_sample: np.ndarray, previous_weights: np.ndarray = None) -> float:
        """
        Calcule la richesse à t+1 en incluant les frais de transaction.
        """
        cash_flow = self.i_cfg.life_events.get(time, 0.0)
        available_wealth = max(0.0, wealth - cash_flow)
        
        if available_wealth <= 0:
            return 0.0
            
        # Calcul des frais de transaction spécifiques par actif
        fees = 0.0
        if previous_weights is not None:
            for i, asset in enumerate(self.m_cfg.assets):
                diff = weights[i] - previous_weights[i]
                f_buy, f_sell = self.i_cfg.asset_fees[asset]
                if diff > 0: # Achat
                    fees += available_wealth * diff * f_buy
                elif diff < 0: # Vente
                    fees += available_wealth * abs(diff) * f_sell
            
        available_wealth -= fees
        
        if available_wealth <= 0:
            return 0.0
            
        portfolio_return = np.sum(weights * (1 + returns_sample))
        next_wealth = available_wealth * portfolio_return
        
        return next_wealth

    def generate_returns_sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Génère des échantillons de rendements log-normaux.
        """
        # Pour simplifier, on utilise une distribution normale pour les rendements arithmétiques
        # Dans un cadre plus rigoureux, on utiliserait des rendements log-normaux.
        return np.random.multivariate_normal(self.means, self.cov, n_samples)
