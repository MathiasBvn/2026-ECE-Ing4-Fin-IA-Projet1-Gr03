import numpy as np
from typing import Tuple, Dict, List
from src.core.config import MarketConfig, InvestmentConfig

class InvestmentMDP:
    """
    Modélisation du Markov Decision Process pour la planification d'investissement.
    """
    
    def __init__(self, market_cfg: MarketConfig, invest_cfg: InvestmentConfig):
        self.m_cfg = market_cfg
        self.i_cfg = invest_cfg
        
        # Préparation des paramètres de rendement
        self.means = np.array([self.m_cfg.expected_returns[a] for a in self.m_cfg.assets])
        self.vols = np.array([self.m_cfg.volatilities[a] for a in self.m_cfg.assets])
        self.corr = np.array(self.m_cfg.correlations)
        
        # Matrice de covariance
        self.cov = np.diag(self.vols) @ self.corr @ np.diag(self.vols)
        
    def utility_function(self, wealth: float) -> float:
        """Fonction d'utilité CRRA."""
        gamma = self.i_cfg.risk_aversion
        if wealth <= 0:
            return -1e10
        
        if gamma == 1.0:
            return np.log(wealth)
        else:
            return (np.power(wealth, 1 - gamma) - 1) / (1 - gamma)

    def transition(self, wealth: float, time: int, weights: np.ndarray,
                   returns_sample: np.ndarray, previous_weights: np.ndarray = None) -> float:
        """
        Calcule la richesse à t+1 en incluant les frais, l'épargne et les pénalités de liquidité.
        """
        # Ajout de l'épargne annuelle
        wealth += self.i_cfg.monthly_savings * 12
        
        cash_flow = self.i_cfg.life_events.get(time, 0.0)
        
        # Gestion de la liquidité et pénalité de vente forcée
        # Si on n'a pas de previous_weights (t=0), on suppose une allocation initiale en Cash
        if previous_weights is None:
            # Au temps 0, on suppose que la richesse initiale est 100% Cash pour couvrir les frais/CF
            previous_weights = np.zeros(len(self.m_cfg.assets))
            cash_idx = self.m_cfg.assets.index("Cash")
            previous_weights[cash_idx] = 1.0

        # Montant de Cash disponible avant rééquilibrage
        cash_idx = self.m_cfg.assets.index("Cash")
        available_cash = wealth * previous_weights[cash_idx]
        
        penalty = 0.0
        if cash_flow > available_cash:
            # On doit vendre d'autres actifs pour couvrir le manque de cash
            shortfall = cash_flow - available_cash
            
            # On vérifie si on doit toucher aux actifs illiquides
            # Pour simplifier : on vend d'abord les actifs liquides (tout sauf SCPI)
            liquid_assets_indices = [i for i, a in enumerate(self.m_cfg.assets) 
                                     if a not in self.i_cfg.illiquid_assets and a != "Cash"]
            
            liquid_value = sum(wealth * previous_weights[i] for i in liquid_assets_indices)
            
            if shortfall > liquid_value:
                # On doit vendre de l'illiquide
                illiquid_to_sell = shortfall - liquid_value
                penalty = illiquid_to_sell * self.i_cfg.fire_sale_penalty
        
        available_wealth = max(0.0, wealth - cash_flow - penalty)
        
        if available_wealth <= 0:
            return 0.0
            
        # Calcul des frais de transaction
        fees = 0.0
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
        return np.random.multivariate_normal(self.means, self.cov, n_samples)
