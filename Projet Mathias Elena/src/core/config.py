from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class MarketConfig:
    """Configuration des paramètres de marché."""
    assets: List[str] = field(default_factory=lambda: [
        "Actions", "Obligations", "Cash", "Or", "Crypto", "SCPI"
    ])
    expected_returns: Dict[str, float] = field(default_factory=lambda: {
        "Actions": 0.08,
        "Obligations": 0.03,
        "Cash": 0.01,
        "Or": 0.04,
        "Crypto": 0.15,
        "SCPI": 0.045
    })
    volatilities: Dict[str, float] = field(default_factory=lambda: {
        "Actions": 0.15,
        "Obligations": 0.05,
        "Cash": 0.005,
        "Or": 0.12,
        "Crypto": 0.60,
        "SCPI": 0.05
    })
    # Matrice 6x6
    correlations: List[List[float]] = field(default_factory=lambda: [
        [1.0, 0.2, 0.0, 0.1, 0.3, 0.4],  # Actions
        [0.2, 1.0, 0.0, 0.2, -0.1, 0.1], # Obligations
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Cash
        [0.1, 0.2, 0.0, 1.0, 0.1, 0.0],  # Or
        [0.3, -0.1, 0.0, 0.1, 1.0, 0.1], # Crypto
        [0.4, 0.1, 0.0, 0.0, 0.1, 1.0]   # SCPI
    ])

@dataclass
class InvestmentConfig:
    """Configuration des paramètres d'investissement."""
    initial_wealth: float = 200.0  # 200k €
    horizon: int = 30
    monthly_savings: float = 0.0 # Épargne mensuelle en k€
    rebalancing_frequency: str = "annual"
    inflation_rate: float = 0.02
    target_wealth: float = 600.0  # Objectif de richesse nominale
    
    # Événements de vie : {année: montant_sortie}
    life_events: Dict[int, float] = field(default_factory=lambda: {
        5: 15.0,    # Achat voiture
        12: 60.0,   # Apport Immobilier
        20: 10.0,   # Études enfant an 1
        21: 10.0,   # Études enfant an 2
        22: 10.0    # Études enfant an 3
    })
    
    # Noms des événements pour les annotations
    event_names: Dict[int, str] = field(default_factory=lambda: {
        5: "Voiture",
        12: "Apport Immo",
        20: "Études",
        21: "Études",
        22: "Études"
    })
    
    # Contraintes de liquidité et pénalités
    illiquid_assets: List[str] = field(default_factory=lambda: ["SCPI"])
    fire_sale_penalty: float = 0.15  # 15% de pénalité si vente forcée d'actifs illiquides
    
    # Paramètres de la fonction d'utilité (CRRA)
    risk_aversion: float = 2.0
    
    # Frais de transaction spécifiques par actif
    asset_fees: Dict[str, tuple] = field(default_factory=lambda: {
        "Actions": (0.001, 0.001),
        "Obligations": (0.0005, 0.0005),
        "Cash": (0.0, 0.0),
        "Or": (0.002, 0.002),
        "Crypto": (0.005, 0.005),
        "SCPI": (0.10, 0.05)
    })

@dataclass
class SolverConfig:
    """Configuration des paramètres des solveurs."""
    wealth_grid_size: int = 50
    min_wealth: float = 0.0
    max_wealth: float = 1200.0
    
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
