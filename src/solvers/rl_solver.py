import gymnasium as gym
from gymnasium import spaces
import numpy as np
try:
    from stable_baselines3 import PPO
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False
from src.core.model import InvestmentMDP
from src.core.config import MarketConfig, InvestmentConfig, SolverConfig
from src.solvers.base import BaseSolver

class InvestmentEnv(gym.Env):
    """
    Environnement Gymnasium pour l'allocation d'actifs.
    """
    def __init__(self, mdp: InvestmentMDP):
        super(InvestmentEnv, self).__init__()
        self.mdp = mdp
        self.current_time = 0
        self.current_wealth = self.mdp.i_cfg.initial_wealth
        self.previous_weights = np.array([0, 0, 1.0], dtype=np.float32) # Start with Cash
        
        # Espace d'observation : [Richesse normalisée, Temps normalisé, Allocation précédente]
        # On normalise pour aider le RL
        self.observation_space = spaces.Box(
            low=0, high=10.0, shape=(8,), dtype=np.float32
        )
        
        # Espace d'action : 6 poids
        self.action_space = spaces.Box(
            low=0, high=1, shape=(6,), dtype=np.float32
        )

    def _get_obs(self):
        # Normalisation simple
        obs = np.concatenate([
            [self.current_wealth / self.mdp.i_cfg.initial_wealth],
            [self.current_time / self.mdp.i_cfg.horizon],
            self.previous_weights
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0
        self.current_wealth = self.mdp.i_cfg.initial_wealth
        # Initialisation par défaut : 100% Cash
        self.previous_weights = np.zeros(6, dtype=np.float32)
        self.previous_weights[2] = 1.0 # Cash
        return self._get_obs(), {}

    def step(self, action):
        # 1. Normalisation des poids (somme = 1, tous positifs)
        action = np.maximum(action, 0)
        weights = action / (np.sum(action) + 1e-8)
        if np.sum(action) == 0:
            weights = np.zeros_like(weights)
            weights[2] = 1.0 # Cash par défaut
        
        # 2. Transition avec frais
        returns_sample = self.mdp.generate_returns_sample(1)[0]
        next_wealth = self.mdp.transition(
            self.current_wealth, self.current_time, weights,
            returns_sample, self.previous_weights
        )
        
        self.current_wealth = next_wealth
        self.current_time += 1
        self.previous_weights = weights
        
        # 3. Récompense
        terminated = self.current_time >= self.mdp.i_cfg.horizon or self.current_wealth <= 0
        reward = 0.0
        
        if terminated:
            # Récompense finale basée sur l'utilité
            reward = self.mdp.utility_function(self.current_wealth)
            # On évite les valeurs trop extrêmes pour le RL
            reward = max(-100.0, reward) 
            
        return self._get_obs(), reward, terminated, False, {}

class RLSolver(BaseSolver):
    """
    Solveur utilisant le Reinforcement Learning (PPO).
    """
    def __init__(self, mdp: InvestmentMDP, solver_cfg: SolverConfig):
        self.mdp = mdp
        self.cfg = solver_cfg
        self.env = InvestmentEnv(mdp)
        self.model = None

    def solve(self) -> None:
        """Entraîne l'agent RL."""
        if not TORCH_AVAILABLE:
            print("RL désactivé : Stable-Baselines3 ou PyTorch non disponible sur ce système.")
            return
            
        print(f"Entraînement de l'agent PPO sur {self.cfg.total_timesteps} pas...")
        self.model = PPO("MlpPolicy", self.env, verbose=0,
                         learning_rate=self.cfg.learning_rate)
        self.model.learn(total_timesteps=self.cfg.total_timesteps)
        print("Entraînement terminé.")

    def get_policy(self, wealth: float, time: int) -> np.ndarray:
        """Retourne l'allocation prédite par l'agent."""
        if self.model is None:
            return np.array([0, 0, 1.0]) # Cash par défaut
            
        # Note: En mode prédiction, on utilise une allocation équilibrée par défaut pour l'obs
        obs = np.concatenate([
            [wealth / self.mdp.i_cfg.initial_wealth],
            [time / self.mdp.i_cfg.horizon],
            np.full(6, 1/6)
        ]).astype(np.float32)
        
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.maximum(action, 0)
        weights = action / (np.sum(action) + 1e-8)
        if np.sum(action) == 0:
            weights = np.zeros_like(weights)
            weights[2] = 1.0
        return weights
