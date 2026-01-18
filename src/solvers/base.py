from abc import ABC, abstractmethod
import numpy as np

class BaseSolver(ABC):
    """Interface abstraite pour les solveurs d'allocation d'actifs."""
    
    @abstractmethod
    def solve(self) -> None:
        """Résout le problème d'optimisation."""
        pass
    
    @abstractmethod
    def get_policy(self, wealth: float, time: int) -> np.ndarray:
        """Retourne l'allocation optimale pour un état donné."""
        pass
