# Plan de Projet : Planification d'Investissement Multi-Périodes

Ce document détaille la stratégie de conception et d'implémentation pour le projet d'optimisation d'allocation d'actifs.

## 1. Analyse du Problème

### Modélisation MDP (Markov Decision Process)
- **État ($S_t$)** : 
  - $W_t$ : Richesse actuelle (continue, à discrétiser pour la DP).
  - $t$ : Période actuelle ($t \in \{0, \dots, T\}$).
  - $F_t$ : Facteurs de marché (ex: rendements passés ou indicateurs de régime).
- **Action ($A_t$)** : 
  - Vecteur de poids $\omega_t = [\omega_{actions}, \omega_{obligations}, \omega_{cash}]$ tel que $\sum \omega_i = 1$ et $\omega_i \ge 0$.
- **Transition** :
  - $W_{t+1} = (W_t - C_t) \cdot \sum_{i} \omega_{i,t} (1 + r_{i, t+1})$
  - $C_t$ : Cash-flow sortant déterministe (événements de vie).
  - $r_{i, t+1}$ : Rendements stochastiques des actifs.
- **Objectif** : Maximiser l'utilité de la richesse finale $E[U(W_T)]$ (ex: utilité CRRA ou simplement $W_T$ sous contrainte de risque).

## 2. Architecture du Projet

```text
.
├── PLAN.md                 # Feuille de route
├── requirements.txt        # Dépendances
├── main.py                 # Point d'entrée pour les comparaisons
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── model.py        # Définition mathématique du MDP
│   │   └── config.py       # Paramètres (T, rendements, cash-flows)
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py         # Interface abstraite pour les solveurs
│   │   ├── dp_solver.py    # Programmation Dynamique (Bellman)
│   │   ├── ortools_solver.py # Optimisation via OR-Tools
│   │   └── rl_solver.py    # Environnement Gymnasium + SB3
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── engine.py       # Simulateur Monte Carlo
│   └── utils/
│       ├── __init__.py
│       └── plotting.py     # Visualisation et rapports
└── tests/                  # Tests unitaires
```

## 3. Étapes de Développement

### Phase 2 : Configuration (Immédiat après validation)
1. Création du `venv`.
2. Installation des dépendances (`numpy`, `scipy`, `pandas`, `ortools`, `gymnasium`, `stable-baselines3`, `matplotlib`, `seaborn`).
3. Initialisation de l'arborescence.

### Phase 3 : Développement Core & DP
1. Implémenter `src/core/model.py` avec la logique de transition et de récompense.
2. Développer `src/solvers/dp_solver.py` :
   - Discrétisation de l'espace des richesses.
   - Induction arrière (Backward Induction) pour calculer la valeur de Bellman.
3. Créer `src/simulation/engine.py` pour valider la politique sur des trajectoires non vues.

### Phase 4 : Extensions (RL & OR-Tools)
1. Créer un wrapper Gymnasium dans `src/solvers/rl_solver.py`.
2. Entraîner un agent PPO.
3. Implémenter une version déterministe ou à scénarios avec `ortools_solver.py`.

### Phase 5 : Comparaison et Analyse
1. Script de benchmark comparant :
   - Temps d'exécution.
   - Richesse finale moyenne.
   - Ratio de Sharpe / Drawdown maximum.
2. Génération des graphiques de frontières efficientes et de trajectoires.
   
3. En détail : Il s'agit de tester vos trois solveurs (dp_solver.py, ortools_solver.py, rl_solver.py) dans des conditions de marché dégradées (inflation imprévue, krach boursier).
À quoi ça sert : En programmation dynamique et RL, un modèle peut être excellent en situation "normale" mais s'effondrer à la moindre crise. Cette étape prouve la fiabilité réelle du conseil en investissement.

## 4. Considérations Techniques
- Utilisation de `numba` ou `numpy` vectorisé pour accélérer la DP.
- Gestion des contraintes de "ruine" (si $W_t < 0$).
- Documentation des fonctions avec références mathématiques.
