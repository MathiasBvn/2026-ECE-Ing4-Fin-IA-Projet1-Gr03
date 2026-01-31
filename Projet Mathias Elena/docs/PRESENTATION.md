# 💰 Wealth Planner AI
## Optimisation d'Investissement Multi-Périodes par IA

**Expertise en Développement Quantitatif & Recherche Opérationnelle**
*Groupe 3 - ECE 2026*

---

## 🎯 La Problématique
### Comment optimiser son patrimoine sur 30 ans ?

- **Complexité** : Arbitrage entre 6 classes d'actifs (Actions, Crypto, SCPI...).
- **Contraintes** : Frais de transaction, inflation, pénalités de liquidité.
- **Aléas** : Volatilité des marchés financiers.
- **Objectifs de vie** : Financer des projets (Immobilier, Études) à des dates précises.

---

## 🧠 Modélisation Mathématique
### Le problème comme un MDP (Markov Decision Process)

- **État ($S_t$)** : Richesse actuelle, Temps restant, Allocation précédente.
- **Action ($A_t$)** : Vecteur de poids d'investissement ($\sum \omega_i = 1$).
- **Transition** : $W_{t+1} = (W_t - C_t - \text{Frais}) \times (1 + r_{\text{portefeuille}})$.
- **Objectif** : Maximiser l'utilité CRRA de la richesse finale.

---

## 🏗️ Architecture Logicielle
### Une conception modulaire et robuste

- **`src/core`** : Moteur mathématique et configurations.
- **`src/solvers`** : Algorithmes d'optimisation découplés.
- **`src/simulation`** : Moteur Monte Carlo (200+ trajectoires).
- **`src/utils`** : Visualisation haute performance (Seaborn/Plotly).
- **`dashboard.py`** : Interface Web interactive (Streamlit).

---

## 🚀 Les 3 Moteurs d'Optimisation

### 1. Programmation Dynamique (DP)
- **Algorithme** : Induction arrière de Bellman.
- **Atout** : Garantie d'optimalité théorique globale.

### 2. Optimisation Linéaire (OR-Tools)
- **Algorithme** : Moyenne-Variance locale.
- **Atout** : Vitesse d'exécution instantanée (< 0.1ms).

### 3. Reinforcement Learning (RL)
- **Algorithme** : PPO (Proximal Policy Optimization).
- **Atout** : Apprentissage de stratégies robustes et lisses.

---

## 📈 Univers d'Actifs & Réalisme
### 6 Classes d'actifs gérées

- **Sécurisé** : Cash, Obligations.
- **Diversifié** : Or, SCPI (Immobilier papier).
- **Dynamique** : Actions, Crypto (BTC).

**Réalisme financier intégré :**
- Frais d'entrée SCPI (10%).
- Pénalités de vente forcée (Liquidité).
- Frais de transaction par mouvement (0.1%).

---

## 💻 Interface Utilisateur (Streamlit)
### L'optimisation accessible à tous

- **Configuration sans code** : Sliders pour le capital et l'épargne.
- **Plan de vie interactif** : Tableau dynamique pour ajouter des événements.
- **Visualisation interactive** : Graphiques Plotly (zoom, survol).
- **KPIs en temps réel** : Probabilité de succès du plan de vie.

---

## 📊 Résultats & Comparaison
### Quelle IA gagne ?

| Métrique | DP | OR-Tools | RL (PPO) |
|----------|----|----------|----------|
| **Richesse Finale** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Sharpe Ratio** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Vitesse** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

- **RL** : Meilleure gestion du risque (Sharpe Ratio > 1.1).
- **DP** : Stratégie la plus stable et prévisible.
- **OR-Tools** : Idéal pour le calcul haute fréquence.

---

## 🏁 Conclusion
### Un outil complet d'aide à la décision

- **Modularité** : Facile d'ajouter de nouveaux actifs ou solveurs.
- **Performance** : Comparaison rigoureuse de 3 paradigmes d'IA.
- **Accessibilité** : Une Web App prête pour l'utilisateur final.

**Perspectives** : Intégration de modèles macro-économiques (régimes de marché).
