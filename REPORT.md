# Rapport de Projet Final : Optimisation d'Allocation d'Actifs (Univers Étendu)

## 1. Extension de l'Univers d'Investissement

Le modèle intègre désormais 6 classes d'actifs avec des caractéristiques de risque et de frais distinctes :

| Actif | Rendement Espéré | Volatilité | Frais (Achat/Vente) |
|-------|------------------|------------|---------------------|
| **Actions** | 8% | 15% | 0.1% / 0.1% |
| **Obligations** | 3% | 5% | 0.05% / 0.05% |
| **Cash** | 1% | 0.5% | 0% / 0% |
| **Or** | 4% | 12% | 0.2% / 0.2% |
| **Crypto (BTC)** | 15% | 60% | 0.5% / 0.5% |
| **SCPI** | 4.5% | 5% | **10% / 5%** |

## 2. Analyse des Performances (Benchmark 6 Actifs)

| Solveur | Temps de résolution | Richesse Finale Moyenne | Sharpe Ratio |
|---------|---------------------|-------------------------|--------------|
| **DP** (Classes de risque) | ~0.84s | 148.54 | 0.81 |
| **OR-Tools** (Myope-Variance) | < 0.0001s | **190.24** | 0.83 |
| **RL** (PPO - 20k steps) | ~25.12s | 145.95 | **1.19** |

### Observations clés :

1.  **Efficacité de la DP par classes** : En regroupant les actifs par classes de risque (Sécurisé, Modéré, Dynamique), la DP parvient à obtenir un résultat robuste. Cette simplification permet d'éviter l'explosion combinatoire tout en maintenant une gestion prudente.
2.  **Impact des frais SCPI** : Les frais d'entrée de 10% sur la SCPI ont naturellement poussé les modèles à limiter les allers-retours sur cet actif.
3.  **Supériorité du RL sur le Risque** : L'agent RL obtient le meilleur **Sharpe Ratio (1.19)**. Il a appris à naviguer autour des événements de vie en stabilisant son allocation pour minimiser la volatilité, malgré un univers d'actifs complexe.
4.  **Robustesse de OR-Tools** : L'approche moyenne-variance déterministe reste très compétitive pour maximiser la richesse brute.

## 3. Visualisation et Design Professionnel

Le projet intègre désormais une charte graphique de type "Rapport d'Investissement" :
- **Palette Sémantique** : Couleurs fixes pour chaque actif (Bleu pour Actions, Vert pour Obligations, Or pour Gold, etc.).
- **Stacked Area Charts** : Visualisation de l'évolution de la composition du portefeuille.
- **Analyse de Convergence** : Graphiques de richesse avec zones d'ombre représentant l'écart-type et les percentiles.
- **Distribution de la Richesse** : Utilisation de *Violin Plots* pour comparer la dispersion des résultats finaux.

## 4. Conclusion Technique

L'ajout de frais de transaction spécifiques et d'actifs variés a considérablement augmenté le réalisme du simulateur. Le projet démontre que le **Reinforcement Learning** offre une flexibilité inégalée pour explorer des univers d'investissement complexes et optimiser le couple rendement/risque.
