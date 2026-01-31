# Rapport Technique : Optimisation de Gestion de Patrimoine par Intelligence Artificielle
# 💰 Wealth Planner AI : Optimisation d'Investissement Multi-Périodes

Projet d'Expertise en Développement Quantitatif et Recherche Opérationnelle (ECE 2026).

Ce projet propose une plateforme avancée de simulation et d'optimisation de gestion de patrimoine. Il permet de concevoir des stratégies d'allocation d'actifs optimales sur le long terme, en tenant compte des objectifs de vie, des contraintes de liquidité et des frais de transaction.


## 🛠️ Installation Détaillée

### 1. Prérequis
- Python 3.10 ou supérieur installé.
- Un terminal (PowerShell recommandé sur Windows).

### 2. Clonage et Configuration
```powershell
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement
# Sur Windows (PowerShell) :
.\venv\Scripts\Activate.ps1
# Sur macOS/Linux :
source venv/bin/activate

# Installation des dépendances
pip install -r "Projet Mathias Elena/requirements.txt"
```

---

## 🚀 Guide d'Utilisation

### Option A : Interface Web Interactive (Recommandé)
C'est la méthode la plus simple pour tester vos propres scénarios.
```powershell
cd "Projet Mathias Elena"
..\venv\Scripts\streamlit run dashboard.py
```
**Dans l'interface :**
1. Ajustez votre **Capital Initial** et votre **Épargne Mensuelle** dans la barre latérale.
2. Définissez votre **Âge** et votre **Horizon de Retraite**.
3. Modifiez le tableau des **Événements de Vie** (ajoutez des lignes pour vos projets).
4. Cliquez sur **"Calculer la Stratégie Optimale"**.
5. Explorez les onglets **Richesse**, **Allocation** et **Comparaison**.

### Option B : Benchmark complet via CLI
Pour générer tous les rapports et fichiers CSV de comparaison :
```powershell
python "Projet Mathias Elena/main.py"
```
Les résultats seront générés dans le dossier `Projet Mathias Elena/output/`.

---


### 1. Introduction et Contexte
    1.1. Le Problème
La gestion de patrimoine à long terme est un problème complexe. Il ne s'agit pas seulement de "gagner de l'argent", mais de financer des projets de vie précis (achat immobilier, études des enfants, retraite) dans un environnement incertain (inflation, krachs boursiers).

Les conseillers financiers traditionnels utilisent souvent des règles statiques (ex: "60% actions, 40% obligations"). Notre projet vise à dépasser cette approche en utilisant des algorithmes d'optimisation avancés et l'Intelligence Artificielle pour adapter dynamiquement l'allocation d'actifs.

    1.2. Objectif
Développer une application capable de proposer une stratégie d'investissement optimale sur 30 ans, en tenant compte :

-De multiples classes d'actifs (Actions, Crypto, Or, SCPI, etc.).

-De contraintes réalistes (frais de transaction, pénalités de liquidité).

-D'objectifs financiers datés (sorties de cash flow).


### 2. Modélisation Mathématique de la Solution
Nous avons modélisé le problème sous la forme d'un Processus de Décision Markovien (MDP) à horizon fini.

    2.1. L'Univers d'Investissement (config.py)
Nous avons défini un univers réaliste composé de 6 classes d'actifs aux propriétés distinctes :

- **Moteurs de Performance** : Actions (Rendement 8%, Volatilité 15%), Crypto (Rendement 15%, Volatilité 60%).

- **Actifs de Sécurité** : Obligations, Cash, Or.

- **Actif Illiquide** : SCPI (Immobilier papier).

    2.2. Les Contraintes de Réalisme (model.py)
Contrairement aux modèles théoriques simplistes, notre moteur intègre des frictions réelles :

- **Frais de Transaction** : Chaque réallocation coûte de l'argent (décourage le "trading fou").

- **Pénalité de Vente Forcée** : Si le modèle doit vendre des SCPI en urgence pour payer une dette, il subit une décote de 15% (simulant l'illiquidité immobilière).

- **Fonction d'Utilité CRRA** : Nous utilisons une utilité Constant Relative Risk Aversion. Cela signifie que l'IA est "punie" mathématiquement si elle prend des risques qui mènent à la ruine.


### 3. Architecture Technique
Nous avons adopté une architecture logicielle modulaire et extensible pour permettre la comparaison de plusieurs intelligences.

Projet Mathias Elena/
│
├── output/                     # 📂 Résultats générés (Preuves de performance)
│   ├── comparison_results.csv  # Données brutes des simulations normales
│   ├── summary.csv             # Tableau récapitulatif (Sharpe, Richesse finale)
│   ├── stress_summary.csv      # Résultats des tests de crise (Krach)
│   ├── *_wealth_prof.png       # Graphiques de convergence (RL, DP, OR-Tools)
│   ├── *_alloc_prof.png        # Graphiques d'allocation d'actifs (Zones colorées)
│   └── robustness_comparison.png # Le graphique clé (Barres Rouge vs Vert)
│
├── src/                        # 🧠 Code Source (Cœur du réacteur)
│   ├── analysis/               # Modules d'analyse post-simulation
│   │   ├── plot_robustness.py  # Génération du graphique comparatif (Normal vs Crise)
│   │   └── stress_analysis.py  # Moteur de Stress-Test (Simulation de Krach)
│   │
│   ├── core/                   # Définitions fondamentales
│   │   ├── config.py           # Paramètres (Univers d'actifs, frais, contraintes de vie)
│   │   └── model.py            # Physique du monde (MDP, Utilité CRRA, Transitions)
│   │
│   ├── simulation/             # Moteur de calcul
│   │   └── engine.py           # Simulation Monte Carlo (Génération de scénarios aléatoires)
│   │
│   ├── solvers/                # Les "Cerveaux" (Algorithmes)
│   │   ├── base.py             # Interface commune (Abstract Base Class)
│   │   ├── dp_solver.py        # Programmation Dynamique (Référence mathématique)
│   │   ├── ortools_solver.py   # Optimisation Linéaire (Google OR-Tools)
│   │   └── rl_solver.py        # Intelligence Artificielle (PPO / Stable-Baselines3)
│   │
│   └── utils/                  # Outils transverses
│       └── plotting.py         # Moteur graphique "Pro" (Intervalles de confiance, etc.)
│
├── .gitignore                  # ⚙️ Configuration Git (Ignore venv/ et output/)
├── config.toml                 # 🎨 Configuration du thème visuel de Streamlit
├── dashboard.py                # 🖥️ Interface Web Utilisateur (Application finale)
├── main.py                     # 🚀 Point d'entrée CLI (Orchestrateur complet : Simu + Plot + Stress)
├── PLAN.md                     # 📝 Feuille de route et étapes du projet
├── README.md                   # 📖 Documentation d'installation et d'usage
├── REPORT.md                   # 📄 Rapport technique détaillé (celui qu'on a rédigé)
├── requirements.txt            # 📦 Liste des librairies (pandas, torch, streamlit...)
└── start.bat                   # ⚡ Script d'installation et lancement automatique (Windows) 


### 4. Comparaison des Algorithmes (Les "Trois Cerveaux")
Pour résoudre ce problème, nous avons implémenté trois stratégies radicalement différentes :

    4.1. Programmation Dynamique (DP) - dp_solver.py
- **Principe** : Utilise l'Équation de Bellman pour résoudre le problème par induction arrière. Elle discrétise l'espace de richesse et calcule, pour chaque état possible, l'action qui maximise l'utilité future espérée.

- **Force** : Garantie mathématique de trouver l'optimum théorique.

- **Usage** : Idéal pour les horizons longs et les sorties de cash prévisibles.

    4.2. Optimisation Linéaire (OR-Tools) - ortools_solver.py
- **Principe** : Utilise le solveur de Google pour maximiser l'espérance de gain à chaque année.

- **Comportement observé** : C'est une stratégie "naïve" et "avare". Le solveur détecte que les actifs risqués (Crypto/Actions) ont le meilleur rendement moyen et y alloue 100% du capital, ignorant la variance (le risque).

    4.3. Reinforcement Learning (RL / PPO) - rl_solver.py
- **Principe** : Utilisation de l'algorithme PPO (Proximal Policy Optimization). Un réseau de neurones apprend par essai-erreur en simulant des millions d'années d'investissement.

- **Force** : Apprentissage "organique". L'IA découvre d'elle-même des concepts complexes comme la diversification temporelle (réduire le risque à l'approche d'une échéance) sans qu'on lui ait programmé explicitement.

- **Usage** : Meilleure gestion du risque (Sharpe Ratio élevé).


### 5. Défis Rencontrés et Solutions
Au cours du développement, nous avons fait face à plusieurs obstacles majeurs :

    5.1. Le Problème de l'Échelle (Scaling)
- **Problème** : Au début, les chiffres de richesse (ex: 200 000€) étaient trop grands pour le réseau de neurones du RL, qui préfère des chiffres entre 0 et 1.

- **Solution** : Dans rl_solver.py, nous avons normalisé les observations (divisé par la richesse initiale) pour stabiliser l'apprentissage de l'IA.

    5.2. Les Graphiques Illisibles
- **Problème** : Nos simulations généraient des graphiques "spaghettis" avec 200 courbes superposées, rendant l'analyse impossible.

- **Solution** : Nous avons développé plotting.py pour générer des graphiques professionnels affichant la moyenne et un intervalle de confiance (zone d'ombre), offrant une vision claire de la tendance et du risque.

    5.3. Le "Faux Positif" d'OR-Tools
- **Problème** : En conditions normales, OR-Tools affichait des performances largement supérieures aux autres, car il prenait des risques inconsidérés qui payaient "en moyenne".

- **Solution** : Implémentation d'un module de Stress Test (stress_analysis.py) pour révéler la fragilité cachée de cette stratégie.


### 6. Analyse des Résultats et Robustesse
C'est le point clé de notre étude. Nous ne nous sommes pas contentés de regarder la performance moyenne. Nous avons comparé les solveurs dans deux mondes : un "Monde Normal" et un "Monde en Crise" (Krach boursier avec volatilité doublée).

    6.1. Le Cas OR-Tools (L'Optimisation "Naïve")
- **En temps normal** : Il affiche les gains les plus élevés (souvent > 1.4M€).

- **En crise** : C'est un effondrement total. Ayant tout misé sur les actifs risqués sans diversifier, il subit un taux de ruine proche de 80% et le capital tombe à un niveau dérisoire (~28 k€).

- **Verdict** : Une stratégie "Tête brûlée", inacceptable pour un particulier qui joue sa retraite.

    6.2. Le Cas DP (La Référence Mathématique)
- **En temps normal** : Une performance modérée (~380k€ - 780k€ selon les configurations), bridée par sa prudence extrême.

- **En crise** : C'est le champion de la sécurité. Avec un taux de ruine quasi-nul (~1.5% à 5%) et le capital préservé le plus élevé (~190 k€), la DP démontre mathématiquement qu'il est possible de survivre à un krach en gérant parfaitement le risque.

- **Verdict** : Le "Gold Standard" de la sécurité, mais techniquement lourd à mettre en place (temps de calcul exponentiel).

    6.3. Le Cas RL (L'Intelligence Artificielle)
- **Positionnement** : L'agent RL se situe à l'équilibre parfait.

- **Performance** : Il sacrifie le rendement "théorique maximum" d'OR-Tools pour acheter de la sécurité.

- **Résilience** : En crise, il parvient à maintenir un capital solide (~146k€ - 164k€) et un taux de ruine faible, se rapprochant des performances de sécurité de la DP.

- **Verdict** : L'IA a réussi à "apprendre" la prudence de la Programmation Dynamique par l'expérience, tout en conservant une flexibilité d'allocation supérieure.

    6.4. Synthèse Visuelle (Graphique d'Allocation)
L'analyse des graphiques d'allocation (_alloc_prof.png) explique ces résultats :

-OR-Tools est monochrome (100% Actions/Crypto) : aucune couverture.

-Le RL et la DP montrent des "couches" de couleurs (Diversification). Ils réduisent la voilure (vente d'actions pour des obligations/cash) à l'approche des échéances de paiement (année 12 pour l'immobilier, année 30 pour la retraite). L'IA a donc redécouvert seule les principes de la gestion de fortune prudente.


### 7. Conclusion
Ce projet a permis de construire une application complète de Robo-Advisor intelligent.

Nous avons démontré que si les méthodes classiques (OR-Tools) sont rapides, elles sont dangereuses pour un épargnant. L'approche par Reinforcement Learning s'est révélée être la plus robuste, capable de construire une allocation d'actifs dynamique qui s'adapte à l'approche des échéances financières (retraite, achats) tout en naviguant prudemment à travers les risques de marché.

L'application finale (Dashboard) permet à un utilisateur de visualiser ces trajectoires et de comprendre l'intérêt d'une gestion diversifiée pilotée par l'IA.