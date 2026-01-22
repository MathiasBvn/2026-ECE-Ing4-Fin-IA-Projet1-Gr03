# 💰 Wealth Planner AI : Optimisation d'Investissement Multi-Périodes

Projet d'Expertise en Développement Quantitatif et Recherche Opérationnelle (ECE 2026).

## 🚀 Lancement Rapide

### 1. Installation
```powershell
# Création de l'environnement virtuel
py -m venv venv

# Activation (PowerShell)
.\venv\Scripts\Activate.ps1

# Installation des dépendances
pip install -r "Projet Mathias Elena/requirements.txt"
```

### 2. Lancer l'Interface Interactive (Streamlit)
```powershell
cd "Projet Mathias Elena"
..\venv\Scripts\streamlit run dashboard.py
```
*Note : Si vous utilisez CMD au lieu de PowerShell, remplacez `;` par `&&`.*

### 3. Lancer le Benchmark complet (CLI)
```powershell
python "Projet Mathias Elena/main.py"
```

---

## 🧠 Présentation du Projet
Ce projet résout le problème complexe de l'allocation d'actifs multi-périodes en intégrant des événements de vie (achat immobilier, études, etc.) et des contraintes de liquidité réelles.

### Approches Comparées :
1.  **Programmation Dynamique (DP)** : Induction arrière de Bellman sur grille de richesse.
2.  **Optimisation Linéaire (OR-Tools)** : Approche moyenne-variance rapide sous contraintes.
3.  **Reinforcement Learning (RL)** : Agent PPO (Stable-Baselines3) apprenant dans un environnement Gymnasium.

### Actifs Supportés :
- Actions, Obligations, Cash, Or, Crypto (BTC), SCPI (Immobilier).
- Gestion des frais de transaction spécifiques et pénalités de liquidité.

---

## 📊 Visualisations Professionnelles
L'application génère des rapports visuels de haute qualité :
- **Stacked Area Charts** pour l'évolution de l'allocation.
- **Convergence de Richesse** avec intervalles de confiance.
- **Violin Plots** pour la distribution du patrimoine final.

---

## 📅 Modalités du projet

### Échéances importantes
- **07 janvier 2026** : Présentation des sujets proposés
- **20 janvier** : Checkpoint intermédiaire
- **02 février** : Présentation finale et rendu

### Taille des groupes
La taille standard d'un groupe est de 3 personnes, avec +1 pour les groupes de 2 et -1 pour les groupes de 4

### Évaluation
- Présentation/communication
- Contenu théorique, contexte et perspectives
- Contenu technique, performances, qualité du code et du logiciel
- Organisation/Collaboration (notamment activité git)

### Livrables attendus
- Code source documenté
- README de présentation avec infos essentielles, procédure d'installation et tests
- Slides de la présentation

### 📋 Instructions de soumission

#### ⚠️ IMPORTANT : Organisation du travail

> **ATTENTION** : Tout votre travail **DOIT** être organisé dans un **sous-répertoire dédié** à votre groupe.
>
> **Structure obligatoire** :
> ```
> /groupe-XX-nom-sujet/
> ├── README.md          # Documentation de votre projet
> ├── src/               # Code source
> ├── docs/              # Documentation technique
> ├── slides/            # Support de présentation (PDF ou lien)
> └── ...
> ```
>
> ❌ **Ne pas** mettre vos fichiers à la racine du dépôt
> ✅ **Tout** doit être dans votre sous-répertoire de groupe

#### Soumission du code et de la documentation
1. **Créer un fork** de ce dépôt sur votre compte GitHub (vous n'avez pas les droits d'écriture sur ce dépôt)
2. **Créer un sous-répertoire** pour votre groupe : `groupe-XX-nom-sujet/` (ex: `groupe-03-portfolio-csp/`)
3. **Développer votre projet** exclusivement dans ce sous-répertoire
4. **Soumettre une Pull Request** vers ce dépôt **au moins 2 jours avant la présentation** (soit le **31 janvier 2026** au plus tard)
5. La PR doit inclure :
   - Le code source complet et fonctionnel dans votre sous-répertoire
   - Un README détaillé dans votre sous-répertoire (installation, utilisation, tests)
   - La documentation technique

#### Soumission du support de présentation
- Les slides de présentation doivent être soumises **avant le début de la présentation** (soit le **02 février 2026** au matin)
- Format accepté : PDF, PowerPoint, ou lien vers Google Slides/Canva
- Ajouter les slides dans votre sous-répertoire (`groupe-XX/slides/`) ou partager le lien dans le README de votre sous-répertoire

#### Checklist de soumission
- [ ] Fork du dépôt créé
- [ ] Sous-répertoire `groupe-XX-nom-sujet/` créé avec tout le contenu dedans
- [ ] README avec procédure d'installation et tests dans le sous-répertoire
- [ ] Pull Request créée et reviewable
- [ ] Slides de présentation soumises (dans le sous-répertoire ou lien dans README)
- [ ] Tous les membres du groupe identifiés dans la PR (noms + GitHub usernames)

---

## 🎯 Sujets détaillés pour le projet

### 1. Optimisation de plannings infirmiers

**Description du problème et contexte**
La planification du personnel soignant consiste à affecter de manière optimale les infirmier·ère·s aux différents shifts (matin, après-midi, nuit) sur une période donnée, tout en respectant des contraintes légales (durées maximales de travail, jours de repos), opérationnelles (couverture des besoins par créneau) et de préférences individuelles. Ce problème NP-difficile se prête particulièrement bien à la programmation par contraintes (CSP) pour modéliser et résoudre l'ensemble des exigences.

**Références multiples**
- **Revue de littérature** : Burke et al., _The state of the art of nurse rostering_ (2004) - Méthodes d'optimisation des plannings
- **CP Optimizer** : [IBM CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-cp-optimizer) - Programmation par contraintes pour le staffing
- **OR-Tools** : [Solver Max - Nurse rostering](https://www.solvermax.com/resources/models/staff-scheduling/nurse-rostering-in-or-tools-cp-sat-solver) - Modèle CSP avec CP-SAT
- **Tutoriel** : [Solving Nurse Rostering with Google OR-Tools](https://medium.com/@mobini/solving-the-nurse-rostering-problem-using-google-or-tools-755689b877c0) - Modélisation détaillée

**Approches suggérées**
- Modéliser les variables (infirmier·ère·s, shifts, jours) avec leurs domaines d'affectation
- Implémenter les contraintes de couverture, repos et préférences individuelles
- Utiliser un solveur CSP (OR-Tools CP-SAT, IBM CP Optimizer) ou approche hybride (CSP + MILP)
- Développer un notebook explicatif avec analyse comparative sur différentes instances de test

**Technologies pertinentes**
- Python avec OR-Tools CP-SAT ou IBM CP Optimizer pour la résolution CSP
- iZinc pour la modélisation déclarative de contraintes
- Pandas pour la manipulation des données de planification
- Matplotlib/Plotly pour la visualisation des emplois du temps

### 2. Modélisation de la propagation COVID-19 avec algorithmes génétiques

**Description du problème et contexte**
La modélisation mathématique de la propagation épidémique est essentielle pour la prise de décision sanitaire. Les algorithmes génétiques permettent d'optimiser les paramètres des modèles SIR/SEIR pour mieux correspondre aux données réelles de propagation COVID-19 et prédire les scénarios futurs.

**Références multiples**
- **Publication principale** : [arXiv:2008.12020](https://arxiv.org/abs/2008.12020) - Modélisation épidémique avec approches évolutionnaires
- **Deep Q-Learning** : [ACM DOI](https://dl.acm.org/doi/pdf/10.1145/3340531.3412179) - Apprentissage par renforcement pour épidémies
- **Optimisation** : [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0960077920302836) - Métaheuristiques pour modèles épidémiologiques

**Approches suggérées**
- Implémenter un modèle SIR/SEIR avec paramètres optimisables
- Développer un algorithme génétique pour calibrer les paramètres sur données réelles
- Intégrer des contraintes réalistes (capacité hospitalière, mesures sanitaires)
- Visualiser l'évolution de l'épidémie sous différents scénarios

**Technologies pertinentes**
- Python avec NumPy, SciPy pour la modélisation mathématique
- DEAP ou PyGAD pour les algorithmes génétiques
- Matplotlib/Plotly pour la visualisation des courbes épidémiques
- Pandas pour la manipulation des données réelles

---

### 3. Problème d'échange de reins (Kidney Exchange)

**Description du problème et contexte**
L'appariement optimal de donneurs et receveurs d'organes incompatibles se modélise comme un graphe orienté où chaque cycle représente un échange de greffes. L'objectif est de maximiser le nombre de transplantations effectuées, sous la contrainte qu'aucun couple ne donne sans recevoir (stabilité individuelle). Ce problème d'optimisation combinatoire NP-difficile admet de multiples variantes selon la longueur des cycles d'échange autorisés (longueur 2, 3 ou plus).

**Références multiples**
- **Publication principale** : Roth et al., _Efficient Kidney Exchange_ (AER 2007) - Fondements théoriques
- **Algorithmes** : Abraham et al., _Clearing Algorithms for Barter Exchange_ (EC 2007) - Méthodes de résolution
- **Implémentation** : [GitHub - kidney_solver](https://github.com/jamestrimble/kidney_solver) - Solveur Python/Gurobi
- **Documentation** : [Wikipedia - Optimal kidney exchange](https://en.wikipedia.org/wiki/Optimal_kidney_exchange) - Définitions et contraintes

**Approches suggérées**
- Modéliser les paires donneur-receveur comme sommets d'un graphe orienté avec arcs de compatibilité
- Implémenter des algorithmes de recherche de cycles optimaux (programmation linéaire ou contraintes)
- Développer des heuristiques pour traiter des instances de grande taille
- Comparer différentes formulations (graphe de cycles, matching multi-dimensionnel)

**Technologies pertinentes**
- Python avec NetworkX pour la manipulation de graphes
- Gurobi ou OR-Tools pour l'optimisation combinatoire
- PuLP pour la modélisation en programmation linéaire
- Visualisation avec Graphviz ou Matplotlib pour représenter les échanges

### 4. Identification d'inhibiteurs moléculaires COVID-19

**Description du problème et contexte**
La recherche d'inhibiteurs moléculaires contre la protéase principale du SARS-CoV-2 est une approche thérapeutique cruciale. Ce sujet explore l'utilisation de techniques computationnelles pour identifier des composés naturels potentiels pouvant bloquer l'activité de cette enzyme virale.

**Références multiples**
- **Publication principale** : [Inhibiteurs COVID-19](http://lavierebelle.org/IMG/pdf/2020_potential_inhibitor_of_covid-19_main_protease_from_several_medicinal_plant_compounds.pdf) - Analyse de composés végétaux
- Bases de données moléculaires : PubChem, ChEMBL pour les structures chimiques
- Outils de docking moléculaire : AutoDock Vina, SwissDock

**Approches suggérées**
- Analyser les structures 3D de la protéase principale COVID-19
- Implémenter des algorithmes de similarité structurelle entre molécules
- Développer un système de scoring pour évaluer le potentiel d'inhibition
- Utiliser des techniques d'apprentissage automatique pour prédire l'activité biologique

**Technologies pertinentes**
- Python avec RDKit pour la chimie computationnelle
- BioPython pour les structures protéiques
- Machine Learning avec scikit-learn pour la prédiction d'activité
- Visualisation moléculaire avec PyMOL ou Chimera

---

### 5. Ordonnancement de production (Job-Shop Scheduling)

**Description du problème et contexte**
Le Job-Shop Scheduling consiste à planifier l'exécution d'un ensemble de tâches (jobs) devant être traitées sur plusieurs machines, chacune ayant une capacité limitée (une tâche par machine à la fois). L'objectif principal est de imiser le makespan (durée totale de production) tout en optimisant l'utilisation des ressources. Des contraintes supplémentaires peuvent être intégrées : maintenance programmée, ressources cumulatives, objectifs multi-critères.

**Références multiples**
- **Introduction** : [Job Shop Scheduling Problem | sysid blog](https://sysid.github.io/job-shop-scheduling-problem/) - Formulation générale du problème
- **Solveurs CP** : [IBM CP Optimizer](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-cp-optimizer) - Présentation des solveurs utilisés
- **Référence historique** : J. Carlier (1982), _Proc. of first job-shop scheduling constraint solver_ - Résolution par contraintes
- **Études de cas** : Travaux académiques sur l'impact de la propagation des contraintes

**Approches suggérées**
- Utiliser des variables d'intervalle pour chaque opération (début et durée fixe)
- Implémenter des contraintes de non-chevauchement (une machine = une tâche à la fois)
- Respecter l'ordre prédéfini des opérations pour chaque job
- Explorer des stratégies d'optimisation hybride (CP combiné avec heuristiques)

**Technologies pertinentes**
- Python avec OR-Tools CP-SAT pour la modélisation et résolution
- IBM CP Optimizer pour les instances industrielles complexes
- iZinc pour la modélisation déclarative de contraintes
- Gantt charts avec Matplotlib pour la visualisation des plannings

### 6. Optimisation hospitalière avec métaheuristiques

**Description du problème et contexte**
L'optimisation des ressources hospitalières est critique pour améliorer la qualité des soins et réduire les coûts. Les métaheuristiques permettent de résoudre des problèmes complexes d'allocation de lits, planification du personnel et gestion des flux patients dans des environnements contraints.

**Références multiples**
- **Décharge patients** : [PMC543827](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC543827/) - Optimisation des durées de séjour
- **Planning patients** : [arXiv:1805.02264](https://arxiv.org/pdf/1805.02264.pdf) - Ordonnancement des interventions
- **Planning soignants** : [Strathprints](https://strathprints.strath.ac.uk/59727/1/Rahimian_etal_COR_2017_A_hybrid_integer_and_constraint_programg_approach.pdf) - Optimisation du personnel

**Approches suggérées**
- Modéliser les contraintes hospitalières (personnel, équipements, réglementations)
- Implémenter plusieurs métaheuristiques (recuit simulé, recherche tabou, colonies de fourmis)
- Développer un système multi-objectifs (qualité des soins, coûts, satisfaction patient)
- Créer une interface de simulation pour tester différents scénarios

**Technologies pertinentes**
- Python avec OR-Tools ou PuLP pour la programmation par contraintes
- Métaheuristiques avec MetaPy ou implémentation personnalisée
- Base de données SQL pour la gestion des données hospitalières
- Interface web avec Flask/Django pour la visualisation

---

### 7. Planification d'emploi du temps universitaire

**Description du problème et contexte**
La planification des emplois du temps universitaires (cours ou examens) consiste à assigner des créneaux horaires et des salles en tenant compte de multiples contraintes : disponibilité des enseignants, capacité et disponibilité des salles, évitement des conflits horaires, répartition équilibrée des cours, et intégration de préférences. Ce problème NP-combinatoire bénéficie grandement de l'approche CSP qui permet une modélisation déclarative des contraintes et des techniques de propagation efficaces.

**Références multiples**
- **CLP pour timetabling** : [Constraint Logic Programming over finite domains](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=00f0110d17de0d95bbbdbea822bebeede956d64e) - Application du CLP aux emplois du temps
- **Thèse de référence** : [Constraint-based Timetabling](https://www.unitime.org/papers/phd05.pdf) - Méthodes CP appliquées à la timetabling
- **Travaux allemands** : Goltz & Matzke (1999), _University Timetabling using Constraint Logic Programming_ - Encodage CLP et analyse
- **Recherche locale** : Schaus et al. (2014), _CBLS for Course Timetabling_ - Optimisation des emplois du temps
- **Compétition** : International Timetabling Competition - Ressources et données réelles de planification

**Approches suggérées**
- Modéliser avec des variables pour les créneaux horaires et salles affectées à chaque cours/examen
- Implémenter des contraintes d'exclusion mutuelle (pas deux activités simultanées pour un même enseignant/salle)
- Gérer les contraintes de capacité et disponibilités des ressources
- Optimiser en minimisant les conflits et maximisant la satisfaction des préférences

**Technologies pertinentes**
- MiniZinc ou Choco pour la modélisation déclarative de contraintes
- OR-Tools CP-SAT pour la résolution avec techniques de propagation avancées
- Python avec frameworks CSP (python-constraint, Google OR-Tools)
- Visualisation avec calendriers interactifs (FullCalendar, bibliothèques Planning)

---

### 8. Systèmes experts médicaux en programmation logique

**Description du problème et contexte**
Les systèmes experts médicaux reproduisent le raisonnement clinique des médecins en utilisant des règles logiques. Ce sujet explore l'implémentation de moteurs d'inférence capables de diagnostiquer des pathologies courantes basées sur des symptômes et antécédents patients.

**Références multiples**
- **Systèmes experts** : [HAL Archives](https://hal.archives-ouvertes.fr/hal-01610722/document) - Conception et architecture
- **Diabète** : [ScienceDirect](https://pdf.sciencedirectassets.com/280203/1-s2.0-S1877050915X00275/1-s2.0-S1877050915028604/main.pdf) - Application pratique au diabète
- **Logique** : [MobileDSS](http://www.mobiledss.uottawa.ca/fileadmin/publications/pdf/paper_jms_2016.pdf) - Programmation logique médicale

**Approches suggérées**
- Développer un moteur d'inférence en chaînage avant/arrière
- Créer une base de connaissances avec règles médicales structurées
- Implémenter des mécanismes de gestion d'incertitude (facteurs de confiance)
- Intégrer une interface pour l'acquisition des symptômes patients

**Technologies pertinentes**
- Prolog pour la programmation logique naturelle
- Python avec PyKE ou CLIPS pour les systèmes experts
- Base de connaissances en format XML/JSON
- Interface web avec React/Vue pour l'interaction utilisateur

---



### 9. Solveur de Wordle par CSP (et LLM)

**Description du problème et contexte**
Wordle est un jeu de mots dans lequel à chaque tentative de mot, on obtient des indications de lettres bien placées, mal placées ou absentes. Ces indices se traduisent par des contraintes sur le mot secret : certaines positions doivent contenir certaines lettres, d'autres non, etc. Un programme peut appliquer ces contraintes à un dictionnaire pour filtrer les mots possibles. Par exemple, une approche par contraintes définit des variables pour chaque lettre du mot secret et impose les retours (vert, jaune, gris) comme contraintes logiques sur ces variables.

**Références multiples**
- **Approche CSP** : [Beating Wordle: Constraint Programming](https://medium.com/better-programming/beating-wordle-constraint-programming-ef0b0b6897fe) - Utilisation d'un solver de contraintes sur un dataset de mots
- **Implémentation** : hakank.org - Implémentation d'un solveur Wordle en OR-Tools CP-SAT
- **Function calling** : [OpenAI Function calling documentation](https://platform.openai.com/docs/guides/function-calling) - Appel de fonctions pour déléguer des tâches (ex. solveur externe)
- **Intégration LLM** : On peut intégrer un LLM en function-calling pour qu'il exploite un solveur CSP sous-jacent et propose des coups optimisés

**Approches suggérées**
- Définir des variables pour chaque lettre du mot secret et imposer les contraintes de retour (vert/jaune/gris)
- Utiliser un solveur de contraintes pour réduire l'espace des solutions à chaque coup
- Intégrer un LLM via function calling pour déduire les contraintes linguistiques
- Développer une stratégie d'optimisation pour minimiser le nombre de tentatives

**Technologies pertinentes**
- Python avec python-constraint ou OR-Tools CP-SAT pour la résolution
- Dictionnaires de mots français/anglais pour les domaines de variables
- API OpenAI ou modèles locaux pour l'intégration LLM
- Interface web avec React/Vue pour une expérience interactive

---

### 10. Solveurs SMT pour la biologie synthétique

**Description du problème et contexte**
La biologie synthétique nécessite la vérification formelle de circuits génétiques pour garantir leur comportement attendu. Les solveurs SMT (Satisfiability Modulo Theories) permettent de vérifier mathématiquement les propriétés de systèmes biologiques complexes avant leur implémentation.

**Références multiples**
- **Publication Z3** : [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/07/pyhwk14.pdf) - Application de Z3 à la biologie
- **Projet Z3** : [Z3 for Biology](https://www.microsoft.com/en-us/research/project/z3-4biology/) - Framework spécialisé
- **Bio Model Analyzer** : [biomodelanalyzer.org](http://biomodelanalyzer.org/) - Outil d'analyse en ligne
- **Dépôt officiel** : [BioModelAnalyzer GitHub](https://github.com/Microsoft/BioModelAnalyzer) - Code source complet

**Approches suggérées**
- Modéliser les circuits génétiques en logique temporelle
- Utiliser Z3 pour vérifier des propriétés (stabilité, oscillations, bistabilité)
- Développer une interface pour spécifier des contraintes biologiques
- Intégrer des bibliothèques de modèles biologiques standards

**Technologies pertinentes**
- Z3 Theorem Prover (C++/Python bindings)
- BioNetGen pour la modélisation de réseaux biochimiques
- SBML (Systems Biology Markup Language) pour les standards
- Python avec SymPy pour les expressions mathématiques

---

### 11. Résolution automatique du puzzle du Démineur

**Description du problème et contexte**
Le jeu du Démineur se résout automatiquement en modélisant le problème sous forme de CSP. Chaque case inconnue de la grille est représentée par une variable booléenne indiquant la présence ou non d'une mine. Pour chaque case ouverte, le chiffre affiché impose que le nombre de mines dans son voisinage corresponde exactement à cette valeur. La propagation de contraintes permet de déduire systématiquement quelles cases sont sûres et lesquelles contiennent une mine, bien que le problème soit NP-complet dans sa version générale.

**Références multiples**
- **Article de référence** : Bayer & Snyder (2013), [A Constraint-Based Approach to Solving Minesweeper](https://digitalcommons.unl.edu/cseconfwork/170/) - Modélisation CSP complète
- **Complexité** : [Minesweeper is NP-complete](https://www.cs.princeton.edu/~wayne/cs423/lectures/np-complete) (Princeton, 2013) - Preuve de difficulté
- **Implémentation** : [GitHub - Minesweeper_CSP](https://github.com/jgesc/Minesweeper_CSP) - Solveur en programmation par contraintes
- **Tutoriel** : Documentation sur la modélisation avec contraintes de somme sur voisinages

**Approches suggérées**
- Définir une variable booléenne par case inconnue (mine présente ou non)
- Ajouter une contrainte d'égalité sur la somme des variables de voisinage pour chaque case ouverte
- Appliquer la propagation (arc-consistency) pour réduire drastiquement l'espace de recherche
- Utiliser le backtracking intelligent pour les configurations ambiguës

**Technologies pertinentes**
- Python avec python-constraint pour une implémentation rapide
- OR-Tools CP-SAT pour la résolution efficace avec propagation avancée
- Z3 SMT solver comme alternative pour les contraintes de somme
- Interface graphique avec Pygame ou Tkinter pour la visualisation interactive

---

### 12. Ontologies médicales et web sémantique

**Description du problème et contexte**
Les ontologies médicales permettent de structurer et d'interconnecter les connaissances médicales pour le web sémantique. Ce sujet explore la création et l'utilisation de réseaux sémantiques pour améliorer l'accès à l'information médicale et faciliter le raisonnement automatisé.

**Références multiples**
- **BioPortal** : [bioportal.bioontology.org](https://bioportal.bioontology.org/) - Référentiel d'ontologies
- **Gene Ontology** : [geneontology.org](http://geneontology.org/) - Ontologie des gènes et protéines
- **EDAM Ontology** : [edamontology.org](http://edamontology.org/page) - Ontologie pour l'analyse de données
- **CIDO** : [Nature Article](https://www.nature.com/articles/s41597-020-0523-6) - Ontologie des maladies infectieuses

**Approches suggérées**
- Créer une ontologie de domaine médical en OWL/RDF
- Développer un moteur de raisonnement sémantique
- Intégrer plusieurs sources de connaissances médicales
- Implémenter une interface de recherche sémantique

**Technologies pertinentes**
- Protégé pour l'édition d'ontologies
- RDFLib ou Apache Jena pour le traitement sémantique
- SPARQL pour les requêtes sur graphes de connaissances
- Python avec Flask pour l'interface web

---

### 13. Problème des mariages stables (Stable Marriage)

**Description du problème et contexte**
L'appariement bipartite entre deux ensembles (étudiants et postes, ou hommes et femmes dans le problème classique) sur la base de préférences de classement mutuelles. Un matching est stable s'il n'existe pas deux agents qui se préfèreraient mutuellement à leurs attributions actuelles. L'algorithme de Gale & Shapley (1962) garantit une solution stable en temps polynomial via les propositions différées. On peut aussi formuler le problème en CSP : rechercher une affectation (bijection) sans paire bloquante.

**Références multiples**
- **Article fondateur** : Gale & Shapley (1962), _College Admissions and Stability_ - Algorithme des propositions différées
- **Modélisation CP** : Manlove & O'Malley (CP 2008), [Modelling Stable Marriage with CP](https://www.dcs.gla.ac.uk/~davidm/pubs/7981.pdf) - Deux encodages CSP et lien avec Gale-Shapley
- **Ouvrage de référence** : Gusfield & Irving (1989), _The Stable Marriage Problem: Structure and Algorithms_ - Théorie complète
- **Applications réelles** : Hospital-Resident matching utilisé pour l'affectation des internes en médecine

**Approches suggérées**
- Modéliser comme un CSP avec variables d'affectation et contraintes de stabilité
- Implémenter l'algorithme de Gale-Shapley pour comparaison avec approche CP
- Établir l'arc-consistance équivalent à l'élimination des paires incompatibles
- Explorer les variantes (capacités multiples, listes incomplètes, liens indifférents)

**Technologies pertinentes**
- Python avec implémentation classique de Gale-Shapley pour référence
- OR-Tools ou MiniZinc pour la modélisation CSP alternative
- NetworkX pour visualiser les préférences et appariements
- Jupyter Notebook pour analyses comparatives des différentes approches

---

### 14. Blockchain pour les dossiers médicaux COVID-19

**Description du problème et contexte**
La blockchain offre une solution décentralisée et sécurisée pour la gestion des dossiers médicaux COVID-19, garantissant l'intégrité, la traçabilité et le partage contrôlé des informations de santé tout en préservant la confidentialité des patients.

**Références multiples**
- **Blockchain santé** : [Medium Article](https://medium.com/pikciochain/how-is-blockchain-revolutionizing-healthcare-7f6d2a48e561) - Vue d'ensemble des applications
- **Projet IBM** : [Medical Blockchain](https://github.com/IBM/Medical-Blockchain) - Implémentation de référence
- **Passeports immunitaires** : [TechRxiv](https://www.techrxiv.org/articles/preprint/Blockchain-based_Solution_for_COVID-19_Digital_Medical_Passports_and_Immunity_Certificates/12800360/1) - Application COVID spécifique

**Approches suggérées**
- Concevoir une architecture blockchain pour dossiers médicaux
- Implémenter des smart contracts pour le contrôle d'accès
- Développer un système de chiffrement pour la confidentialité
- Créer une interface patient/médecin pour la gestion des données

**Technologies pertinentes**
- Ethereum/Hyperledger Fabric pour la blockchain
- Solidity pour les smart contracts
- IPFS pour le stockage décentralisé
- Web3.js pour l'interface web blockchain

---

### 15. Composition musicale assistée par contraintes

**Description du problème et contexte**
La programmation par contraintes permet d'assister la composition musicale en générant ou complétant automatiquement une pièce musicale tout en respectant les règles harmoniques et de contrepoint de la musique tonale occidentale (style baroque par exemple). Chaque note de chaque voix (soprano, alto, ténor, basse) sur chaque temps est modélisée par une variable dont le domaine est l'ensemble des notes possibles dans la gamme, avec des contraintes musicales strictes pour éviter les erreurs classiques.

**Références multiples**
- **Ouvrage de référence** : Anders Torsten (2012), [Constraint Programming in Music](https://www.wiley.com/en-us/Constraint+Programming+in+Music-p-x000591252) (Wiley) - Théorie complète
- **Publication récente** : [Expressing Musical Ideas with CP](https://www.ijcai.org/proceedings/2024/0858.pdf) (IJCAI 2024) - Modèle de l'harmonie tonale
- **Recherche avancée** : Pachet & Roy (2014), "Non-Conformant Harmonization" - Créativité computationnelle
- **Tutoriel** : [OpenMusic Tutorial on CP in Musical Composition](https://repmus.ircam.fr/openmusic/tutorials/constraint) (IRCAM 2016) - Applications pratiques

**Approches suggérées**
- Définir des variables représentant les notes pour chaque voix et chaque temps
- Spécifier les contraintes correspondant aux règles musicales (harmonie, interdiction des parallèles)
- Implémenter des contraintes de contrepoint (mouvement indépendant des voix, intervalles acceptables)
- Développer un mode interactif permettant au compositeur de fixer certaines notes

**Technologies pertinentes**
- Python avec python-constraint ou OR-Tools pour le moteur de contraintes
- MusicXML et music21 pour la notation et manipulation musicale
- MIDI pour l'export et la lecture audio des compositions générées
- Interface web avec notation interactive (VexFlow, abcjs) pour l'édition

---

### 16. Coloration de graphe et de carte (Graph/Map Coloring)

**Description du problème et contexte**
Attribuer des couleurs à chaque nœud d'un graphe (p. ex. régions d'une carte) de sorte que deux nœuds adjacents n'aient pas la même couleur. On cherche à minimiser le nombre de couleurs utilisées ou à respecter un nombre fixé de couleurs. C'est un problème NP-difficile très connu, utilisé comme exemple classique en CSP. En programmation par contraintes, on crée une variable « couleur » pour chaque nœud avec un domaine de couleurs autorisées, puis on impose pour chaque arête que les deux extrémités aient des valeurs différentes (contrainte binaire).

**Références multiples**
- **Tutoriel AIMMS** : [Color a Map with Constraint Programming](https://how-to.aimms.com/Articles/226/226-color-a-map-with-constraint-programming.html) - Approche CP pour la coloration de carte
- **Blog phabe.ch** : Map coloring problem in MiniZinc (2019) - Implémentation pratique
- **Théorie** : Applegate & Cook (1989), _A Computational Study of Graph Coloring_ - Étude algorithmique
- **Célèbre théorème** : On sait que 4 couleurs suffisent pour n'importe quelle carte planaire

**Approches suggérées**
- Créer une variable « couleur » pour chaque nœud avec domaine de couleurs autorisées
- Imposer des contraintes binaires pour chaque arête (extrémités de couleurs différentes)
- Utiliser la propagation de contraintes (node consistency, arc consistency) pour réduire l'espace de recherche
- Explorer différentes heuristiques d'ordre de variable pour optimiser la résolution

**Technologies pertinentes**
- Python avec OR-Tools ou python-constraint pour la modélisation CSP
- MiniZinc pour une approche déclarative
- NetworkX pour la manipulation et visualisation de graphes
- Graphviz ou Matplotlib pour la représentation visuelle des solutions

---

### 17. Construction de mots-croisés par contraintes

**Description du problème et contexte**
La génération automatique de grilles de mots-croisés peut se formuler en problème de satisfaction de contraintes. On doit remplir une grille noire/blanche avec des mots qui se croisent de façon cohérente (les lettres qui se croisent doivent être identiques). Une approche consiste à pré-définir la grille (emplacements des cases noires) puis à affecter un mot de dictionnaire à chaque « slot » horizontal ou vertical. Les contraintes lient les slots entre eux via les lettres communes.

**Références multiples**
- **Guide CP** : [Generating Crossword Grids Using Constraint Programming](https://pedtsr.ca/2023/generating-crossword-grids-using-constraint-programming.html) - Modélisation pas à pas avec OR-Tools CP-SAT
- **Solver Max** : Exemple de formulation MILP pour composer une grille de mots-croisés
- **Référence historique** : G. Gervet (1995), _Crossword puzzle solving via constraint logic programming_ - Approche CLP
- **Extensions** : On peut ajouter des contraintes de thématique ou maximiser un score

**Approches suggérées**
- Pré-définir la structure de la grille (emplacements des cases noires)
- Affecter un mot de dictionnaire à chaque slot horizontal et vertical
- Lier les slots via des contraintes sur les lettres communes (intersections)
- Utiliser la propagation de contraintes pour éliminer rapidement les combinaisons impossibles

**Technologies pertinentes**
- OR-Tools CP-SAT pour la résolution efficace avec propagation
- MiniZinc pour la modélisation déclarative
- Dictionnaires de mots français/anglais structurés par longueur
- Interface web pour l'édition et la visualisation interactive des grilles

---

### 18. Équilibrage de chaîne d'assemblage (Assembly Line Balancing)

**Description du problème et contexte**
La répartition des tâches d'assemblage sur une séquence de postes de travail de manière à minimiser le nombre de postes (ou à respecter un temps de cycle donné). Chaque tâche a une durée et des précédences, et la somme des durées affectées à un poste ne doit pas dépasser le temps de cycle. Ce problème d'équilibrage est NP-difficile et présente de nombreuses variantes industrielles. Une modélisation classique utilise la programmation par contraintes ou en nombres entiers pour attribuer les tâches à des postes tout en respectant les contraintes d'ordre et de temps.

**Références multiples**
- **Benchmark Hexaly** : [Simple Assembly Line Balancing Problem (SALBP)](https://www.hexaly.com/benchmark/hexaly-vs-gurobi-vs-cpo-simple-assembly-line-balancing-problem-salbp) - Comparatif de solveurs MILP vs CP
- **État de l'art** : Scholl & Becker (2006), _State-of-the-art in assembly line balancing_ - Revue complète
- **Performance** : Des études montrent que même des solveurs génériques (CP Optimizer, Gurobi) peuvent traiter efficacement des cas de grande taille
- **Applications** : Instances industrielles jusqu'à 1000 tâches

**Approches suggérées**
- Modéliser les variables d'affectation de tâches à des postes
- Imposer les contraintes de précédence entre tâches
- Respecter la contrainte de temps de cycle pour chaque poste
- Optimiser le nombre de postes ou l'équilibre de charge

**Technologies pertinentes**
- IBM CP Optimizer ou Hexaly pour les instances industrielles complexes
- OR-Tools CP-SAT ou Gurobi pour une approche hybride CP/MILP
- MiniZinc pour la modélisation déclarative
- Visualisation avec Gantt charts (Matplotlib, Plotly) pour analyser l'équilibrage

---

### 19. Configuration de produit par contraintes / Configuration de systèmes

**Description du problème et contexte**
Ce sujet traite de la problématique de la configuration de produits ou de systèmes complexes (ordinateurs, automobiles, etc.) où le client peut personnaliser son produit en choisissant parmi un ensemble d'options. L'objectif est de garantir que les choix effectués sont compatibles entre eux grâce à l'application d'un grand nombre de règles de compatibilité et d'exclusion. Chaque option est représentée par une variable et les interdépendances sont modélisées par des contraintes logiques.

**Références multiples**
- **Implémentation** : [GitHub - or-tools-product-configurator](https://github.com/foohardt/or-tools-product-configurator) - Configuration de produit avec Google OR-Tools
- **Théorie** : Mittal & Frayman (1989), "Towards a Generic Model of Configuration Tasks" (IJCAI) - Modèle générique
- **Ouvrage** : Hotz, Felfernig & Stumptner (2014), "Configuration Knowledge Representation" - Représentation des connaissances
- **Microsoft** : [Constraints in product configuration models](https://learn.microsoft.com/en-us/dynamics365/supply-chain/pim/build-product-configuration-model#constraints) - Documentation pratique

**Approches suggérées**
- Définir des variables pour chaque composant/option avec leurs domaines possibles
- Imposer des contraintes d'exclusion ou d'implication entre options
- Utiliser un solveur CSP pour propager les contraintes en temps réel
- Développer une interface utilisateur interactive guidant vers des configurations valides

**Technologies pertinentes**
- OR-Tools CP-SAT pour la propagation de contraintes en temps réel
- Python avec python-constraint pour prototypage rapide
- Interface web (React/Vue) pour configuration interactive
- Optimisation multi-critères pour minimiser coût ou maximiser performance

---

### 20. Calendrier sportif (Sports Tournament Scheduling)

**Description du problème et contexte**
L'élaboration du calendrier de rencontres d'un championnat (par ex. tournoi toutes rondes en football), en respectant de multiples contraintes: alternance domicile/extérieur, disponibilités de stades, équité entre équipes (pas plus de X déplacements consécutifs, etc.). L'ordonnancement d'un tournoi « round-robin » peut se modéliser par contrainte avec des variables représentant qui rencontre qui à chaque journée, et des global constraints pour éviter les « breaks » (deux matchs Domicile ou Extérieur de suite).

**Références multiples**
- **Article CP** : Régin (CP 2008), _Minimizing breaks in sports schedules_ - Modèle CP pour tournoi rondes simples
- **Revue** : Schaerf (1999), _Sports scheduling_ - Revue d'approches
- **Compétition** : ITC 2021 Sports Scheduling Track - Compétition utilisant CP et métaheuristiques
- **Preuves théoriques** : La CP a permis de prouver des bornes théoriques, comme le nombre minimal de « breaks » (n–2 pour n équipes)

**Approches suggérées**
- Modéliser avec des variables représentant les rencontres à chaque journée
- Utiliser des global constraints pour gérer les contraintes d'alternance domicile/extérieur
- Implémenter des contraintes d'équité (nombre de déplacements, répartition des adversaires)
- Optimiser selon plusieurs critères (minimisation des breaks, équilibre du calendrier)

**Technologies pertinentes**
- IBM CP Optimizer ou OR-Tools CP-SAT pour les global constraints
- MiniZinc pour la modélisation déclarative de contraintes complexes
- Python pour l'interfaçage et la génération de données
- Visualisation du calendrier avec bibliothèques de planning (FullCalendar, Gantt)

---

### 21. Problème de tournées de véhicules (VRP) / Optimisation de tournées de livraison « vertes »

**Description du problème et contexte**
La planification optimale des tournées d'une flotte de véhicules chargés de livrer des colis ou des marchandises. L'objectif principal est de minimiser la distance parcourue ou le coût total, tout en respectant des contraintes de capacités, fenêtres temporelles, et pour la version « verte », les contraintes liées à l'autonomie des véhicules électriques, la nécessité de passages par des stations de recharge, et la minimisation de l'empreinte carbone.

**Références multiples**
- **Introduction** : [PyVRP documentation](https://pyvrp.org/setup/introduction_to_vrp.html) - Introduction complète au VRP
- **Guide pratique** : [Solving the Vehicle Routing Problem (Routific, 2024)](https://www.routific.com/blog/what-is-the-vehicle-routing-problem) - Approches de résolution
- **Ouvrage** : Toth & Vigo (2014), _Vehicle Routing: Problems, Methods, and Applications_ (SIAM) - Référence complète
- **VRP électrique** : [A Constraint Programming Approach to Electric Vehicle Routing](https://www.researchgate.net/publication/333231312_A_Constraint_Programming_Approach_to_Electric_Vehicle_Routing_with_Time_Windows) - Approche CP pour véhicules électriques

**Approches suggérées**
- Définir des variables pour l'ordre de passage des clients sur chaque tournée
- Implémenter des contraintes de routing, capacité et fenêtres temporelles
- Pour les véhicules électriques, intégrer les contraintes d'autonomie et de recharge
- Utiliser un solveur CSP combiné avec des heuristiques de recherche locale (Large Neighborhood Search)

**Technologies pertinentes**
- OR-Tools CP-SAT pour la modélisation et résolution avec propagation avancée
- MiniZinc pour une approche déclarative
- PyVRP pour des implémentations spécialisées
- Visualisation de tournées avec Folium, Leaflet ou Google Maps API

---

### 22. Argumentation abstraite de Dung

**Description du problème et contexte**
Les frameworks d'argumentation abstraite de Dung (AF) fournissent un cadre mathématique pour représenter et évaluer des arguments en conflit. Le module `arg.dung` de TweetyProject offre une implémentation complète de ce formalisme, permettant de construire des graphes d'arguments et d'attaques (`DungTheory`), et de calculer l'acceptabilité des arguments selon différentes sémantiques (admissible, complète, préférée, stable, fondée, idéale, semi-stable, CF2, etc.).

**Références multiples**
- **Article fondateur** : Dung (1995), _On the Acceptability of Arguments and its Fundamental Role in Nonmonotonic Reasoning_ - Base théorique
- **Ouvrage** : _Abstract Argumentation Frameworks_ (2022) - Théorie complète
- **Recherche** : _Computational Problems in Abstract Argumentation_ (2023) - Aspects algorithmiques
- **TweetyProject** : [Documentation arg.dung](http://tweetyproject.org/api/latest-release/net/sf/tweety/arg/dung/package-summary.html) - Implémentation Java

**Approches suggérées**
- Construire des graphes d'arguments et d'attaques avec `DungTheory`
- Implémenter le calcul d'extensions selon différentes sémantiques (admissible, complète, préférée, stable)
- Développer des algorithmes pour déterminer l'acceptabilité des arguments
- Créer une visualisation interactive des graphes d'argumentation

**Technologies pertinentes**
- TweetyProject `arg.dung` pour la modélisation et le calcul d'extensions
- NetworkX ou Graphviz pour la visualisation de graphes
- Python avec JPype pour l'intégration Java-Python
- Jupyter Notebook pour les démonstrations interactives

---

### 23. Argumentation basée sur les hypothèses (ABA)

**Description du problème et contexte**
L'argumentation basée sur les hypothèses (ABA) est un framework qui représente les arguments comme des déductions à partir d'hypothèses. Le module `arg.aba` de TweetyProject permet de modéliser des systèmes où les arguments sont construits à partir de règles d'inférence et d'hypothèses, avec des mécanismes pour gérer les attaques entre arguments dérivés.

**Références multiples**
- **Théorie** : _Assumption-Based Argumentation_ (2022) - Fondements formels
- **Algorithmes** : _Computational Aspects of Assumption-Based Argumentation_ (2023) - Méthodes de calcul
- **Extension** : _ABA+: Assumption-Based Argumentation with Preferences_ (2022) - Gestion des préférences
- **TweetyProject** : [Documentation arg.aba](http://tweetyproject.org/api/latest-release/net/sf/tweety/arg/aba/package-summary.html) - Implémentation

**Approches suggérées**
- Modéliser des bases de connaissances avec règles d'inférence et hypothèses
- Implémenter la construction d'arguments par déduction
- Développer des mécanismes de détection d'attaques entre arguments
- Calculer l'acceptabilité des arguments selon les sémantiques ABA

**Technologies pertinentes**
- TweetyProject `arg.aba` pour la modélisation ABA
- Logiques non-monotones pour le raisonnement
- Python pour l'interface et la visualisation
- Prolog pour une implémentation alternative des règles

---

### 24. Argumentation structurée (ASPIC+)

**Description du problème et contexte**
ASPIC+ est un framework d'argumentation structurée qui combine la logique formelle avec des mécanismes de gestion des conflits et des préférences. Il permet de construire des arguments à partir de bases de connaissances contenant des axiomes et des règles (strictes et défaisables), et de modéliser différents types d'attaques (rebutting, undercutting, undermining).

**Références multiples**
- **Framework** : _ASPIC+: An Argumentation Framework for Structured Argumentation_ (2022) - Spécification complète
- **Théorie** : _Rationality Postulates for Structured Argumentation_ (2023) - Propriétés formelles
- **Traduction** : _From Natural Language to ASPIC+_ (2022) - Méthodes de formalisation
- **Applications** : Travaux sur l'argumentation juridique et médicale avec ASPIC+

**Approches suggérées**
- Modéliser des bases de connaissances avec axiomes, règles strictes et règles défaisables
- Implémenter la construction d'arguments structurés
- Gérer les préférences entre règles et arguments
- Analyser les différents types d'attaques (rebutting, undercutting, undermining)

**Technologies pertinentes**
- Implémentation ASPIC+ (bibliothèques existantes ou développement custom)
- Logique du premier ordre pour la représentation des connaissances
- Python ou Java pour l'implémentation
- Visualisation des arguments structurés et de leurs relations

---

### 25. Abstract Dialectical Frameworks (ADF)

**Description du problème et contexte**
Les ADF généralisent les frameworks d'argumentation abstraite de Dung en associant à chaque argument une condition d'acceptation. Le module `arg.adf` de TweetyProject implémente ce formalisme avancé où chaque argument est associé à une formule propositionnelle (sa condition d'acceptation) qui détermine son statut en fonction de l'état des autres arguments. Cette approche permet de modéliser des dépendances complexes comme le support, l'attaque conjointe, ou des combinaisons arbitraires de relations.

**Références multiples**
- **Article fondateur** : Brewka et al. (2013), _Abstract Dialectical Frameworks_ - Définition formelle
- **Implémentation** : _Implementing KR Approaches with Tweety_ (2018) - Guide pratique
- **TweetyProject** : [Documentation arg.adf](http://tweetyproject.org/api/latest-release/net/sf/tweety/arg/adf/package-summary.html) - API complète
- **Solveurs** : Intégration avec solveurs SAT incrémentaux pour le calcul efficace

**Approches suggérées**
- Définir des arguments avec conditions d'acceptation personnalisées (formules propositionnelles)
- Modéliser des relations complexes (support, attaque conjointe, dépendances conditionnelles)
- Utiliser des solveurs SAT incrémentaux pour calculer les extensions
- Visualiser les ADF avec leurs conditions d'acceptation

**Technologies pertinentes**
- TweetyProject `arg.adf` pour la modélisation
- Solveurs SAT (SAT4J, Lingeling) pour le calcul d'extensions
- Logique propositionnelle pour les conditions d'acceptation
- Visualisation de graphes avec annotations de formules

---

### 26. Classification des sophismes

**Description du problème et contexte**
Les sophismes sont des erreurs de raisonnement qui peuvent sembler valides mais qui violent les principes de la logique. Une taxonomie structurée des sophismes est essentielle pour développer des systèmes de détection automatique et d'analyse critique des arguments. Ce projet vise à enrichir et structurer la classification des sophismes en intégrant des approches historiques et contemporaines.

**Références multiples**
- **Ouvrage classique** : _Fallacies: Classical and Contemporary Readings_ (édition mise à jour, 2022) - Taxonomie complète
- **Guide moderne** : _Logical Fallacies: The Definitive Guide_ (2023) - Définitions et exemples
- **Détection** : _Automated Detection of Fallacies in Arguments_ (2022) - Approches computationnelles
- **Base de données** : Corpus annotés de sophismes pour l'apprentissage automatique

**Approches suggérées**
- Développer une taxonomie hiérarchique des sophismes (formels, informels, rhétoriques)
- Créer une base de données d'exemples annotés pour chaque type de sophisme
- Implémenter des règles de détection basées sur des patterns linguistiques et logiques
- Utiliser l'apprentissage automatique pour la classification automatique

**Technologies pertinentes**
- NLP (spaCy, NLTK) pour l'analyse linguistique
- Machine Learning (scikit-learn, transformers) pour la classification
- Base de données (SQL, MongoDB) pour le stockage des exemples
- Interface web pour la visualisation et l'annotation

---

### 27. Taxonomie des schémas argumentatifs

**Description du problème et contexte**
Les schémas argumentatifs sont des modèles récurrents de raisonnement utilisés dans l'argumentation quotidienne. Les travaux de Walton identifient plus de 60 schémas argumentatifs courants, chacun avec ses questions critiques associées. Ce projet vise à développer une taxonomie complète et structurée de ces schémas pour faciliter leur identification et leur analyse automatique.

**Références multiples**
- **Ouvrage de référence** : Walton, Reed & Macagno, _Argumentation Schemes_ (édition mise à jour, 2022) - Catalogue complet
- **Identification** : _Automatic Identification of Argument Schemes_ (2023) - Méthodes computationnelles
- **Modélisation** : _A Computational Model of Argument Schemes_ (2022) - Formalisation
- **Applications** : Travaux sur l'utilisation des schémas dans l'analyse de débats

**Approches suggérées**
- Structurer une taxonomie hiérarchique des schémas argumentatifs de Walton
- Associer à chaque schéma ses questions critiques et des exemples concrets
- Développer des méthodes de reconnaissance automatique de schémas dans les textes
- Créer une interface pour explorer et interroger la taxonomie

**Technologies pertinentes**
- Ontologies (OWL, Protégé) pour la structuration formelle
- NLP pour l'extraction et la classification de schémas
- Base de connaissances (Neo4j, RDF) pour les relations entre schémas
- Visualisation interactive des schémas et de leurs relations

---

### 28. Agent de détection de sophismes et biais cognitifs

**Description du problème et contexte**
La détection des sophismes et des biais cognitifs est essentielle pour évaluer la qualité argumentative et lutter contre la désinformation. Ce sujet vise à améliorer l'agent Informal pour détecter plus précisément différents types de sophismes et fournir des explications claires sur leur nature, tout en intégrant des capacités d'analyse des biais cognitifs pour identifier les mécanismes psychologiques exploités dans les arguments fallacieux.

**Références multiples**
- **Détection automatisée** : _Automated Fallacy Detection_ (2022) - Méthodes computationnelles
- **Analyse rhétorique** : _Computational Approaches to Rhetorical Analysis_ (2023) - Techniques d'analyse
- **Explicabilité** : _Explainable Fallacy Detection_ (2022) - Systèmes explicables
- **Biais cognitifs** : _Cognitive Biases in Argumentation_ (2024) - Mécanismes psychologiques
- **Désinformation** : _Psychological Mechanisms of Misinformation_ (2023) - Manipulation informationnelle

**Approches suggérées**
- Développer des techniques spécifiques pour chaque type de sophisme
- Intégrer l'ontologie des sophismes pour améliorer la classification
- Créer un système d'explication des détections avec contexte psychologique
- Analyser l'impact persuasif des sophismes détectés
- Intégrer avec des systèmes de lutte contre la désinformation

**Technologies pertinentes**
- NLP avancé (spaCy, transformers) pour l'analyse linguistique
- Classification multi-classes avec deep learning
- Modèles de psychologie cognitive pour l'analyse des biais
- Systèmes d'explication IA (LIME, SHAP) pour la transparence

---

### 29. Agent de génération de contre-arguments

**Description du problème et contexte**
La génération de contre-arguments permet d'évaluer la robustesse des arguments en produisant automatiquement des réfutations pertinentes et solides. Ce système peut aider dans les débats, l'analyse critique et l'amélioration de la qualité argumentative en identifiant les vulnérabilités des arguments.

**Références multiples**
- **Génération automatique** : _Automated Counter-Argument Generation_ (2022) - Méthodes de génération
- **Argumentation stratégique** : _Strategic Argumentation in Dialogue_ (2023) - Stratégies de réfutation
- **Génération contrôlée** : _Controlled Text Generation for Argumentation_ (2022) - Techniques de contrôle
- **Évaluation** : _Quality Assessment of Generated Arguments_ (2023) - Métriques d'évaluation

**Approches suggérées**
- Implémenter différentes stratégies de contre-argumentation basées sur les frameworks formels
- Analyser les vulnérabilités argumentatives pour cibler les points faibles
- Développer des techniques de génération de texte contrôlée
- Créer un système d'évaluation de la qualité des contre-arguments générés

**Technologies pertinentes**
- LLMs (GPT, Claude) pour la génération de texte naturel
- Frameworks d'argumentation (Tweety) pour l'analyse formelle
- Fine-tuning de modèles sur corpus de débats
- Évaluation automatique de la pertinence et de la force des arguments

---

### 30. Intégration de LLMs locaux légers

**Description du problème et contexte**
Les LLMs locaux permettent une analyse plus rapide et confidentielle sans dépendance aux API externes. Ce projet explore l'utilisation de modèles de langage locaux de petite taille (comme Qwen 3) pour effectuer l'analyse argumentative, en optimisant pour l'inférence rapide tout en maintenant une qualité d'analyse acceptable.

**Références multiples**
- **Qwen 3** : Documentation officielle - Modèles légers récents
- **Optimisation** : _Efficient Inference for Large Language Models_ (2023) - Techniques d'optimisation
- **Quantization** : _Model Quantization Techniques_ (2024) - Réduction de taille
- **Benchmarks** : HELM - Évaluation comparative des performances
- **Distillation** : _Knowledge Distillation for LLMs_ (2023) - Transfert de connaissances

**Approches suggérées**
- Intégrer des modèles légers (Qwen 3) avec llama.cpp
- Appliquer des techniques de quantization (GGUF format)
- Optimiser l'inférence pour des performances temps réel
- Comparer les performances avec les modèles via API cloud

**Technologies pertinentes**
- llama.cpp pour l'inférence optimisée
- GGUF format pour les modèles quantifiés
- Python bindings pour l'intégration
- Techniques de prompt engineering pour maximiser la qualité

---

### 31. Fact-checking automatisé et détection de désinformation

**Description du problème et contexte**
La vérification des faits et la détection de désinformation sont essentielles pour évaluer la solidité factuelle des arguments et protéger l'intégrité du débat public. Ce système devrait pouvoir extraire les affirmations vérifiables, rechercher des informations pertinentes, évaluer la fiabilité des sources, identifier les techniques de manipulation informationnelle, et analyser la propagation de la désinformation.

**Références multiples**
- **Fact-checking** : _Automated Fact-Checking: Current Status and Future Directions_ (2022) - État de l'art
- **Extraction** : _Claim Extraction and Verification_ (2023) - Méthodes d'extraction
- **Campagnes coordonnées** : _Detecting Coordinated Disinformation Campaigns_ (2024) - Détection de patterns
- **Désordre informationnel** : _Information Disorder: Toward an interdisciplinary framework_ (2023) - Cadre théorique
- **Crédibilité** : _Source Credibility Assessment in the Era of Fake News_ (2024) - Évaluation des sources

**Approches suggérées**
- Extraire automatiquement les affirmations vérifiables dans les textes
- Créer un moteur de recherche spécialisé pour trouver des sources fiables
- Implémenter un système d'évaluation de la fiabilité des sources
- Détecter les patterns typiques de désinformation et fake news
- Analyser la propagation de l'information à travers différents canaux

**Technologies pertinentes**
- NLP avancé pour l'extraction d'affirmations
- Information retrieval pour la recherche de sources
- Machine learning pour l'évaluation de fiabilité
- Network analysis pour la propagation d'information
- API de bases de données de fact-checking existantes

---

### 32. Développement d'un serveur MCP pour l'analyse argumentative

**Description du problème et contexte**
Le Model Context Protocol (MCP) permet d'exposer des capacités d'IA à d'autres applications de manière standardisée. Ce projet vise à publier le travail collectif sous forme d'un serveur MCP utilisable dans des applications comme Roo, Claude Desktop ou Semantic Kernel, rendant l'analyse argumentative accessible à un large écosystème d'outils.

**Références multiples**
- **Spécification MCP** : Model Context Protocol (version 2023-2024) - Protocole officiel
- **Interopérabilité** : _Building Interoperable AI Systems_ (2023) - Systèmes interconnectés
- **API Design** : _RESTful API Design: Best Practices_ (2022) - Bonnes pratiques
- **Documentation** : Exemples d'implémentation MCP existants

**Approches suggérées**
- Implémenter les spécifications MCP pour exposer les fonctionnalités d'analyse
- Créer des outils MCP pour extraction, détection de sophismes, évaluation
- Développer des ressources MCP donnant accès aux taxonomies et exemples
- Assurer la compatibilité avec différentes applications clientes

**Technologies pertinentes**
- MCP SDK pour l'implémentation du protocole
- JSON Schema pour la définition des outils et ressources
- API REST/WebSocket pour la communication
- Documentation OpenAPI/Swagger pour l'API

---

### 33. Serveur MCP pour les frameworks d'argumentation Tweety

**Description du problème et contexte**
Les frameworks d'argumentation de Tweety offrent des fonctionnalités puissantes mais leur utilisation nécessite une connaissance approfondie de l'API Java. Un serveur MCP dédié permettrait d'exposer ces fonctionnalités de manière standardisée et accessible, facilitant l'utilisation des différents frameworks (Dung, bipolaire, pondéré, ADF, etc.) depuis n'importe quelle application compatible MCP.

**Références multiples**
- **Spécification MCP** : Model Context Protocol (version 2023-2024) - Protocole
- **TweetyProject** : Documentation de l'API - Frameworks d'argumentation
- **Interopérabilité** : _Building Interoperable AI Systems_ (2023) - Systèmes interconnectés
- **Java-Python** : JPype documentation - Bridge Java-Python

**Approches suggérées**
- Développer un serveur MCP spécifique pour les modules `arg.*` de Tweety
- Exposer des outils pour construction, analyse et visualisation de frameworks
- Implémenter des ressources MCP pour les sémantiques d'acceptabilité
- Fournir des exemples d'intégration avec différentes applications

**Technologies pertinentes**
- MCP SDK pour le serveur
- JPype pour l'interface Java-Python
- TweetyProject pour les frameworks d'argumentation
- JSON Schema pour les définitions d'outils

---

### 34. Interface web pour l'analyse argumentative

**Description du problème et contexte**
Une interface web intuitive facilite l'utilisation du système d'analyse argumentative par un large public. Cette interface devrait permettre de visualiser et d'interagir avec les analyses argumentatives de manière fluide, avec des fonctionnalités de navigation, filtrage, recherche et annotation pour explorer les structures argumentatives complexes.

**Références multiples**
- **Visualisation d'arguments** : _Argument Visualization Tools in the Classroom_ (2022) - Applications pédagogiques
- **UX pour systèmes complexes** : _User Experience Design for Complex Systems_ (2023) - Design patterns
- **Inspiration** : Interfaces de Kialo ou Arguman (études de cas, 2022) - Exemples existants
- **Interaction** : _Interactive Argument Analysis Interfaces_ (2023) - Techniques d'interaction

**Approches suggérées**
- Créer une interface moderne avec React/Vue.js/Angular
- Implémenter des visualisations interactives avec D3.js ou Cytoscape.js
- Développer des fonctionnalités de navigation et d'exploration intuitive
- Intégrer des capacités d'annotation et de commentaire collaboratif

**Technologies pertinentes**
- Framework frontend moderne (React, Vue, Angular)
- Bibliothèques de visualisation (D3.js, Cytoscape.js)
- Design systems (Material UI, Tailwind CSS)
- WebSockets pour les interactions temps réel

---

### 35. Visualisation avancée de graphes d'argumentation

**Description du problème et contexte**
La visualisation des graphes d'argumentation et des réseaux de désinformation aide à comprendre les relations complexes entre arguments et à identifier les patterns de propagation. Ce projet vise à développer des outils de visualisation avancés pour différents frameworks d'argumentation, avec des algorithmes de layout optimisés et des techniques de visualisation cognitive.

**Références multiples**
- **COMMA** : _Computational Models of Argument_ (conférences 2022-2024) - État de l'art
- **Visualisation** : Travaux de Floris Bex sur la visualisation d'arguments (2022-2023)
- **Graph Drawing** : _Graph Drawing: Algorithms for the Visualization of Graphs_ (2023) - Algorithmes
- **Désinformation** : _Visual Analytics for Disinformation Detection_ (2024) - Analyse visuelle
- **Cognition** : _Cognitive Visualization Techniques for Complex Arguments_ (2023) - Techniques cognitives

**Approches suggérées**
- Implémenter des algorithmes de layout optimisés pour graphes argumentatifs
- Développer des visualisations temporelles pour la propagation d'information
- Créer des techniques de visualisation cognitive pour faciliter la compréhension
- Intégrer avec des systèmes de détection de désinformation

**Technologies pertinentes**
- Bibliothèques de visualisation (Sigma.js, Cytoscape.js, vis.js, D3.js)
- Algorithmes de layout de graphes (force-directed, hierarchical)
- Visualisation temporelle pour l'analyse de propagation
- Techniques d'interaction avancées (zoom, pan, filtering)

---

### 36. Système de débat assisté par IA

**Description du problème et contexte**
Un système de débat assisté par IA peut aider à structurer et améliorer les échanges argumentatifs en temps réel. Cette application complète permettrait à des utilisateurs de débattre avec l'assistance d'agents IA qui analysent leurs arguments, identifient les faiblesses, suggèrent des contre-arguments, et aident à structurer les débats de manière constructive.

**Références multiples**
- **COMMA** : _Computational Models of Argument_ - Base théorique
- **Plateforme Kialo** : Étude de cas - Débat structuré en ligne
- **Technologies d'argumentation** : Recherches de Chris Reed sur les technologies d'argumentation
- **Débat IA** : _AI-Assisted Argumentation and Debate_ (2023) - Applications pratiques

**Approches suggérées**
- Utiliser des LLMs pour l'analyse et la génération d'arguments
- Intégrer les frameworks d'argumentation Tweety pour l'évaluation formelle
- Développer une interface web interactive pour les débats
- Implémenter des mécanismes d'assistance contextuelle

**Technologies pertinentes**
- LLMs pour génération et analyse d'arguments
- TweetyProject pour évaluation formelle
- Framework frontend pour interface interactive
- WebSockets pour communication temps réel

---

### 37. Plateforme éducative d'apprentissage de l'argumentation

**Description du problème et contexte**
L'éducation à l'argumentation et à la pensée critique est essentielle pour former des citoyens capables de naviguer dans un environnement informationnel complexe. Cette plateforme complète intégrerait des parcours d'apprentissage personnalisés, des tutoriels interactifs, des exercices pratiques, des évaluations adaptatives, et des mécanismes de gamification pour favoriser l'engagement.

**Références multiples**
- **Analytics** : _Learning Analytics for Argumentation Skills_ (2023) - Suivi des compétences
- **Gamification** : _Gamification in Critical Thinking Education_ (2024) - Motivation et engagement
- **Apprentissage adaptatif** : _Adaptive Learning Systems: Design and Implementation_ (2023) - Personnalisation
- **Compétences** : _Measuring and Developing Argumentation Skills_ (2022) - Évaluation
- **Désinformation** : _Educational Interventions Against Misinformation_ (2024) - Lutte contre fake news

**Approches suggérées**
- Créer des tutoriels interactifs sur les sophismes et biais cognitifs
- Développer des exercices pratiques avec feedback automatisé
- Implémenter un système d'évaluation des compétences argumentatives
- Intégrer des mécanismes de gamification (badges, niveaux, défis)
- Créer un tableau de bord de suivi des apprentissages

**Technologies pertinentes**
- LMS (Learning Management System) ou développement custom
- Gamification engine (badges, points, leaderboards)
- Analytics pour le suivi des progressions
- Système d'évaluation automatisée basé sur IA

---

### 38. Système d'analyse de débats politiques

**Description du problème et contexte**
L'analyse des débats politiques et la surveillance des médias permettent d'évaluer objectivement la qualité argumentative des discours et de détecter les campagnes de désinformation dans l'espace public. Ce système complet analyserait les arguments, sophismes et stratégies rhétoriques utilisées, fournirait une évaluation factuelle, détecterait les tendances émergentes et analyserait la propagation des narratifs à travers différents médias.

**Références multiples**
- **Analyse politique** : _Computational Approaches to Analyzing Political Discourse_ de Hovy et Lim
- **Fact-checking** : Projets comme FactCheck.org ou PolitiFact (études de cas, 2022)
- **Automatisation** : _Automated Fact-Checking: Current Status and Future Directions_ (2022)
- **Surveillance médiatique** : _Media Monitoring in the Digital Age_ (2024)
- **Comportement inauthentique** : _Detecting Coordinated Inauthentic Behavior in Social Media_ (2023)
- **Diffusion** : _Temporal Analysis of Information Diffusion_ (2024)

**Approches suggérées**
- Développer une analyse de débats en temps réel
- Créer une plateforme de surveillance médiatique multi-sources
- Implémenter la détection de sophismes, biais et stratégies rhétoriques
- Intégrer le fact-checking automatisé des affirmations
- Analyser la propagation des arguments dans les médias
- Détecter les campagnes coordonnées de désinformation

**Technologies pertinentes**
- NLP en temps réel pour l'analyse de discours
- Fact-checking automatisé avec recherche d'information
- Analyse de sentiment et de rhétorique
- Détection de campagnes coordonnées avec network analysis
- Visualisation de propagation d'information

---

### 39. ArgumentuShield: Protection cognitive contre la désinformation

**Description du problème et contexte**
Face à la sophistication croissante des techniques de désinformation, ce système innovant vise à renforcer les défenses cognitives des individus contre la manipulation informationnelle. ArgumentuShield intègre des méthodes d'inoculation cognitive, des outils personnalisés d'analyse critique adaptés aux vulnérabilités spécifiques de chaque utilisateur, des interfaces qui favorisent la réflexion critique, et des mécanismes d'apprentissage continu adaptatifs.

**Références multiples**
- **Inoculation** : Roozenbeek & van der Linden (2019), _The fake news game: actively inoculating against the risk of misinformation_
- **Correction** : Lewandowsky et al. (2012), _Misinformation and Its Correction: Continued Influence and Successful Debiasing_
- **Techniques** : Cook et al. (2017), _Neutralizing misinformation through inoculation: Exposing misleading argumentation techniques_
- **Psychologie** : _Cognitive Psychology of Misinformation Resistance_ (2023)

**Approches suggérées**
- Développer des méthodes d'inoculation cognitive contre les techniques de manipulation
- Créer des outils personnalisés analysant les vulnérabilités spécifiques des utilisateurs
- Concevoir des interfaces qui favorisent la réflexion critique sans friction
- Implémenter des mécanismes d'apprentissage continu adaptatifs
- Intégrer ArgumentuMind pour la modélisation cognitive

**Technologies pertinentes**
- Machine learning pour l'analyse des vulnérabilités personnelles
- Techniques d'inoculation basées sur la psychologie cognitive
- Interfaces adaptatives favorisant la réflexion
- Apprentissage par renforcement pour l'adaptation continue
- Intégration avec systèmes de détection de désinformation

---

## 🏦 Sujets spécialisés Finance

Les sujets suivants sont spécifiquement conçus pour les étudiants de la filière Finance, appliquant les techniques d'IA exploratoire et symbolique aux problématiques financières.

---

### 40. Optimisation de portefeuille avec contraintes réelles (CSP/MILP)

**Description du problème et contexte**
L'optimisation de portefeuille classique (Markowitz) ignore les contraintes pratiques : coûts de transaction, lots minimums, nombre maximal d'actifs, contraintes sectorielles. Ce problème NP-difficile se formule naturellement en programmation mixte entière (MILP) ou CSP, permettant d'intégrer des contraintes de cardinalité (max N actifs), de diversification sectorielle et de rééquilibrage sous coûts de transaction.

**Références multiples**
- **Revue** : [A recent review on optimisation methods applied to credit scoring models](https://www.emerald.com/jefas/article/28/56/352/206236/A-recent-review-on-optimisation-methods-applied-to) - Journal of Economics, Finance and Administrative Science
- **MILP** : [Linear and Mixed Integer Programming for Portfolio Optimization](https://www.researchgate.net/publication/283777316_Linear_and_Mixed_Integer_Programming_for_Portfolio_Optimization) - ResearchGate
- **Rebalancing** : [Constructing Optimal Portfolio Rebalancing Strategies](https://link.springer.com/article/10.1007/s10614-024-10555-y) - Computational Economics 2024
- **Contraintes CVaR** : [Constrained Max Drawdown: a Fast and Robust Portfolio Optimization](https://arxiv.org/html/2401.02601v1) - arXiv 2024

**Approches suggérées**
- Modéliser les variables de décision (poids des actifs) avec contraintes de cardinalité via variables binaires
- Implémenter les contraintes de diversification sectorielle et géographique
- Intégrer les coûts de transaction dans la fonction objectif
- Comparer formulations MILP (Gurobi) et CSP (OR-Tools) sur données réelles

**Technologies pertinentes**
- Python avec Gurobi, CPLEX ou OR-Tools pour l'optimisation
- cvxpy pour la modélisation convexe avec contraintes
- Pandas et yfinance pour les données financières historiques
- Matplotlib/Plotly pour la visualisation des frontières efficientes

---

### 41. Stratégies de trading par algorithmes génétiques

**Description du problème et contexte**
L'optimisation de stratégies de trading algorithmique nécessite d'explorer un espace combinatoire immense de règles et paramètres. Les algorithmes génétiques permettent d'évoluer des populations de stratégies, combinant indicateurs techniques et règles de décision, tout en évitant le surapprentissage grâce à des techniques de validation robustes.

**Références multiples**
- **Revue comparative** : [Robust Metaheuristic Optimization for Algorithmic Trading](https://www.mdpi.com/2227-7390/14/1/69) - MDPI Mathematics 2024
- **GA pour trading** : [Applicability of genetic algorithms for stock market prediction: A systematic survey](https://www.sciencedirect.com/science/article/abs/pii/S1574013724000364) - ScienceDirect 2024
- **Directional Changes** : [A genetic algorithm for multi-threshold trading strategies](https://link.springer.com/article/10.1007/s10462-025-11419-z) - Artificial Intelligence Review 2025
- **Vectorial GP** : [Evolving Financial Trading Strategies with Vectorial Genetic Programming](https://arxiv.org/html/2504.05418v1) - arXiv 2025

**Approches suggérées**
- Encoder les stratégies comme chromosomes (indicateurs, seuils, règles)
- Définir une fonction fitness multi-objectifs (rendement, Sharpe ratio, max drawdown)
- Implémenter la sélection, croisement et mutation adaptés au domaine financier
- Utiliser le walk-forward testing pour éviter le curve-fitting

**Technologies pertinentes**
- Python avec DEAP ou PyGAD pour les algorithmes génétiques
- Backtrader ou Zipline pour le backtesting de stratégies
- TA-Lib pour les indicateurs techniques
- QuantConnect pour validation sur données réelles

---

### 42. Détection de fraude financière par graphes

**Description du problème et contexte**
La détection de fraude dans les transactions financières exploite la structure de graphe des relations entre comptes. Les réseaux de neurones sur graphes (GNN) et les algorithmes de détection d'anomalies permettent d'identifier des patterns de fraude sophistiqués (fraude en réseau, blanchiment d'argent) invisibles aux méthodes traditionnelles basées sur des règles.

**Références multiples**
- **Revue systématique** : [Financial fraud detection using graph neural networks: A systematic review](https://www.sciencedirect.com/science/article/abs/pii/S0957417023026581) - Expert Systems with Applications
- **NVIDIA Blueprint** : [Supercharging Fraud Detection with Graph Neural Networks](https://developer.nvidia.com/blog/supercharging-fraud-detection-in-financial-services-with-graph-neural-networks/) - NVIDIA 2024
- **FraudGT** : [A Simple, Effective, and Efficient Graph Transformer](https://jshun.csail.mit.edu/FraudGT.pdf) - ICAIF 2024
- **Curated papers** : [Graph fraud detection papers](https://github.com/safe-graph/graph-fraud-detection-papers) - GitHub

**Approches suggérées**
- Modéliser les transactions comme graphe orienté (comptes = nœuds, transactions = arêtes)
- Implémenter des algorithmes de détection d'anomalies sur graphes
- Utiliser les GNN pour l'apprentissage de représentations des nœuds
- Développer des métriques d'évaluation adaptées au déséquilibre des classes

**Technologies pertinentes**
- Python avec PyTorch Geometric ou DGL pour les GNN
- NetworkX pour l'analyse de graphes classique
- Neo4j pour le stockage et requêtes sur graphes
- Scikit-learn pour les métriques et baseline ML

---

### 43. Système expert de conformité réglementaire (AML/KYC)

**Description du problème et contexte**
La conformité anti-blanchiment (AML) et Know Your Customer (KYC) repose sur des règles complexes définies par les régulateurs. Un système expert permet de formaliser ces règles en logique, d'automatiser la détection de transactions suspectes et de fournir des explications auditables pour les décisions de signalement.

**Références multiples**
- **AML/AI** : [Anti-money laundering supervision by intelligent algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0167404823002547) - Computers & Security
- **Oracle AML** : [Anti-Money Laundering AI Explained](https://www.oracle.com/financial-services/aml-ai/) - Oracle
- **Règles AML** : [AML Transaction Monitoring Rules and Best Practices](https://www.sanctions.io/blog/anti-money-laundering-aml-transaction-monitoring-rules-and-best-practices) - Sanctions.io
- **ML pour AML** : [Machine Learning and AI in AML Compliance](https://sumsub.com/blog/aml-machine-learning/) - Sumsub 2024

**Approches suggérées**
- Formaliser les règles de détection AML en logique (Prolog ou moteur de règles)
- Implémenter les scénarios de détection (structuration, transactions inhabituelles)
- Développer un système d'explication des alertes générées
- Intégrer des techniques ML pour réduire les faux positifs

**Technologies pertinentes**
- Python avec PyKE ou CLIPS pour le moteur de règles
- Prolog pour la modélisation logique des règles
- Drools ou business rules engine pour l'industrialisation
- Interface web pour la gestion des alertes et investigations

---

### 44. Ontologie financière et conformité sémantique (FIBO)

**Description du problème et contexte**
L'ontologie FIBO (Financial Industry Business Ontology) standardise les concepts financiers pour la conformité réglementaire. Ce sujet explore l'utilisation des technologies du web sémantique pour automatiser la vérification de conformité, en alignant les données d'entreprise avec les exigences réglementaires via le raisonnement ontologique.

**Références multiples**
- **FIBO** : [Financial Industry Business Ontology](https://github.com/edmcouncil/fibo) - GitHub EDM Council
- **FinRegOnt** : [Semantic Compliance in Finance](https://finregont.com/) - Financial Regulation Ontology
- **Knowledge Graphs Finance** : [The Power of Ontologies and Knowledge Graphs in Finance](https://graphwise.ai/blog/the-power-of-ontologies-and-knowledge-graphs-practical-examples-from-the-financial-industry/) - Graphwise 2024
- **Tutoriel** : [Introduction to Financial Regulation Ontology](https://finregont.com/tutorial/) - FinRegOnt

**Approches suggérées**
- Explorer et étendre une sous-partie de FIBO pertinente (ex: instruments financiers)
- Modéliser des règles réglementaires en OWL avec classes définies
- Implémenter un raisonneur pour inférer la conformité automatiquement
- Développer des requêtes SPARQL pour l'audit de conformité

**Technologies pertinentes**
- Protégé pour l'édition d'ontologies OWL
- Apache Jena ou RDFLib pour le traitement RDF/SPARQL
- HermiT ou Pellet pour le raisonnement ontologique
- Python pour l'intégration et l'interface utilisateur

---

### 45. Vérification formelle de smart contracts financiers (SMT)

**Description du problème et contexte**
Les smart contracts DeFi gèrent des milliards de dollars et sont vulnérables aux bugs exploitables. Les solveurs SMT (Satisfiability Modulo Theories) permettent de vérifier formellement les propriétés de sécurité des contrats financiers (absence d'overflow, invariants de solde, conditions de liquidation) avant leur déploiement.

**Références multiples**
- **Ethereum Formal Verification** : [Formal verification of smart contracts](https://ethereum.org/developers/docs/smart-contracts/formal-verification/) - ethereum.org
- **SMTChecker** : [Solidity SMTChecker and Formal Verification](https://docs.soliditylang.org/en/latest/smtchecker.html) - Solidity Documentation
- **ESBMC-Solidity** : [An SMT-Based Model Checker for Solidity](https://arxiv.org/pdf/2111.13117) - arXiv
- **Memory Splitting** : [Practical Verification of Smart Contracts](https://dl.acm.org/doi/10.1145/3689796) - OOPSLA 2024

**Approches suggérées**
- Modéliser un contrat financier simple (token ERC-20, AMM basique) en Solidity
- Spécifier les propriétés de sécurité (invariants de balance, access control)
- Utiliser Z3 ou SMTChecker pour prouver/réfuter ces propriétés
- Documenter les vulnérabilités détectées et corrections apportées

**Technologies pertinentes**
- Solidity pour l'écriture des smart contracts
- Z3 Theorem Prover pour la vérification SMT
- Foundry ou Hardhat pour le développement et tests
- Mythril ou Slither pour l'analyse statique complémentaire

---

### 46. Graphe de connaissances pour la gestion des risques financiers

**Description du problème et contexte**
Les graphes de connaissances financiers (Financial Event Knowledge Graphs) permettent de modéliser les relations complexes entre entités (entreprises, personnes, événements) pour identifier et propager les risques. Cette approche neuro-symbolique combine le raisonnement sur graphes avec l'apprentissage automatique pour la prédiction de risques.

**Références multiples**
- **FEEKG** : [Risk identification through knowledge Association: A financial event evolution knowledge graph](https://www.sciencedirect.com/science/article/abs/pii/S0957417424008650) - Expert Systems with Applications 2024
- **Supply Chain Risk** : [Knowledge graph reasoning for supply chain risk management](https://www.tandfonline.com/doi/full/10.1080/00207543.2022.2100841) - Taylor & Francis
- **FinReflectKG** : [Agentic Construction and Evaluation of Financial Knowledge Graphs](https://arxiv.org/pdf/2508.17906) - arXiv
- **SEMANTiCS 2024** : [Knowledge Graphs in the Age of LLMs and Neuro-Symbolic AI](https://ebooks.iospress.nl/volume/knowledge-graphs-in-the-age-of-language-models-and-neuro-symbolic-ai-proceedings-of-the-20th-international-conference-on-semantic-systems) - IOS Press

**Approches suggérées**
- Construire un graphe de connaissances à partir de données financières publiques
- Modéliser les relations entité-événement-risque en multi-couches
- Implémenter des algorithmes de propagation de risque sur le graphe
- Utiliser des GNN pour la prédiction de liens et de risques émergents

**Technologies pertinentes**
- Neo4j ou Amazon Neptune pour le stockage du graphe
- Python avec PyKEEN ou DGL-KE pour l'apprentissage sur graphes
- spaCy/Stanza pour l'extraction d'entités nommées financières
- Dash/Streamlit pour la visualisation interactive du graphe

---

### 47. Scoring de crédit équitable par optimisation sous contraintes

**Description du problème et contexte**
Le scoring de crédit par ML pose des problèmes d'équité (biais contre certains groupes). La recherche récente propose d'intégrer des contraintes d'équité directement dans l'optimisation du modèle, formulant le problème comme une optimisation sous contraintes où les métriques d'équité (demographic parity, equalized odds) sont des contraintes à satisfaire.

**Références multiples**
- **Fairness Constraints** : [Fairness in Credit Scoring: Assessment, Implementation and Profit Implications](https://arxiv.org/pdf/2103.01907) - arXiv
- **GA Credit Strategy** : [Finding an Optimal Approval Strategy using Genetic Algorithm](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4461370) - SSRN
- **ML Credit Review** : [Machine learning powered financial credit scoring: a systematic review](https://link.springer.com/article/10.1007/s10462-025-11416-2) - AI Review 2025
- **Threshold Optimization** : [Optimizing Acceptance Threshold using Reinforcement Learning](https://github.com/MykolaHerasymovych/Optimizing-Acceptance-Threshold-in-Credit-Scoring-using-Reinforcement-Learning) - GitHub

**Approches suggérées**
- Formuler le scoring comme problème d'optimisation avec contraintes d'équité
- Implémenter différentes métriques d'équité comme contraintes
- Comparer les approches (pre-processing, in-processing, post-processing)
- Analyser le trade-off équité vs. performance prédictive

**Technologies pertinentes**
- Python avec Fairlearn ou AIF360 pour les contraintes d'équité
- scikit-learn pour les modèles de base
- cvxpy pour l'optimisation sous contraintes
- SHAP/LIME pour l'explicabilité des décisions

---

### 48. IA explicable pour décisions d'investissement (XAI Finance)

**Description du problème et contexte**
Les modèles ML en finance (trading, gestion de portefeuille) sont souvent des boîtes noires incompatibles avec les exigences réglementaires de justification des décisions. L'IA explicable (XAI) combine l'argumentation computationnelle avec les techniques d'explicabilité pour fournir des justifications compréhensibles et auditables des recommandations d'investissement.

**Références multiples**
- **CFA Institute** : [Explainable AI in Finance: Addressing the Needs of Diverse Stakeholders](https://rpc.cfainstitute.org/research/reports/2025/explainable-ai-in-finance) - CFA Institute 2025
- **Revue systématique** : [A Systematic Review of Explainable AI in Finance](https://arxiv.org/pdf/2503.05966) - arXiv 2025
- **BIS** : [How regulators can address AI explainability](https://www.bis.org/fsi/fsipapers24.pdf) - Bank for International Settlements 2024
- **XAI Review** : [Explainable AI (XAI) in finance: a systematic literature review](https://link.springer.com/article/10.1007/s10462-024-10854-8) - AI Review 2024

**Approches suggérées**
- Implémenter un modèle de recommandation d'investissement (ML ou basé règles)
- Intégrer des techniques XAI (SHAP, LIME, counterfactual explanations)
- Développer un système d'argumentation pour structurer les justifications
- Créer une interface présentant les recommandations avec explications

**Technologies pertinentes**
- Python avec SHAP, LIME ou Captum pour l'explicabilité
- TweetyProject ou frameworks d'argumentation pour la structuration
- LLMs pour la génération d'explications en langage naturel
- Streamlit/Dash pour l'interface de visualisation

---

### 49. Planification d'investissement multi-périodes par programmation dynamique

**Description du problème et contexte**
La planification d'investissement sur plusieurs périodes avec objectifs de retraite, contraintes de liquidité et événements de vie (achat immobilier, études des enfants) se modélise comme un problème de programmation dynamique stochastique. Les approches CSP et métaheuristiques permettent d'intégrer des contraintes complexes non-linéaires.

**Références multiples**
- **Dynamic Programming** : [Optimal Rebalancing Strategy Using Dynamic Programming for Institutional Portfolios](https://www.researchgate.net/publication/228224355_Optimal_Rebalancing_Strategy_Using_Dynamic_Programming_for_Institutional_Portfolios) - ResearchGate
- **Asset Allocation** : [Principles of Asset Allocation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2025/principles-asset-allocation) - CFA Institute 2025
- **Vanguard** : [The rebalancing edge: Optimizing threshold-based strategies](https://corporate.vanguard.com/content/dam/corp/research/pdf/the_rebalancing_edge_optimizing_target_date_fund_rebalancing_through_threshold_based_strategies.pdf) - Vanguard Research 2024
- **Metaheuristics** : [Practical Portfolio Optimization with Metaheuristics](https://arxiv.org/pdf/2503.15965) - arXiv 2025

**Approches suggérées**
- Modéliser le problème comme MDP (Markov Decision Process) avec états financiers
- Implémenter l'équation de Bellman avec discrétisation de l'espace d'états
- Intégrer les contraintes de liquidité et objectifs de vie
- Comparer programmation dynamique, métaheuristiques et reinforcement learning

**Technologies pertinentes**
- Python avec NumPy/SciPy pour la programmation dynamique
- OR-Tools ou Gurobi pour les contraintes
- Gymnasium (OpenAI Gym) pour la formulation RL
- Monte Carlo simulation pour les scénarios stochastiques

---

### 50. Optimisation d'exécution d'ordres par contraintes (TWAP/VWAP)

**Description du problème et contexte**
L'exécution optimale de gros ordres boursiers nécessite de découper les transactions en sous-ordres pour minimiser l'impact sur le marché. Les stratégies TWAP (Time-Weighted Average Price) et VWAP (Volume-Weighted Average Price) se formulent comme des problèmes d'optimisation sous contraintes (volume, timing, coût d'impact) où la programmation par contraintes permet d'intégrer des règles de trading complexes.

**Références multiples**
- **Stanford** : [Volume Weighted Average Price Optimal Execution](https://web.stanford.edu/~boyd/papers/pdf/vwap_opt_exec.pdf) - Boyd et al.
- **IJCAI** : [An End-to-End Optimal Trade Execution Framework](https://www.ijcai.org/Proceedings/2020/0627.pdf) - IJCAI 2020
- **Columbia** : [An Optimal Control Strategy for Execution of Large Stock Orders Using LSTMs](https://cfe.columbia.edu/sites/cfe.columbia.edu/files/content/LSTM_PRICE_IMPACT_Bloomberg.pdf) - Columbia Finance
- **Safe Execution** : [Safe and Compliant Cross-Market Trade Execution via Constrained RL](https://arxiv.org/pdf/2510.04952) - arXiv 2025

**Approches suggérées**
- Modéliser le problème d'exécution comme CSP avec contraintes de volume et timing
- Implémenter les stratégies TWAP et VWAP comme baselines
- Développer une optimisation sous contraintes d'impact de marché
- Comparer avec des approches de contrôle optimal et reinforcement learning

**Technologies pertinentes**
- Python avec OR-Tools ou cvxpy pour l'optimisation sous contraintes
- Backtrader ou vectorbt pour le backtesting
- Données tick-by-tick (Binance, Alpaca) pour validation
- Pandas pour l'analyse de séries temporelles financières

---

### 51. Market Making optimal avec contraintes d'inventaire

**Description du problème et contexte**
Le market making consiste à fournir de la liquidité en proposant continuellement des prix bid et ask. Le market maker fait face à un problème d'optimisation complexe : maximiser le profit du spread tout en gérant le risque d'inventaire. Ce problème de contrôle stochastique avec contraintes se formule comme un problème HJB (Hamilton-Jacobi-Bellman) discrétisé ou comme CSP dynamique.

**Références multiples**
- **Référence fondatrice** : [Dealing with the Inventory Risk: A solution to the market making problem](https://arxiv.org/abs/1105.3115) - Guéant et al.
- **Stochastic Control** : [Stochastic Control for Optimal Market-Making](https://web.stanford.edu/~ashlearn/RLForFinanceBook/MarketMaking.pdf) - Stanford
- **QuantPedia** : [Optimal Market Making Models with Stochastic Volatility](https://quantpedia.com/optimal-market-making-models-with-stochastic-volatility/) - QuantPedia
- **SIAM** : [Adaptive Optimal Market Making Strategies with Inventory Liquidation](https://epubs.siam.org/doi/10.1137/23M1571058) - SIAM 2024

**Approches suggérées**
- Modéliser les contraintes d'inventaire (position max, VaR limite)
- Implémenter l'équation HJB discrétisée pour le spread optimal
- Développer une version CSP avec contraintes de risque explicites
- Tester sur données de carnet d'ordres simulées ou réelles

**Technologies pertinentes**
- Python avec NumPy/SciPy pour la résolution numérique
- OR-Tools pour la formulation CSP des contraintes
- Gymnasium pour la formulation RL alternative
- Données LOB (Limit Order Book) de LOBSTER ou Binance

---

### 52. Optimisation de portefeuille multi-objectifs Pareto (NSGA-II/III)

**Description du problème et contexte**
L'optimisation de portefeuille moderne dépasse le cadre bi-objectif rendement/risque pour intégrer des objectifs multiples : critères ESG, liquidité, drawdown maximum, turnover. Les algorithmes évolutionnaires multi-objectifs (NSGA-II, NSGA-III) permettent d'explorer la frontière de Pareto et de proposer un ensemble de solutions non-dominées au décideur.

**Références multiples**
- **NSGA-III Portfolio** : [Multi-Objective Portfolio Optimization: Application of NSGA-III](https://www.mdpi.com/2227-7072/13/1/15) - MDPI Finance 2025
- **EvoFolio** : [EvoFolio: portfolio optimization based on multi-objective evolutionary algorithms](https://link.springer.com/article/10.1007/s00521-024-09456-w) - Neural Computing 2024
- **Learning-guided** : [A learning-guided multi-objective evolutionary algorithm for constrained portfolio optimization](https://www.sciencedirect.com/science/article/pii/S1568494614003913) - Applied Soft Computing
- **Pareto Chain** : [Pareto evolutionary algorithm based on Markov chain for portfolio optimization](https://www.sciencedirect.com/science/article/pii/S0957417425027617) - Expert Systems 2025

**Approches suggérées**
- Définir 3-5 objectifs financiers (rendement, volatilité, ESG, max drawdown, liquidité)
- Implémenter NSGA-II et NSGA-III pour générer la frontière de Pareto
- Visualiser la frontière de Pareto en 2D/3D avec sélection interactive
- Comparer avec les méthodes de scalarisation classiques (weighted sum)

**Technologies pertinentes**
- Python avec pymoo ou DEAP pour les algorithmes multi-objectifs
- Pandas et yfinance pour les données financières
- Plotly pour la visualisation 3D interactive de la frontière Pareto
- scikit-learn pour les métriques ESG et de risque

---

### 53. Optimisation de la gestion de trésorerie (Cash Flow Scheduling)

**Description du problème et contexte**
La gestion optimale de la trésorerie d'entreprise implique de planifier les paiements et encaissements pour maximiser le rendement des liquidités tout en respectant les contraintes de solvabilité. Ce problème de scheduling financier avec fenêtres temporelles se modélise naturellement en programmation par contraintes avec des variables de timing et des contraintes de flux cumulatif.

**Références multiples**
- **Bi-objective** : [A New Bi-Objective Model for Resource-Constrained Project Scheduling with Cash Flow](https://arxiv.org/abs/2509.00002) - arXiv 2025
- **Risk-averse** : [A risk-averse distributionally robust model for cash flow management](https://www.sciencedirect.com/science/article/pii/S0377221724003965) - European Journal of OR 2024
- **Simulated Annealing** : [Simulated annealing for centralised multiproject scheduling with cash flow](https://link.springer.com/article/10.1007/s10479-023-05580-3) - Annals of OR 2023
- **Vendor Payments** : [A Cash Flow Optimization Model for Aligning Vendor Payments](https://www.irejournals.com/paper-details/1709383) - IRE Journals

**Approches suggérées**
- Modéliser les paiements comme tâches avec fenêtres temporelles et contraintes de précédence
- Implémenter des contraintes de solde minimum et limites de crédit
- Développer une fonction objectif multi-critères (coût de financement, retards, escomptes)
- Utiliser métaheuristiques (recuit simulé) ou CSP pour la résolution

**Technologies pertinentes**
- Python avec OR-Tools CP-SAT pour le scheduling sous contraintes
- Pandas pour la gestion des flux de trésorerie
- PuLP ou Gurobi pour la formulation MILP alternative
- Streamlit pour un tableau de bord de gestion de trésorerie

---

### 54. IA Neuro-Symbolique pour la Finance (Hybrid AI)

**Description du problème et contexte**
L'IA neuro-symbolique combine les forces des réseaux de neurones (apprentissage à partir de données) et de l'IA symbolique (raisonnement explicable, intégration de connaissances expertes). En finance, cette approche hybride permet de créer des systèmes qui apprennent des patterns complexes tout en respectant des règles métier explicites et en fournissant des explications auditables.

**Références multiples**
- **SmythOS** : [Symbolic AI in Finance: Transforming Risk Management](https://smythos.com/managers/finance/symbolic-ai-in-finance/) - SmythOS
- **Neuro-Symbolic AML** : [Neuro-Symbolic AI: Finance's Edge in Fraud and Compliance](https://www.linkedin.com/pulse/neuro-symbolic-ai-finances-edge-fraud-compliance-leo-akin-odutola-tf7pc) - LinkedIn 2025
- **Capgemini** : [Teaming up AI capabilities for fraud prevention](https://www.capgemini.com/insights/expert-perspectives/mulder-and-scully-for-fraud-prevention-teaming-up-ai-capabilities/) - Capgemini
- **Hybrid Detection** : [Detecting Financial Fraud with Hybrid Deep Learning](https://arxiv.org/pdf/2504.03750) - arXiv 2025

**Approches suggérées**
- Combiner un réseau de neurones pour la détection de patterns avec un système de règles métier
- Implémenter une architecture où les règles symboliques contraignent ou guident l'apprentissage
- Développer un système d'explication qui traduit les prédictions en raisonnements logiques
- Appliquer à un cas d'usage financier (fraude, crédit, trading)

**Technologies pertinentes**
- Python avec PyTorch pour les réseaux de neurones
- PyKE ou CLIPS pour le moteur de règles symboliques
- NeuralLP ou Neural Theorem Provers pour l'intégration
- SHAP/LIME pour connecter les explications aux règles

