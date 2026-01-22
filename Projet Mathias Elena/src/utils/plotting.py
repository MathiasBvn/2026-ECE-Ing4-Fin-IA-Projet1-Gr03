import matplotlib
matplotlib.use('Agg') # Backend non-interactif
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

# Palette de couleurs sémantique fixe
ASSET_COLORS = {
    "Actions": "#1f77b4",      # Bleu institutionnel
    "Obligations": "#2ca02c",  # Vert rassurant
    "Cash": "#7f7f7f",         # Gris neutre
    "Or": "#ffd700",           # Or/Jaune distinct
    "Crypto": "#9467bd",       # Violet Tech
    "SCPI": "#8c564b"          # Marron/Brique
}

def setup_style():
    """Configure le style global pour des graphiques professionnels."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

def plot_wealth_convergence(df: pd.DataFrame, title: str, save_path: str = None,
                            target_wealth: float = None, inflation_rate: float = 0.0,
                            life_events: Dict[int, float] = None,
                            event_names: Dict[int, str] = None):
    """Affiche la convergence de la richesse avec événements et liquidité."""
    setup_style()
    plt.figure(figsize=(14, 8))
    
    # Vérification de robustesse : s'assurer que les données de richesse sont correctement formatées
    # Si le DataFrame contient une seule trajectoire, s'assurer que les calculs de statistiques fonctionnent
    if 'wealth' not in df.columns:
        print(f"  Erreur : Colonne 'wealth' non trouvée dans le DataFrame")
        plt.close()
        return
    
    # Vérification de dimension pour gérer le cas DP avec une seule trajectoire
    # Si les données de richesse sont en 1D (cas DP unique), on les transforme en 2D
    wealth_data = df['wealth'].values
    if wealth_data.ndim == 1:
        # Cas d'une seule trajectoire : on crée un DataFrame avec une seule simulation
        # Les calculs de mean/std fonctionneront correctement
        pass  # Le groupby pandas gère déjà ce cas correctement
    
    # Vérification supplémentaire : s'assurer qu'il y a plusieurs trajectoires pour les quantiles
    # Si une seule trajectoire, les quantiles p5 et p95 seront égaux à la moyenne
    n_trajectories = df['trajectory'].nunique() if 'trajectory' in df.columns else 1
    if n_trajectories == 1:
        print(f"  Note : Une seule trajectoire détectée - les intervalles de confiance ne sont pas significatifs")
    
    # Calcul des statistiques
    grouped = df.groupby('time')['wealth']
    stats_nom = grouped.mean().to_frame(name='mean')
    stats_nom['p5'] = grouped.quantile(0.05)
    stats_nom['p95'] = grouped.quantile(0.95)
    
    # Stats pour la liquidité
    stats_liq = df.groupby('time')['liquidity'].mean()
    
    # Zone d'ombre (Percentiles 5-95) pour la richesse nominale
    plt.fill_between(stats_nom.index, stats_nom['p5'], stats_nom['p95'], 
                     color='#1f77b4', alpha=0.1, label='Intervalle 5%-95% (Nominal)')
    
    # Ligne moyenne Nominale
    plt.plot(stats_nom.index, stats_nom['mean'], color='#1f77b4', linewidth=3, label='Richesse Nominale Moyenne')
    
    # Courbe de Liquidité Disponible
    plt.plot(stats_liq.index, stats_liq, color='#ff7f0e', linewidth=2, linestyle='--', label='Liquidité (Cash + Oblig)')
    
    # Ligne d'objectif cible
    if target_wealth:
        plt.axhline(y=target_wealth, color='black', linestyle=':', alpha=0.6, label=f'Objectif ({target_wealth}k€)')
    
    # Ajout des événements de vie
    if life_events:
        # On récupère les limites actuelles pour placer le texte
        y_max = df['wealth'].max()
        for year, amount in life_events.items():
            if amount > 0:
                plt.axvline(x=year, color='red', linestyle='-', alpha=0.3)
                name = event_names.get(year, "Événement") if event_names else "Événement"
                plt.text(year, y_max * 0.9, f"{name}\n-{amount}k€", 
                         color='red', fontweight='bold', ha='center', fontsize=10)
    
    plt.title(title, pad=20)
    plt.xlabel("Temps (Années)")
    plt.ylabel("Valeur du Portefeuille (k€)")
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_allocation_stacked(df: pd.DataFrame, title: str, save_path: str = None):
    """Affiche l'évolution de l'allocation sous forme de Stacked Area Chart."""
    setup_style()
    
    alloc_cols = [c for c in df.columns if c.startswith('alloc_')]
    avg_alloc = df.groupby('time')[alloc_cols].mean()
    avg_alloc = avg_alloc.clip(lower=0)
    avg_alloc = avg_alloc.div(avg_alloc.sum(axis=1), axis=0)
    
    asset_names = [c.replace('alloc_', '').capitalize() for c in alloc_cols]
    colors = [ASSET_COLORS.get(name, "#000000") for name in asset_names]
    
    fig, ax = plt.subplots(figsize=(13, 7))
    avg_alloc.plot(kind='area', stacked=True, ax=ax, alpha=0.85, color=colors)
    
    plt.title(title, pad=20)
    plt.xlabel("Temps (Années)")
    plt.ylabel("Allocation (%)")
    plt.ylim(0, 1)
    plt.legend(asset_names, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_wealth_distribution(all_results_df: pd.DataFrame, horizon: int, save_path: str = None):
    setup_style()
    plt.figure(figsize=(12, 7))
    final_wealth = all_results_df[all_results_df['time'] == horizon]
    sns.violinplot(data=final_wealth, x='solver', y='wealth', inner="quartile", palette="muted")
    plt.title("Distribution de la Richesse Finale par Solveur", pad=20)
    plt.xlabel("Solveur")
    plt.ylabel("Richesse Finale (k€)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_allocation_professional(allocation_df: pd.DataFrame, output_dir: str,
                                  filename: str = "alloc_prof.png"):
    """
    Génère un graphique professionnel de l'allocation d'actifs.
    
    Args:
        allocation_df: DataFrame contenant les colonnes d'allocation (alloc_*)
        output_dir: Répertoire de sortie
        filename: Nom du fichier de sortie
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    setup_style()
    
    # Liste fixe d'actifs dans l'ordre défini dans config.py
    labels = ['Actions', 'Obligations', 'Cash', 'Or', 'Crypto', 'SCPI']
    
    # Colonnes d'allocation dans le bon ordre
    alloc_cols = [f'alloc_{label.lower()}' for label in labels]
    
    # Vérifier que toutes les colonnes existent
    available_cols = [c for c in alloc_cols if c in allocation_df.columns]
    
    if not available_cols:
        print(f"  Avertissement : Aucune colonne d'allocation trouvée dans le DataFrame")
        return
    
    # Vérification de sécurité : tronquer labels et colors si moins de colonnes disponibles
    num_assets = len(available_cols)
    if num_assets < len(labels):
        labels = labels[:num_assets]
    
    # Calculer l'allocation moyenne par temps
    avg_alloc = allocation_df.groupby('time')[available_cols].mean()
    avg_alloc = avg_alloc.clip(lower=0)
    avg_alloc = avg_alloc.div(avg_alloc.sum(axis=1), axis=0)
    
    # Couleurs correspondantes aux actifs (6 couleurs distinctes)
    colors = [ASSET_COLORS.get(label, "#000000") for label in labels]
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(13, 7))
    avg_alloc.plot(kind='area', stacked=True, ax=ax, alpha=0.85, color=colors)
    
    plt.title("Évolution de l'Allocation d'Actifs", pad=20)
    plt.xlabel("Temps (Années)")
    plt.ylabel("Allocation (%)")
    plt.ylim(0, 1)
    
    # Création manuelle des rectangles pour la légende pour être sûr des couleurs
    legend_handles = []
    for label, color in zip(labels, colors):
        legend_handles.append(mpatches.Patch(color=color, label=label))
    plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), title="Actifs")
    
    plt.tight_layout()
    
    # Sauvegarde
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Graphique d'allocation sauvegardé : {filename}")


def plot_results_professional(all_results_df: pd.DataFrame, horizon: int,
                                output_dir: str = "output", target_wealth: float = None,
                                inflation_rate: float = 0.0, life_events: Dict[int, float] = None,
                                event_names: Dict[int, str] = None):
    """Génère l'ensemble des graphiques professionnels pour le rapport."""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dictionnaire pour stocker les résultats par solveur
    results = {}
    
    for solver_name in all_results_df['solver'].unique():
        solver_df = all_results_df[all_results_df['solver'] == solver_name]
        results[solver_name] = solver_df
        
        # --- BLOC DE FORÇAGE POUR L'ALLOCATION ---
        print(f"\nGénération du graphique d'allocation pour {solver_name}...")
        
        # Vérifier si les colonnes d'allocation existent
        alloc_cols = [c for c in solver_df.columns if c.startswith('alloc_')]
        
        if alloc_cols:
            # On construit le nom de fichier dynamiquement
            filename = f"{solver_name.lower().replace(' ', '_')}_alloc_prof.png"
            
            try:
                plot_allocation_professional(
                    solver_df,
                    output_dir,
                    filename=filename
                )
                print(f"  -> Succès : {filename}")
            except Exception as e:
                print(f"  -> Erreur lors du plot allocation {solver_name}: {e}")
        else:
            print(f"  Attention : Pas d'historique d'allocation trouvé pour {solver_name}")
        # --- FIN DU BLOC DE FORÇAGE ---
        
        # Graphique de richesse
        plot_wealth_convergence(solver_df, f"Convergence de la Richesse : {solver_name}",
                                f"{output_dir}/{solver_name.lower()}_wealth_prof.png",
                                target_wealth=target_wealth, inflation_rate=inflation_rate,
                                life_events=life_events, event_names=event_names)
        
    plot_wealth_distribution(all_results_df, horizon, f"{output_dir}/comparison_distribution.png")

def plot_risk_reward_comparison(all_results_df: pd.DataFrame, horizon: int,
                                  save_path: str = None, title: str = "Frontière Efficiente : Risque vs Rendement"):
    """
    Affiche un nuage de points Risque vs Rendement pour comparer les solveurs.
    
    Axe X : Écart-type de la richesse finale (Risque)
    Axe Y : Richesse finale moyenne (Rendement)
    Chaque point représente un solveur.
    """
    setup_style()
    plt.figure(figsize=(12, 8))
    
    # Calcul des statistiques par solveur
    final_wealth = all_results_df[all_results_df['time'] == horizon]
    
    stats = []
    for solver_name in final_wealth['solver'].unique():
        solver_data = final_wealth[final_wealth['solver'] == solver_name]['wealth']
        wealth = np.array(solver_data)
        # Gestion robuste des dimensions (Cas DP vs Monte Carlo)
        if wealth.ndim == 1:
            # Si 1D (ex: DP), on transforme en (1, T) pour simuler 1 scénario
            wealth = wealth.reshape(1, -1)
        mean_wealth = wealth.mean(axis=0).mean()
        std_wealth = wealth.std(axis=0).mean()
        stats.append({
            'solver': solver_name,
            'mean': mean_wealth,
            'std': std_wealth
        })
    
    stats_df = pd.DataFrame(stats)
    
    # Couleurs pour chaque solveur
    solver_colors = {
        'DP': '#1f77b4',      # Bleu
        'OR-Tools': '#2ca02c', # Vert
        'RL': '#ff7f0e'       # Orange
    }
    
    # Tracer les points
    for _, row in stats_df.iterrows():
        color = solver_colors.get(row['solver'], '#333333')
        plt.scatter(row['std'], row['mean'],
                   s=300, c=color, alpha=0.7, edgecolors='black', linewidth=2,
                   label=row['solver'])
        
        # Ajouter le nom du solveur au-dessus du point
        plt.annotate(row['solver'],
                    (row['std'], row['mean']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Ajouter une ligne de référence (frontière efficiente théorique)
    # On trace une courbe de tendance pour visualiser la relation risque-rendement
    if len(stats_df) > 1:
        sorted_df = stats_df.sort_values('std')
        plt.plot(sorted_df['std'], sorted_df['mean'],
                'k--', alpha=0.3, linewidth=1, label='Tendance')
    
    plt.title(title, pad=20)
    plt.xlabel("Écart-type de la Richesse Finale (k€) - Risque", fontsize=14)
    plt.ylabel("Richesse Finale Moyenne (k€) - Rendement", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', frameon=True, fontsize=12)
    
    # Ajuster les limites pour mieux visualiser
    x_min = stats_df['std'].min() * 0.9
    x_max = stats_df['std'].max() * 1.1
    y_min = stats_df['mean'].min() * 0.9
    y_max = stats_df['mean'].max() * 1.1
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graphique sauvegardé dans : {save_path}")
    plt.close()
