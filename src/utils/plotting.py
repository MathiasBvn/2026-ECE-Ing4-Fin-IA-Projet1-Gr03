import matplotlib
matplotlib.use('Agg') # Backend non-interactif
import matplotlib.pyplot as plt
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

def plot_wealth_convergence(df: pd.DataFrame, title: str, save_path: str = None):
    """Affiche la convergence de la richesse avec intervalle de confiance."""
    setup_style()
    plt.figure(figsize=(12, 7))
    
    # Calcul des stats
    stats = df.groupby('time')['wealth'].agg(['mean', 'std', lambda x: np.percentile(x, 5), lambda x: np.percentile(x, 95)])
    stats.columns = ['mean', 'std', 'p5', 'p95']
    
    # Zone d'ombre (écart-type)
    plt.fill_between(stats.index, stats['mean'] - stats['std'], stats['mean'] + stats['std'], 
                     color='blue', alpha=0.1, label='Écart-type')
    
    # Zone d'ombre (Percentiles 5-95)
    plt.fill_between(stats.index, stats['p5'], stats['p95'], 
                     color='blue', alpha=0.05, linestyle='--', label='Intervalle 5%-95%')
    
    # Ligne moyenne
    plt.plot(stats.index, stats['mean'], color='#1f77b4', linewidth=3, label='Richesse Moyenne')
    
    plt.title(title, pad=20)
    plt.xlabel("Temps (Périodes)")
    plt.ylabel("Richesse")
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_allocation_stacked(df: pd.DataFrame, title: str, save_path: str = None):
    """Affiche l'évolution de l'allocation sous forme de Stacked Area Chart."""
    setup_style()
    
    # Trouver les colonnes d'allocation
    alloc_cols = [c for c in df.columns if c.startswith('alloc_')]
    avg_alloc = df.groupby('time')[alloc_cols].mean()
    
    # Garantir des valeurs positives pour le stacked area chart
    avg_alloc = avg_alloc.clip(lower=0)
    # Re-normaliser pour que la somme soit 1 après le clip
    avg_alloc = avg_alloc.div(avg_alloc.sum(axis=1), axis=0)
    
    # Mapper les noms de colonnes aux noms d'actifs pour les couleurs
    asset_names = [c.replace('alloc_', '').capitalize() for c in alloc_cols]
    colors = [ASSET_COLORS.get(name, "#000000") for name in asset_names]
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    avg_alloc.plot(kind='area', stacked=True, ax=ax, alpha=0.85, color=colors)
    
    plt.title(title, pad=20)
    plt.xlabel("Temps (Périodes)")
    plt.ylabel("Allocation (%)")
    plt.ylim(0, 1)
    
    # Légende à l'extérieur
    plt.legend(asset_names, loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_wealth_distribution(all_results_df: pd.DataFrame, horizon: int, save_path: str = None):
    """Compare la distribution de la richesse finale entre les solveurs (Box/Violin Plot)."""
    setup_style()
    plt.figure(figsize=(12, 7))
    
    # Filtrer pour la richesse finale
    final_wealth = all_results_df[all_results_df['time'] == horizon]
    
    # Violin Plot + Box Plot
    sns.violinplot(data=final_wealth, x='solver', y='wealth', inner="quartile", palette="muted")
    sns.swarmplot(data=final_wealth, x='solver', y='wealth', color="white", alpha=0.4, size=3)
    
    plt.title("Distribution de la Richesse Finale par Solveur", pad=20)
    plt.xlabel("Solveur")
    plt.ylabel("Richesse Finale")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def plot_results_professional(all_results_df: pd.DataFrame, horizon: int, output_dir: str = "output"):
    """Génère l'ensemble des graphiques professionnels pour le rapport."""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 1. Graphiques d'allocation par solveur
    for solver_name in all_results_df['solver'].unique():
        solver_df = all_results_df[all_results_df['solver'] == solver_name]
        plot_allocation_stacked(solver_df, f"Stratégie d'Allocation : {solver_name}", 
                                f"{output_dir}/{solver_name.lower()}_alloc_prof.png")
        plot_wealth_convergence(solver_df, f"Convergence de la Richesse : {solver_name}", 
                                f"{output_dir}/{solver_name.lower()}_wealth_prof.png")
        
    # 2. Comparaison des distributions
    plot_wealth_distribution(all_results_df, horizon, f"{output_dir}/comparison_distribution.png")
