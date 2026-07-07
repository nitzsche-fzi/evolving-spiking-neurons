 #!/usr/bin/env python3
"""
Visualizer script for GA run results that plots survivor distributions for different fitness methods.

This script takes a pkl file from a GA run, selects a given generation, and plots the survivor 
distribution for every fitness method in manager.py (worst_rank, sum_rank, and sum_fitness).
Hardware fitness integration is optional.

Usage:
    python3 plot_fitness_methods.py results/ga/n2d2_hw_01.pkl --generation 0 --include-hw-fitness
    python3 plot_fitness_methods.py results/ga/n2d2_hw_01.pkl --generation -1 --n-survivors 0.1
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer

def plot_fitness_methods(visualizer: GAVisualizer, gen_idx: int = -1, include_hw_fitness: bool = True, n_survivors: float = 0.1):
    if gen_idx == -1:
        gen_idx = len(visualizer.manager.history) - 1

    # Collect data
    _, _, accs, energy = visualizer.manager._collect_individuals(range(gen_idx + 1))
    if not include_hw_fitness:
        energy = None
    
    # Compute fitness and select survivors for each method
    fitness_methods = ["sum_fitness", "sum_rank", "worst_rank"]
    survivor_data = {}
    for method in fitness_methods:
        fitness = visualizer.manager._compute_fitness(accs, energy, method, verbose=include_hw_fitness)
        survivor_indices, non_survivor_indices = visualizer.select_survivors(fitness, n_survivors)
        survivor_data[method] = (survivor_indices, non_survivor_indices)

    # Create subplots for each fitness method
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
    fig, axes = plt.subplots(1, len(fitness_methods), figsize=(5*len(fitness_methods), 5))
    if len(fitness_methods) == 1:
        axes = [axes]
    
    for i, method in enumerate(fitness_methods):
        ax = axes[i]
        survivor_indices, non_survivor_indices = survivor_data[method]
        
        # Plot survivors and non-survivors
        ax.scatter(accs[0][survivor_indices], accs[1][survivor_indices],
                  alpha=0.3, color='steelblue', label='Survivors', edgecolor="none", s=20)
        ax.scatter(accs[0][non_survivor_indices], accs[1][non_survivor_indices],
                  alpha=0.3, color='rosybrown', label='Non-Survivors', edgecolor="none", s=20)

        ax.set_title(f"{method.replace('_', ' ').title()}")
        # Only show x-axis label on the middle plot (sum_rank)
        if method == "sum_rank":
            ax.set_xlabel(f"Accuracy in {visualizer.prettify_task_name(visualizer.tasks[0])}")
        # Only show y-axis label on the left plot (sum_fitness)
        if method == "sum_fitness":
            ax.set_ylabel(f"Accuracy in {visualizer.prettify_task_name(visualizer.tasks[1])}")
        # Only show legend on the middle plot (sum_rank)
        if method == "sum_rank":
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    hw_suffix = "_hw" if include_hw_fitness else ""
    filename = f"fitness_methods_gen{gen_idx}{hw_suffix}.jpg"
    plt.savefig(Path(visualizer.output_path) / filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize survivor distributions for different fitness methods in GA runs"
    )
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    parser.add_argument('--generation', type=int, default=-1, 
                       help='Generation to analyze (default: -1 for latest)')
    parser.add_argument('--include-hw-fitness', action='store_true',
                       help='Include hardware fitness in calculations')
    parser.add_argument('--n-survivors', type=float, default=0.1,
                       help='Factor for number of survivors (default: 0.1, i.e., 10%%)')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_fitness_methods(visualizer, args.generation, args.include_hw_fitness, args.n_survivors)


