"""Plots the minimax survivor distribution for a given generation.

Note: currently only supports runs with exactly two tasks.

Example:
    python3 scripts/visualization/visualize_ga.py results/ga/n2d2_hw_01.pkl
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer

def plot_survivors(visualizer: GAVisualizer, gen_idx: int = -1):
    if gen_idx == -1:
        gen_idx = len(visualizer.manager.history) - 1

    # Compute fitness and select survivors
    survivor_factor = visualizer.manager.n_survivors / visualizer.manager.population_size
    _, _, all_accs, all_energy = visualizer.manager._collect_individuals(range(gen_idx + 1))
    fitness = visualizer.manager._compute_fitness(all_accs, all_energy, method="worst_rank")
    survivor_indices, non_survivor_indices = visualizer.select_survivors(fitness, survivor_factor, survivor_factor)

    # Plot accuracy of survivors and non-survivors
    task_names = [visualizer.prettify_task_name(task) for task in visualizer.tasks]
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 8))
    plt.scatter(all_accs[0][survivor_indices], all_accs[1][survivor_indices],
                    alpha=0.3, color='steelblue', label='Survivors', edgecolor="none")
    plt.scatter(all_accs[0][non_survivor_indices], all_accs[1][non_survivor_indices],
                alpha=0.3, color='rosybrown', label='Non-Survivors', edgecolor="none")
    plt.title(f"Minimax Survivor Distribution: {task_names[0]} vs {task_names[1]}")
    plt.xlabel(f"Eval Results {task_names[0]}")
    plt.ylabel(f"Eval Results {task_names[1]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(visualizer.output_path) / f"minimax_survivors_{task_names[0]}_vs_{task_names[1]}.jpg")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the ga pickle file')
    parser.add_argument('gen_idx', type=int, default=-1, help='Generation to plot (default: latest)')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_survivors(visualizer, args.gen_idx) 