"""Plots spike rates against each other and against test accuracy.

Uses the latest generation of a GA run.

Note: this currently uses training spike rates rather than evaluation spike
rates.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer


def plot_spike_rates(visualizer: GAVisualizer, smoothing: int = 20):
    """Draws the spike-rate scatter/comparison plots for the latest generation."""
    sns.set_theme(style="whitegrid", context="talk")

    latest_generation = visualizer.manager.history[-1]

    _, _, all_accs, all_energy = visualizer.manager._collect_individuals([-1])
    fitness = visualizer.manager._compute_fitness(all_accs, all_energy, method="worst_rank")

    for task in visualizer.tasks:
        task_name = visualizer.prettify_task_name(task)
        task_results = latest_generation["results"][task]

        spike_rates_raw = np.array(task_results["spike_rates"])
        spike_rates = visualizer.smooth_training_data(spike_rates_raw, smoothing)

        final_accuracy = np.array(task_results["eval_results"]) # shape (pop_size,)

        final_spike_rate = spike_rates[:, -1]
        initial_spike_rate = spike_rates[:, 0]
        model_spike_rate = np.array([p.spike_rate for p in latest_generation["params"]])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        def plot(
            ax: plt.Axes,
            x,
            y,
            rank,
            xlabel: str,
            ylabel: str,
            title: str,
            x_log: bool = False,
            y_log: bool = False,
        ):
            def _axis_limit(values: np.ndarray) -> float:
                max_val = np.nanmax(values)
                if not np.isfinite(max_val) or max_val <= 0:
                    return 1.0
                return max_val

            sc = ax.scatter(x, y, c=rank, cmap=cm.viridis_r, alpha=0.4, edgecolors='none')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if x_log:
                ax.set_xscale("log")
            else:
                ax.set_xlim(0, _axis_limit(x) * 1.05)
            if y_log:
                ax.set_yscale("log")
            else:
                ax.set_ylim(0, _axis_limit(y) * 1.05)
            ax.set_title(title)
            ax.grid(True)
            return sc

        plots = [
            (model_spike_rate,   final_accuracy,     "Model Spike Rate",   "Test Accuracy",      "Acc vs Model Spike Rate"),
            (initial_spike_rate, final_accuracy,     "Initial Spike Rate", "Test Accuracy",      "Acc vs Init SR"),
            (final_spike_rate,   final_accuracy,     "Final Spike Rate",   "Test Accuracy",      "Acc vs Final SR"),
            (model_spike_rate,   initial_spike_rate, "Model Spike Rate",   "Initial Spike Rate", "Init SR vs Model SR"),
            (model_spike_rate,   final_spike_rate,   "Model Spike Rate",   "Final Spike Rate",   "Final SR vs Model SR"),
            (initial_spike_rate, final_spike_rate,   "Initial Spike Rate", "Final Spike Rate",   "Final vs Init SR"),
        ]

        for ax, plot_args in zip(axes, plots):
            if len(plot_args) == 5:
                x, y, xlabel, ylabel, title = plot_args
                sc = plot(ax, x, y, fitness, xlabel, ylabel, title)
            else:
                x, y, xlabel, ylabel, title, x_log, y_log = plot_args
                sc = plot(ax, x, y, fitness, xlabel, ylabel, title, x_log=x_log, y_log=y_log)

        fig.subplots_adjust(right=0.88, top=0.92, wspace=0.3, hspace=0.4)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label("Fitness Rank (lower is better)")

        fig.suptitle(f"Scatterplot Overview for Task: {task_name}", fontsize=16)
        fig.savefig(Path(visualizer.output_path) / f"spike_rates_{task_name}_scatter_grid.jpg")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the ga pickle file')
    parser.add_argument('smoothing', type=int, nargs='?', default=20,)
    args = parser.parse_args()
    
    visualizer = GAVisualizer(args.ga_file)
    plot_spike_rates(visualizer, args.smoothing)
