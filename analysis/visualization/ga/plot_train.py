"""Plots accuracy, max gradient and spike rate over training for the latest generation."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer


def plot_train(visualizer: GAVisualizer, smoothing: int = 0):
    sns.set_theme(style="whitegrid", context="talk")

    top_individuals = visualizer._find_top_individuals(consider_past_gens=False) # Make sure to only consider the latest generation
    best_individual = top_individuals[-1]["fitness"]

    for task in visualizer.tasks:
        best_task_individual = top_individuals[-1][f"fitness_{task}"]

        task_results = visualizer.manager.history[-1]["results"][task]
        accuracies   = np.array(task_results["accuracies"])
        spike_rates  = np.array(task_results["spike_rates"])
        max_grad     = np.array(task_results["max_grad"])

        accuracies  = visualizer.smooth_training_data(accuracies,  smoothing)
        spike_rates = visualizer.smooth_training_data(spike_rates, smoothing)

        task_name = visualizer.prettify_task_name(task)

        # Plot accuracy
        _plot_quantiles_shaded(
            visualizer=visualizer,
            data=accuracies,
            best_data=accuracies[best_individual["local_index"]],
            best_task_data=accuracies[best_task_individual["local_index"]],
            title=f"Accuracy over Training for {task_name}, Smoothed over {smoothing} Training Steps",
            ylabel="Accuracy",
            filename=f"train_{task_name}_accuracy.jpg"
        )

        # Plot spike rate
        _plot_quantiles_shaded(
            visualizer=visualizer,
            data=spike_rates,
            best_data=spike_rates[best_individual["local_index"]],
            best_task_data=spike_rates[best_task_individual["local_index"]],
            title=f"Spike Rate over Training for {task_name}, Smoothed over {smoothing} Training Steps",
            ylabel="Spike Rate",
            filename=f"train_{task_name}_spike_rate.jpg"
        )

        _plot_max_grad(visualizer, max_grad, task_name, smoothing)

def _plot_quantiles_shaded(visualizer: GAVisualizer, data: np.ndarray, best_data: np.ndarray, best_task_data: np.ndarray, title: str, ylabel: str, filename: str):
    plt.figure(figsize=(14, 6))
    steps = np.arange(data.shape[1])
    quantile_bands = [
        (0.0, 1.0, 0.1),   # outermost band, most transparent
        (0.1, 0.9, 0.2),
        (0.25, 0.75, 0.3),
    ]

    # Plot quantile bands
    for q_low, q_high, alpha in quantile_bands:
        low = np.quantile(data, q_low, axis=0)
        high = np.quantile(data, q_high, axis=0)
        plt.fill_between(
            steps,
            low,
            high,
            alpha=alpha,
            color="steelblue",
            label=f"{int(q_low*100)}–{int(q_high*100)}%",
            edgecolor="none"
        )

    # Plot the best individual's curve
    plt.plot(steps, best_task_data, color="black", lw=1, label="Best Task Individual")
    plt.plot(steps, best_data,      color="coral", lw=1, label="Best Individual")

    plt.title(title)
    plt.xlabel("Training Step")
    plt.ylabel(ylabel)
    
    # Automatically adjust y-axis scaling based on data range
    ax = plt.gca()
    ax.set_ylim(bottom=0)
    if ylabel == "Accuracy":
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 100:.0f}%'))
    else:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=False))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x * 100:.1f}%'))

    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(visualizer.output_path) / filename)
    plt.close()

def _plot_max_grad(visualizer: GAVisualizer, max_grad: np.ndarray, task_name: str, smoothing: int):
    plt.figure(figsize=(14, 6))
    # max grad shape: [n_steps,]
    smoothed_max_grad = visualizer.smooth_training_data(max_grad.reshape(1, -1), smoothing)[0]  # Get the first (and only) row after smoothing
    steps = np.arange(max_grad.shape[0])
    x_values = np.arange(len(max_grad))
    plt.scatter(x_values, max_grad, color="steelblue", s=10, label="Max Gradient", alpha=0.5)
    plt.plot(steps, smoothed_max_grad, color="black", lw=2, label="Smoothed Max Gradient")
    plt.title(f"Max Gradient over Training for {task_name}, Smoothed over {smoothing} Training Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Max Gradient")
    plt.yscale('log')  # Use logarithmic scale for better visibility
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(visualizer.output_path) / f"train_{task_name}_max_grad.jpg")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    parser.add_argument('--smoothing', type=int, nargs='?', default=0, help='Smoothing factor for the data')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_train(visualizer, args.smoothing)
