"""Plots GA accuracy progress over generations.

Shows the best neuron's per-task accuracy, the best neuron's mean accuracy, and
the population mean accuracy.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer


def plot_ga_progress(visualizer: GAVisualizer):
    """Plot accuracies for best neuron (based on fitness, accuracy and energy) and mean population over generations."""

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
    _plot_ga_progress_accuracy(visualizer, consider_past_gens=True)

    #_plot_ga_progress_accuracy(visualizer, consider_past_gens=False) # This is probably not useful as it provides misleading information

def _plot_ga_progress_accuracy(visualizer: GAVisualizer, consider_past_gens: bool = False):
    """Plot accuracies for best neuron (based on fitness, accuracy and energy) and mean population over generations."""

    n_gens = visualizer.n_generations
    x = list(range(n_gens))

    top_individuals = visualizer._find_top_individuals(consider_past_gens=consider_past_gens) # shape (n_gens)
    population_acc = visualizer.pop_task_accuracy
    for metric in visualizer.relevant_metrics:
        plt.figure(figsize=(12, 8))

        ## Best neuron
        # Plot per-task best neuron accuracies
        best_accuracies = np.array([
            [top_individuals[gen_idx][metric]["accuracies"][task] for gen_idx in range(n_gens)] for task in visualizer.tasks
        ]) # shape (n_tasks, n_gens)
        for task_idx, task in enumerate(visualizer.tasks):
            task_name = visualizer.prettify_task_name(task)
            sns.lineplot(x=x, y=best_accuracies[task_idx], **visualizer.styles["accuracy"][task_name], label=f"Best {task_name}")

        # Plot mean accuracy of best neuron across tasks
        sns.lineplot(x=x, y=best_accuracies.mean(axis=0), **visualizer.styles["accuracy"]["mean"], label="Best All Tasks")

        ## Population
        # Plot mean population accuracy across all tasks
        sns.lineplot(
            x=x, y=population_acc.mean(axis=0), **visualizer.styles["accuracy_pop"]["mean"], label="Population All Tasks"
        )
        # Plot mean population accuracy per task
        for task_idx, task in enumerate(visualizer.tasks):
            task_name = visualizer.prettify_task_name(task)
            sns.lineplot(
                x=x, y=population_acc[task_idx], **visualizer.styles["accuracy_pop"][task_name], label=f"Population {task_name}"
            )


        plt.title(f"Best {metric.capitalize()} Neuron")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.xticks(range(n_gens))  # Force integer x-axis ticks
        plt.ylim(0, 1.05)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.grid(True, which='major', axis='both')
        plt.legend(fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        suffix = "" if consider_past_gens else "_single_gen"
        plt.savefig(Path(visualizer.output_path) / f"ga_progress_best_{metric}{suffix}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_ga_progress(visualizer)