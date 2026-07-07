"""Plots population accuracy and energy across all generations of a GA run."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer

def plot_ga_population(visualizer: GAVisualizer, normalize_energy: bool = False):
    """Plot population accuracies and energy over generations."""
    normalize_energy = normalize_energy and visualizer.has_hardware
    
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

    # Plotting
    plt.figure(figsize=(12, 8))
    x = list(range(visualizer.n_generations))
    ax1 = plt.gca()
    if visualizer.has_hardware and not normalize_energy:
        ax2 = ax1.twinx()
        axes = (ax1, ax2)
    else:
        axes = (ax1, )

    ###### ACCURACY #######
    # Population accuracy per task
    for task_idx, task in enumerate(visualizer.tasks):
        task_name = visualizer.prettify_task_name(task)
        sns.lineplot(x=x, y=visualizer.pop_task_accuracy[task_idx], **visualizer.styles["accuracy"][task_name],
            label=f"Accuracy on {task_name}", ax=ax1,
        )
    # Population accuracy across all tasks
    sns.lineplot(x=x, y=visualizer.pop_task_accuracy.mean(axis=0), **visualizer.styles["accuracy"]["mean"],
        label="Accuracy Across All Tasks", ax=ax1,
    )

    title = f"Population Accuracy{f' and Energy' if visualizer.has_hardware else ''}"
    ax1.set_title(title)
    ax1.set_xlabel("Generation")
    ax1_ylabel = f"Accuracy{' / Normalized Energy' if normalize_energy else ''}"
    ax1.set_ylabel(ax1_ylabel)
    ax1.set_xticks(range(visualizer.n_generations))
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.grid(True, which='major', axis='both')

    ###### ENERGY #######
    if visualizer.has_hardware:
        energy = visualizer.get_pop_energy_all_gens(normalize_energy)
        ax_energy = ax1 if normalize_energy else ax2 
        prefix = "Normalized " if normalize_energy else ""
        # Population energy across all tasks
        sns.lineplot(x=x, y=energy.mean(axis=0), **visualizer.styles["energy"]["total"],
            label=f"{prefix}Energy Across All Tasks", ax=ax_energy)
        # Population energy per task
        for task_idx, task in enumerate(visualizer.tasks):
            task_name = visualizer.prettify_task_name(task)
            sns.lineplot(x=x, y=energy[task_idx], **visualizer.styles["energy"][task_name],
                label=f"{prefix}Energy on {task_name}", ax=ax_energy
            )

        if not normalize_energy:
            ax2.set_ylabel("Energy [pJ]")
            energy_ticks, step_size = visualizer.get_y_axis_ticks(energy, step=10)
            ax2.set_yticks(energy_ticks)
            ax2.set_ylim(0, energy_ticks[-1] + step_size / 2)

    handles, labels = visualizer.combine_legends(axes)
    plt.legend(handles, labels, fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig(Path(visualizer.output_path) / "ga_progress_population.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_ga_population(visualizer)
