import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from analysis.visualization.ga.visualizer import GAVisualizer


def plot_ga_energy(visualizer: GAVisualizer):
    """Plots per-generation energy and spike-rate curves for the best neuron.

    For each fitness metric it draws the best neuron's total energy, hardware
    'spike out' energy and spike rate, alongside the best overall hardware
    'spike out' energy across the population. Does nothing if the run has no
    hardware results.
    """
    if not visualizer.has_hardware:
        print("[WARN _plot_ga_energy]: There are no hardware results in the pickle file. Skipping energy plots.")
        return

    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
    x = list(range(visualizer.n_generations))
    

    top_individuals = visualizer._find_top_individuals(consider_past_gens=True) # shape (n_gens)
    gen_lowest_energy_hw_so = np.array([top_individuals[gen_idx]["energy_hw"]["hw_metrics"][1] for gen_idx in range(visualizer.n_generations)])

    for metric in visualizer.relevant_metrics:
        best_energy = np.array(
            [list(top_individuals[gen_idx][metric]["energy"].values()) for gen_idx in range(visualizer.n_generations)]
        ).sum(axis=1)
        best_energy_hw_so  = np.array([top_individuals[gen_idx][metric]["hw_metrics"][1] for gen_idx in range(visualizer.n_generations)])
        best_spike_rates = visualizer.get_best_individual_property("spike_rate", metric) * 100

        # Plotting
        plt.figure(figsize=(12, 8))
        ax_energy = plt.gca()
        ax_acc_sr = ax_energy.twinx()

        ###### ENERGY #######
        # Energy of best neuron
        sns.lineplot(x=x, y=best_energy, **visualizer.styles["energy"]["total"], label="Total Energy", alpha=0.7, ax=ax_energy)
        # Plot hardware (spike out) energy of best neuron
        sns.lineplot(x=x, y=best_energy_hw_so, **visualizer.styles["energy"]["hw_so"], label="Hardware Energy", ax=ax_energy)
        # Plot lowest hardware (spike out) energy 
        sns.lineplot(x=x, y=gen_lowest_energy_hw_so, **visualizer.styles["energy"]["hw_low"], label="Lowest Hardware Energy in Generation", ax=ax_energy)
        # Plot empty line to better align the legends
        sns.lineplot(x=x, y=np.zeros(visualizer.n_generations), label=" ", color="white", ax=ax_energy)

        ###### SPIKE RATE #######
        # Plot spike rates per task of best neuron
        for i, task in enumerate(visualizer.tasks):
            task_name = visualizer.prettify_task_name(task)
            sns.lineplot(x=x, y=best_spike_rates[i], **visualizer.styles["spike_rate"][task_name], label=f"Spike Rate {task_name}", ax=ax_acc_sr)
        

        plt.title(f"Best {metric.capitalize()} Neuron")
        plt.xlabel("Generation")
        ax_acc_sr.set_ylabel("Spike Rate [%]")
        acc_sr_ticks, step_size = visualizer.get_y_axis_ticks(best_spike_rates, step=1)
        ax_acc_sr.set_yticks(acc_sr_ticks)
        ax_acc_sr.set_ylim(0, acc_sr_ticks[-1] + step_size / 2)
        ax_energy.set_ylabel("Energy [pJ]")
        energy_ticks, step_size = visualizer.get_y_axis_ticks([best_energy_hw_so, gen_lowest_energy_hw_so, best_energy], step=10)
        ax_energy.set_yticks(energy_ticks)
        ax_energy.set_ylim(0, energy_ticks[-1] + step_size / 2)
        plt.xticks(range(visualizer.n_generations))  # Force integer x-axis ticks
        plt.grid(True, which='major', axis='both')
        
        handles, labels = visualizer.combine_legends([ax_acc_sr, ax_energy])
        plt.legend(handles, labels, fontsize='small', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        plt.savefig(Path(visualizer.output_path) / f"ga_energy_best_{metric}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_ga_energy(visualizer)
