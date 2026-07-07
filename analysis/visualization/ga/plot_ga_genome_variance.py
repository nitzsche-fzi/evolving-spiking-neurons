"""
Plot the per-dimension variance of genomes across generations.

Each line corresponds to one genome dimension, showing how its variance evolves
over generations.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer


def plot_ga_genome_variance(visualizer: GAVisualizer):
    """Plot variance per genome dimension across generations."""
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)

    n_gens = visualizer.n_generations
    x = np.arange(n_gens)

    variances = []
    stop_dims = None
    for gen_idx in range(n_gens):
        gen = visualizer.manager.history[gen_idx]
        genomes, _ = visualizer.manager._extract_genomes(gen["params"], range(len(gen["params"])))
        nonzero = genomes != 0
        nonzero_counts = nonzero.sum(axis=0)
        if stop_dims is None:
            stop_dims = nonzero_counts <= 1
        else:
            stop_dims = np.logical_or(stop_dims, nonzero_counts <= 1)
        var_gen = np.full(genomes.shape[1], np.nan, dtype=np.float32)
        valid_dims = ~stop_dims
        for dim_idx in np.where(valid_dims)[0]:
            values = genomes[nonzero[:, dim_idx], dim_idx]
            if values.size > 1:
                std_value = np.std(values)
                if std_value > 0:
                    var_gen[dim_idx] = std_value
        variances.append(var_gen)

    variances = np.stack(variances, axis=0)  # shape (n_gens, n_dims)

    mean_std = np.nanmean(variances, axis=1)

    plt.figure(figsize=(12, 8))
    for dim_idx in range(variances.shape[1]):
        plt.plot(x, variances[:, dim_idx], color="tab:blue", alpha=0.25, linewidth=1.0)
    plt.plot(x, mean_std, color="black", linewidth=2.0, label="Mean Std Dev")

    plt.title(f"Genome Variance per Dimension ({variances.shape[1]} dims)")
    plt.xlabel("Generation")
    plt.ylabel("Std Dev")
    plt.yscale("log")
    plt.ylim(bottom=0.01)
    plt.xticks(range(n_gens))
    plt.grid(True, which="major", axis="both")
    plt.legend(fontsize="small", loc="upper right")
    plt.tight_layout()
    plt.savefig(Path(visualizer.output_path) / "ga_genome_variance.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ga_file", type=str, help="Path to the GA pickle file")
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_ga_genome_variance(visualizer)
