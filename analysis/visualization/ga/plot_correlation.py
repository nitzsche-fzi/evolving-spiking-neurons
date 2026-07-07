"""Plots genome correlation matrices and their eigenvalue progression over generations."""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer
from lib.neuron_eval.ga.genome import flatten_genome

from sparse_data_modeling import SparseDataModel


def plot_correlation(visualizer: GAVisualizer):
    """Plot correlation matrices and the progression of the eigenvalues of the correlation matrices over generations."""
    correlation_matrices = []
    for parents in visualizer.manager.history:
        genomes = []

        for param in parents["params"]:
            genome, _ = flatten_genome(param)
            genomes.append(genome)
        genomes = np.array(genomes)  # shape: (n_survivors, genome_length)
        model = SparseDataModel(genomes) 
        correlation_matrices.append(model.get_analysis_corr())

    sns.set_theme(style="whitegrid", context="talk")
    _plot_correlation_matrices(visualizer, correlation_matrices)
    _plot_eigenvalues_spectra(visualizer, correlation_matrices)


def _plot_correlation_matrices(visualizer: GAVisualizer, correlation_matrices: list):
    for i, corr in enumerate(correlation_matrices):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="bwr", ax=ax, vmin=-1, vmax=1)
        ax.set_title(f'Correlation Matrix - Generation {i}')
        plt.tight_layout()
        plt.savefig(Path(visualizer.output_path) / f"corr_matrix_gen_{i}.jpg")
        plt.close()

def _plot_eigenvalues_spectra(visualizer: GAVisualizer, correlation_matrices: list):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = cm.viridis(np.linspace(0, 1, visualizer.n_generations))

    for i, (corr, color) in enumerate(zip(correlation_matrices, colors)):
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.sort(eigvals)[::-1]
        ax.plot(eigvals, label=f'Gen {i}', color=color, alpha=0.8)

    sm = cm.ScalarMappable(cmap=cm.viridis, norm=mcolors.Normalize(vmin=0, vmax=visualizer.n_generations - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Generation", rotation=270, labelpad=15)

    plt.title("Sorted Eigenvalues of Correlation Matrices Over Generations")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Eigenvalue")
    #plt.yscale('log')  # Useful if eigenvalues vary over many orders of magnitude
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(Path(visualizer.output_path) / "eigenvalues_spectra_progression.jpg")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_correlation(visualizer)
