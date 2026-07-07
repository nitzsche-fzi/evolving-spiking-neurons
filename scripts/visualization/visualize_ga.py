#!/usr/bin/env python3
"""Generates the full set of GA diagnostic plots for a saved run.

Loads a GA state pickle and runs every GA visualiser (progress, population,
energy, rejections, survivors, genome slices, ...) over it.

Example:
    python3 scripts/visualization/visualize_ga.py results/ga/n2d2_hw60.pkl
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import argparse

from analysis.visualization.ga.visualizer import GAVisualizer
from analysis.visualization.ga.plot_ga_progress import plot_ga_progress
from analysis.visualization.ga.plot_ga_population import plot_ga_population
from analysis.visualization.ga.plot_ga_energy import plot_ga_energy
from analysis.visualization.ga.plot_rejections import plot_rejections
from analysis.visualization.ga.plot_survivor_dist import plot_survivors
from analysis.visualization.ga.plot_fitness_methods import plot_fitness_methods
from analysis.visualization.ga.plot_correlation import plot_correlation
from analysis.visualization.ga.plot_train import plot_train
from analysis.visualization.ga.plot_spike_rates import plot_spike_rates
from analysis.visualization.ga.plot_pmsn_clr_genome_slices import plot_pmsn_clr_genome_slices
from analysis.visualization.ga.plot_pmsn_clr_eda_vs_survivors import plot_pmsn_clr_eda_vs_survivors
from analysis.visualization.ga.plot_ga_genome_variance import plot_ga_genome_variance
from analysis.visualization.ga.plot_pmsn_clr_survivor_progression import plot_pmsn_clr_survivor_progression

def main():
    parser = argparse.ArgumentParser(description='Visualize GA progress')
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    parser.add_argument('smoothing', type=int, nargs='?', default=20,)
    args = parser.parse_args()

    print(f"Starting visualizations for {args.ga_file} with options:")
    print(f"  smoothing: {args.smoothing}")
    visualizer = GAVisualizer(args.ga_file)
    plot_pmsn_clr_survivor_progression(visualizer)
    plot_ga_genome_variance(visualizer)
    plot_pmsn_clr_eda_vs_survivors(visualizer)
    plot_pmsn_clr_genome_slices(visualizer)
    plot_ga_progress(visualizer)
    plot_ga_population(visualizer)
    plot_ga_energy(visualizer)
    plot_rejections(visualizer)
    plot_survivors(visualizer)
    plot_fitness_methods(visualizer)
    plot_correlation(visualizer)
    plot_train(visualizer, args.smoothing)
    plot_spike_rates(visualizer, args.smoothing)
    print(f"Finished visualizations for {args.ga_file}")

if __name__ == "__main__":
    main()
