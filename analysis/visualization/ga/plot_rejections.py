"""Plots the sampling rejection reasons across all generations of a GA run."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from analysis.visualization.ga.visualizer import GAVisualizer


def plot_rejections(visualizer: GAVisualizer):
    all_reasons = set()
    reports = list()
    for gen in visualizer.manager.history:
        report = gen.get("report") or gen.get("sampling_report")
        if report is None:
            continue
        reports.append(report)
        all_reasons.update(report.keys())

    if not reports:
        print("No rejection reports found in the GA history. skipping rejection plot.")
        return

    all_reasons = sorted(all_reasons)
    reason_to_idx = {reason: i for i, reason in enumerate(all_reasons)}

    # Build count matrix
    n_reasons = len(all_reasons)
    data = np.zeros((visualizer.n_generations, n_reasons), dtype=int)
    total_rejections = np.zeros(visualizer.n_generations, dtype=int)

    for i, report in enumerate(reports):
        for reason, val in report.items():
            data[i, reason_to_idx[reason]] = val
        total_rejections[i] = data[i].sum()

    # Normalize per generation
    normalized_data = np.zeros_like(data, dtype=float)
    np.divide(data, total_rejections[:, None], out=normalized_data, where=total_rejections[:, None] != 0)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Plot in normal order
    bottom = np.zeros(visualizer.n_generations)
    bars = []
    labels = []
    for reason in all_reasons:
        idx = reason_to_idx[reason]
        reason_name = _prettify_reason_name(reason)
        bar = ax1.bar(range(visualizer.n_generations), normalized_data[:, idx], bottom=bottom, label=reason_name)
        bottom += normalized_data[:, idx]
        bars.append(bar)
        labels.append(reason_name)

    ax1.set_ylabel("Fraction of Rejection Reasons")
    ax1.set_title("Normalized Rejection Reasons per Generation")
    ax1.set_xlim(right=visualizer.n_generations - 1)

    # Manually reverse legend order to match stack top-down
    ax1.legend(bars[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Bottom plot: total rejections (log scale) with grid
    ax2.plot(range(visualizer.n_generations), total_rejections, marker='o')
    ax2.set_yscale("log")
    ax2.set_ylabel("Total Rejections (log)")
    ax2.set_xlabel("Generation")
    ax2.set_title("Total Number of Rejected Samples per Generation")
    ax2.grid(axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(Path(visualizer.output_path) / "rejection_reasons.jpg")
    # plt.close()

def _prettify_reason_name(reason: str) -> str:
    return reason.replace("_", " ").title()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ga_file', type=str, help='Path to the GA pickle file')
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_rejections(visualizer)
