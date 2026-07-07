"""Visual test: plots the preactivation statistics of DenseLayer initialization."""

import numpy as np
import torch
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

from lib.neuron_eval.combined_networks.dense import DenseLayer


@pytest.mark.visual
def test_dense_initialization_stats_sweep_visual():
    """
    Visual sweep for DenseLayer initialization.

    Spike mode:
        One plot per spike rate:
            0.005, 0.01, 0.05, 0.1, 0.2

    Continuous mode:
        One plot per (in_mean, in_var) pair.

    Each plot:
        - all individuals overlaid
        - N(1,1) reference
        - top-right metadata box
    """

    torch.manual_seed(123)
    np.random.seed(123)

    ROOT = Path(__file__).resolve().parents[1]
    BASE = ROOT / "tmp" / "visual_tests" / "dense_init_sweep"
    BASE.mkdir(parents=True, exist_ok=True)

    print(f"[visual sweep] Saving plots in: {BASE}")

    # ----------------------------------------------------------
    # Global config
    # ----------------------------------------------------------
    population_size = 5
    input_size = 512
    output_size = 1
    batch_size = 5000

    # ----------------------------------------------------------
    # Spike-rate sweep: one image per rate
    # ----------------------------------------------------------
    spike_rates = [0.005, 0.01, 0.05, 0.1, 0.2]

    # ----------------------------------------------------------
    # Continuous configs
    # ----------------------------------------------------------
    continuous_stats = [
        (1.0, 0.5),
        (0.5, 0.1),
        (2.0, 1.0),
        (1.5, 0.2),
        (0.8, 0.05),
    ]

    def plot_single_config(all_vals, title, stats_text, outpng):
        """Single figure for a single config."""
        plt.figure(figsize=(7, 5))

        # Plot histograms for each individual
        for i, vals in enumerate(all_vals):
            plt.hist(vals, bins=80, density=True, alpha=0.4, label=f"ind {i}")

        # Reference N(1,1)
        concat = np.concatenate(all_vals)
        xs = np.linspace(concat.min(), concat.max(), 400)
        ref = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (xs - 1.0) ** 2)
        plt.plot(xs, ref, "r-", lw=2, label="N(1,1)")

        plt.axvline(1.0, color="black", linestyle="--", alpha=0.6)

        plt.title(title)
        plt.xlabel("Output value")
        plt.ylabel("Density")

        # metadata in top-right
        plt.text(
            0.98, 0.98,
            stats_text,
            fontsize=8,
            va="top",
            ha="right",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

        plt.legend()
        plt.savefig(outpng, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[visual sweep] Saved {outpng}")

    # ----------------------------------------------------------
    # Spike mode: one plot per single rate
    # ----------------------------------------------------------
    for r in spike_rates:
        outpng = BASE / f"spike_{r:.3f}.png"

        layer = DenseLayer(
            population_size=population_size,
            input_size=input_size,
            output_size=output_size,
            seed=42,
            spike_rates=[r] * population_size,   # each individual gets this same rate
        )

        # Bernoulli input for ALL individuals at that rate
        x = torch.bernoulli(torch.full((batch_size, population_size, input_size), r))

        with torch.no_grad():
            y = layer(x).squeeze(-1).numpy()

        all_vals = [y[:, i] for i in range(population_size)]

        stats_text = (
            f"mode = spike\n"
            f"spike_rate = {r}\n"
            f"population_size = {population_size}\n"
            f"input_size = {input_size}\n"
            f"batch_size = {batch_size}"
        )

        plot_single_config(
            all_vals,
            f"Dense Init — spike rate {r}",
            stats_text,
            outpng
        )

    # ----------------------------------------------------------
    # Continuous mode: one plot per (mean,var)
    # ----------------------------------------------------------
    for in_mean, in_var in continuous_stats:
        outpng = BASE / f"cont_mean{in_mean:.2f}_var{in_var:.3f}.png"

        layer = DenseLayer(
            population_size=population_size,
            input_size=input_size,
            output_size=output_size,
            seed=42,
            in_mean=in_mean,
            in_var=in_var,
        )

        x = torch.randn(batch_size, population_size, input_size)
        x = x * np.sqrt(in_var) + in_mean

        with torch.no_grad():
            y = layer(x).squeeze(-1).numpy()

        all_vals = [y[:, i] for i in range(population_size)]

        stats_text = (
            f"mode = continuous\n"
            f"in_mean = {in_mean}\n"
            f"in_var = {in_var}\n"
            f"population_size = {population_size}\n"
            f"input_size = {input_size}\n"
            f"batch_size = {batch_size}"
        )

        plot_single_config(
            all_vals,
            f"Dense Init — mean={in_mean}, var={in_var}",
            stats_text,
            outpng
        )
