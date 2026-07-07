"""Visual test: plots the normalized Heaviside surrogate gradient functions."""

import numpy as np
import torch
import pytest
import matplotlib.pyplot as plt
from pathlib import Path

from esn.surrogate_gradient.NormalizedHeaviside import (
    get_normalized_heaviside,
    superspike,
    scaled_superspike,
    atan_derivative,
    straight_through,
    triangular,
    gaussian,
)

@pytest.mark.visual
def test_heaviside_surrogate_curve_visual():

    torch.manual_seed(123)
    np.random.seed(123)

    ROOT = Path(__file__).resolve().parents[1]
    BASE = ROOT / "tmp" / "visual_tests" / "surrogate_gradient"
    BASE.mkdir(parents=True, exist_ok=True)

    shape = (16, 2, 8)

    surrogate_map = dict(
        superspike=superspike,
        scaled_superspike=scaled_superspike,
        atan_derivative=atan_derivative,
        straight_through=straight_through,
        triangular=triangular,
        gaussian=gaussian,
    )

    dists = [
        ("N(0,10)",  dict(mean=0.0, std=10.0)),
        ("N(-3,3)",  dict(mean=-3.0, std=3.0)),
        ("N(0,0.1)", dict(mean=0.0, std=0.1)),
    ]

    alphas = [0.1, 1.0, 10.0]

    # One figure per surrogate method: 3x3 grid (dist x alpha)
    for method_name, surrogate_func in surrogate_map.items():

        fig, axs = plt.subplots(
            nrows=len(dists),
            ncols=len(alphas),
            figsize=(18, 14),
            sharex=False,
            sharey=False,
        )

        for row, (dist_label, params) in enumerate(dists):
            for col, alpha in enumerate(alphas):

                ax = axs[row][col]

                # Construct normalized heaviside for this alpha
                H = get_normalized_heaviside(
                    surrogate_method=method_name,
                    alpha=alpha,
                    reduce_dims=(0, 2),
                    beta=0.99,
                )

                # Sample data for this distribution
                x = torch.randn(*shape) * params["std"] + params["mean"]

                spread = H._compute_instant_spread(x)
                alpha_norm = alpha / (spread.mean() + 1e-12)

                # Determine plotting range
                x_flat = x.cpu().numpy().flatten()
                lo = x_flat.min()
                hi = x_flat.max()
                lo -= 0.1 * (hi - lo)
                hi += 0.1 * (hi - lo)

                xs = torch.linspace(lo, hi, 1000)
                ones = torch.ones_like(xs)

                with torch.no_grad():
                    curve = surrogate_func(ones, xs, alpha_norm).cpu().numpy()
                    scatter_vals = surrogate_func(torch.ones_like(x), x, alpha_norm).cpu().numpy().flatten()

                ax.plot(xs.numpy(), curve, lw=2)
                ax.scatter(x_flat, scatter_vals, s=8, alpha=0.35, color="red")

                ax.set_title(f"{dist_label}, α={alpha}\nα_norm={float(alpha_norm):.3g}")
                ax.grid(alpha=0.3)

                if row == len(dists)-1:
                    ax.set_xlabel("x")
                if col == 0:
                    ax.set_ylabel("dσ/dx")

        fig.suptitle(f"Surrogate derivative — {method_name} (3×3 grid)", fontsize=16)

        outpng = BASE / f"{method_name}.png"
        fig.savefig(outpng, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"[visual] Saved {outpng}")
