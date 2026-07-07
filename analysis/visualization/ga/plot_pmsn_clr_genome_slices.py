"""Plots 2D slices of PMSN-CLR genome parameters against fitness for the latest generation."""
import argparse
from pathlib import Path

import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer


def _to_float_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x - x.mean()
    y = y - y.mean()
    cov = (x * y).mean()
    denom = x.std(unbiased=False) * y.std(unbiased=False) + eps
    return cov / denom


def _pair_r2(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(target)
    if mask.sum() < 3:
        return 0.0
    X = np.column_stack([np.ones(mask.sum()), x[mask], y[mask]])
    t = target[mask]
    coeffs, _, _, _ = np.linalg.lstsq(X, t, rcond=None)
    pred = X @ coeffs
    ss_res = np.sum((t - pred) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot <= 0:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return float(max(0.0, min(1.0, r2)))


def _dimension_label(dim_index: int, n_monomials: int) -> str:
    if dim_index == 0:
        return r"$\mathrm{threshold}$"
    flat_idx = dim_index - 1
    return rf"$c_{{{flat_idx + 1}}}$"


def _monomial_label(comb: tuple[int, ...]) -> str:
    if len(comb) == 0:
        return "1"
    counts = {}
    for idx in comb:
        counts[idx] = counts.get(idx, 0) + 1
    parts = []
    for idx in sorted(counts.keys()):
        power = counts[idx]
        base = "I" if idx == 0 else rf"x_{{{idx}}}"
        if power == 1:
            parts.append(base)
        else:
            parts.append(f"{base}^{{{power}}}")
    return r" \cdot ".join(parts)


def _build_equation_lines(n_polynomials: int, degree: int) -> list[str]:
    n_inputs = n_polynomials + 1
    pre_combinations = list(
        itertools.chain.from_iterable(
            itertools.combinations_with_replacement(range(n_inputs), d)
            for d in range(degree + 1)
        )
    )
    lines = []
    coeff_idx = 1
    for poly_idx in range(n_polynomials):
        terms = []
        for comb in pre_combinations:
            monomial = _monomial_label(comb)
            if monomial == "1":
                terms.append(rf"c_{{{coeff_idx}}}")
            else:
                terms.append(rf"c_{{{coeff_idx}}} \cdot {monomial}")
            coeff_idx += 1
        eq = rf"$\frac{{d}}{{dx_{{{poly_idx + 1}}}}} = " + " + ".join(terms) + r"$"
        lines.append(eq)
    return lines


def plot_pmsn_clr_genome_slices(visualizer: GAVisualizer):
    if visualizer.manager.neuron_type != "pmsn_clr":
        print(f"[INFO] Skipped: neuron population is '{visualizer.manager.neuron_type}', not PMSN_CLR.")
        return

    sns.set_theme(style="whitegrid", context="talk")

    generation_indices = [0, -1] if visualizer.n_generations > 1 else [-1]
    for gen_idx in generation_indices:
        generation = visualizer.manager.history[gen_idx]
        params = generation["params"]
        if not params:
            print(f"[WARN] No parameters found in generation {gen_idx}. Skipping plot.")
            continue

        coeffs_list = [_to_float_tensor(p.polynomial_coeffs) for p in params]
        coeffs = torch.stack(coeffs_list, dim=0)
        thresholds = torch.tensor(
            [float(_to_float_tensor(p.threshold).item()) for p in params],
            dtype=torch.float32,
        )
        pop_size = coeffs.shape[0]
        n_polynomials = coeffs.shape[1]
        n_monomials = coeffs.shape[2]
        degree = int(getattr(params[0], "degree", 0))

        coeffs_flat = coeffs.reshape(pop_size, n_polynomials * n_monomials)
        X = torch.cat([thresholds.reshape(pop_size, 1), coeffs_flat], dim=1)

        _, _, all_accs, all_energy = visualizer.manager._collect_individuals([gen_idx])
        fitness = visualizer.manager._compute_fitness(all_accs, all_energy, method="worst_rank")
        fitness_t = _to_float_tensor(fitness)

        finite_mask = torch.isfinite(fitness_t)
        if not finite_mask.all():
            X = X[finite_mask]
            fitness_t = fitness_t[finite_mask]

        if X.shape[0] == 0:
            print(f"[WARN] All fitness values are NaN; no data to plot (generation {gen_idx}).")
            continue

        X = X.detach().cpu().float()
        fitness_t = fitness_t.detach().cpu().float()

        eps = 1e-8
        stds = X.std(dim=0, unbiased=False)
        Xz = (X - X.mean(dim=0)) / (stds + eps)
        valid_mask = stds > 1e-12

        D = X.shape[1]
        scores = torch.zeros(D, dtype=torch.float32)
        for j in range(D):
            if not valid_mask[j]:
                continue
            scores[j] = _pearson_corr(Xz[:, j], fitness_t).abs()

        fitness_np = fitness_t.numpy()
        r2_1d = {}
        for j in range(D):
            if not valid_mask[j]:
                continue
            r2_1d[j] = _pair_r2(X[:, j].numpy(), np.zeros_like(fitness_np), fitness_np)
        ranked_dims = [int(j) for j in torch.argsort(scores, descending=True)]
        threshold_candidates = []
        for j in ranked_dims:
            if j == 0 or not valid_mask[j]:
                continue
            r2 = _pair_r2(X[:, 0].numpy(), X[:, j].numpy(), fitness_np)
            threshold_candidates.append((r2, 0, j))
        threshold_candidates.sort(key=lambda x: x[0], reverse=True)
        threshold_pairs = [(a, b) for _, a, b in threshold_candidates[:5]]
        plotted_dims = {dim for pair in threshold_pairs for dim in pair}

        candidate_dims = [j for j in ranked_dims if j != 0 and valid_mask[j]]
        candidate_dims = candidate_dims[: min(30, len(candidate_dims))]
        pair_candidates = []
        for idx, a in enumerate(candidate_dims):
            for b in candidate_dims[idx + 1 :]:
                r2 = _pair_r2(X[:, a].numpy(), X[:, b].numpy(), fitness_np)
                pair_candidates.append((r2, a, b))

        def _adjusted_score(r2: float, a: int, b: int, used_dims: set[int]) -> float:
            penalties = []
            if a in used_dims and a in r2_1d:
                penalties.append(r2_1d[a])
            if b in used_dims and b in r2_1d:
                penalties.append(r2_1d[b])
            penalty = max(penalties) if penalties else 0.0
            return r2 - penalty

        coeff_pairs = []
        remaining = pair_candidates[:]
        for _ in range(min(5, len(remaining))):
            best = None
            best_score = None
            for r2, a, b in remaining:
                score = _adjusted_score(r2, a, b, plotted_dims)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (r2, a, b)
            if best is None:
                break
            _, a, b = best
            coeff_pairs.append((a, b))
            plotted_dims.update([a, b])
            remaining = [item for item in remaining if not (item[1] == a and item[2] == b)]

        pairs = threshold_pairs + coeff_pairs
        pair_set = {tuple(sorted(p)) for p in pairs}
        if len(pairs) < 10:
            all_pairs = []
            valid_dims = [j for j in range(D) if valid_mask[j]]
            for idx, a in enumerate(valid_dims):
                for b in valid_dims[idx + 1 :]:
                    r2 = _pair_r2(X[:, a].numpy(), X[:, b].numpy(), fitness_np)
                    all_pairs.append((r2, a, b))
            remaining = [item for item in all_pairs if (item[1], item[2]) not in pair_set]
            while len(pairs) < 10 and remaining:
                best = None
                best_score = None
                for r2, a, b in remaining:
                    score = _adjusted_score(r2, a, b, plotted_dims)
                    if best_score is None or score > best_score:
                        best_score = score
                        best = (r2, a, b)
                if best is None:
                    break
                _, a, b = best
                key = (a, b) if a < b else (b, a)
                pairs.append((a, b))
                pair_set.add(key)
                plotted_dims.update([a, b])
                remaining = [item for item in remaining if not (item[1] == a and item[2] == b)]
        if not pairs:
            print(f"[WARN] No valid dimension pairs found for plotting (generation {gen_idx}).")
            continue

        fig, axes = plt.subplots(2, 5, figsize=(22, 9))
        axes = axes.flatten()
        equation_lines = _build_equation_lines(n_polynomials, degree)
        equation_lines.append(r"$x_1 > \mathrm{threshold} \Rightarrow \mathrm{reset}$")
        fig.text(
            0.5,
            0.99,
            "\n".join(equation_lines),
            ha="center",
            va="top",
            fontsize=10,
        )
        sc = None
        for ax, (a, b) in zip(axes, pairs):
            x = X[:, a].numpy()
            y = X[:, b].numpy()
            sc = ax.scatter(
                x,
                y,
                c=fitness_t.numpy(),
                cmap=cm.viridis_r,
                s=10,
                alpha=0.7,
                edgecolors="none",
            )
            ax.set_xlabel(_dimension_label(a, n_monomials))
            ax.set_ylabel(_dimension_label(b, n_monomials), rotation=90, labelpad=10, va="center")
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_label_coords(1.05, 0.5)
            redundancy = _pearson_corr(Xz[:, a], Xz[:, b]).abs().item()
            r2_pct = 100.0 * _pair_r2(x, y, fitness_t.numpy())
            ax.set_title(
                f"score={scores[a].item():.2f}+{scores[b].item():.2f}, var={r2_pct:.0f}%",
                fontsize=10,
            )
            ax.grid(True, alpha=0.3)

        for ax in axes[len(pairs) :]:
            ax.axis("off")

        fig.subplots_adjust(right=0.88, top=0.74, wspace=0.55, hspace=0.4)
        if sc is not None:
            cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(sc, cax=cbar_ax)
            cbar.set_label("fitness")

        generation_label = "gen0" if gen_idx == 0 else "latest"
        fig.suptitle(
            f"PMSN_CLR genome slices vs fitness ({generation_label})",
            fontsize=16,
            y=0.86,
        )
        fig.savefig(Path(visualizer.output_path) / f"pmsn_clr_genome_slices_{generation_label}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ga_file", type=str, help="Path to the ga pickle file")
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_pmsn_clr_genome_slices(visualizer)
