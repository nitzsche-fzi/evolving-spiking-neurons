"""Plots PMSN-CLR survivor genome slices across generations."""
import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
import seaborn as sns

from analysis.visualization.ga.visualizer import GAVisualizer
from lib.neuron_eval.ga.genome import reconstruct_genome


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


def _dimension_label(dim_index: int) -> str:
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


def _build_feature_matrix(params) -> tuple[torch.Tensor, int, int, int]:
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
    return X, n_polynomials, n_monomials, degree


def _filter_finite(X: torch.Tensor, fitness: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finite_mask = torch.isfinite(fitness)
    if not finite_mask.all():
        X = X[finite_mask]
        fitness = fitness[finite_mask]
    return X, fitness


def _select_pairs(X: torch.Tensor, fitness: torch.Tensor) -> list[tuple[int, int]]:
    eps = 1e-8
    stds = X.std(dim=0, unbiased=False)
    Xz = (X - X.mean(dim=0)) / (stds + eps)
    valid_mask = stds > 1e-12
    D = X.shape[1]

    scores = torch.zeros(D, dtype=torch.float32)
    for j in range(D):
        if not valid_mask[j]:
            continue
        scores[j] = _pearson_corr(Xz[:, j], fitness).abs()

    fitness_np = fitness.numpy()
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
    return pairs


def _collect_survivor_series(visualizer: GAVisualizer) -> list[tuple[int, torch.Tensor]]:
    series = []
    for gen_idx in range(len(visualizer.manager.history)):
        generations = range(gen_idx + 1)
        genomes, infos, accs, energy = visualizer.manager._collect_individuals(generations)
        if len(genomes) == 0:
            continue
        fitness = visualizer.manager._compute_fitness(accs, energy, method="worst_rank")
        survivor_idx = np.argsort(fitness)[: visualizer.manager.n_survivors]
        survivor_genomes = [genomes[i] for i in survivor_idx]
        survivor_infos = [infos[i] for i in survivor_idx]
        survivor_params = [
            reconstruct_genome(genome, info, visualizer.manager.neuron_type)
            for genome, info in zip(survivor_genomes, survivor_infos)
        ]
        X_survivor, _, _, _ = _build_feature_matrix(survivor_params)
        survivor_fitness = _to_float_tensor(fitness[survivor_idx])
        X_survivor, survivor_fitness = _filter_finite(X_survivor, survivor_fitness)
        if X_survivor.shape[0] == 0:
            continue
        series.append((gen_idx, X_survivor.detach().cpu().float()))
    return series


def _plot_survivor_progression(ax, pair, series, cmap, norm, n_gens):
    a, b = pair
    x_vals = []
    y_vals = []
    size_start = 15.0
    size_end = 5.0
    for gen_idx, X in series:
        x = X[:, a].numpy()
        y = X[:, b].numpy()
        color = cmap(norm(gen_idx))
        if n_gens > 1:
            t = gen_idx / (n_gens - 1)
        else:
            t = 0.0
        size = size_start + t * (size_end - size_start)
        ax.scatter(
            x,
            y,
            c=[color],
            s=size,
            alpha=0.35,
            edgecolors="none",
        )
        x_vals.append(x)
        y_vals.append(y)

    if x_vals and y_vals:
        x_all = np.concatenate(x_vals)
        y_all = np.concatenate(y_vals)
        if x_all.size > 0 and y_all.size > 0:
            ax.set_xlim(np.nanmin(x_all), np.nanmax(x_all))
            ax.set_ylim(np.nanmin(y_all), np.nanmax(y_all))

    ax.set_xlabel(_dimension_label(a))
    ax.set_ylabel(_dimension_label(b))
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3)


def plot_pmsn_clr_survivor_progression(visualizer: GAVisualizer):
    if visualizer.manager.neuron_type != "pmsn_clr":
        print(f"[INFO] Skipped: neuron population is '{visualizer.manager.neuron_type}', not PMSN_CLR.")
        return
    if len(visualizer.manager.history) < 2:
        print("[WARN] Need at least two generations to show survivor progression.")
        return

    sns.set_theme(style="whitegrid", context="talk")

    latest_idx = len(visualizer.manager.history) - 1
    latest_gen = visualizer.manager.history[latest_idx]
    latest_params = latest_gen.get("params", [])
    if not latest_params:
        print("[WARN] Latest generation has no parameters. Skipping plot.")
        return

    X_latest, n_polynomials, n_monomials, degree = _build_feature_matrix(latest_params)
    _, _, latest_accs, latest_energy = visualizer.manager._collect_individuals([latest_idx])
    latest_fitness = visualizer.manager._compute_fitness(latest_accs, latest_energy, method="worst_rank")
    latest_fitness_t = _to_float_tensor(latest_fitness)
    X_latest, latest_fitness_t = _filter_finite(X_latest, latest_fitness_t)
    if X_latest.shape[0] == 0:
        print("[WARN] All latest fitness values are NaN; no data to plot.")
        return

    pairs = _select_pairs(X_latest, latest_fitness_t)
    if not pairs:
        print("[WARN] No valid dimension pairs found for plotting.")
        return

    series = _collect_survivor_series(visualizer)
    if not series:
        print("[WARN] No survivor data found across generations.")
        return

    n_pairs = len(pairs)
    n_cols = 4
    rows = int(np.ceil(n_pairs / n_cols))
    fig = plt.figure(figsize=(24, 4 + rows * 3.5))
    outer = fig.add_gridspec(
        nrows=4,
        ncols=2,
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
        width_ratios=[1.0, 0.021],
        height_ratios=[0.02, 0.06, 0.04, 0.88],
    )

    title_ax = fig.add_subplot(outer[0, 0])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        "PMSN_CLR genome slices: survivor progression across generations",
        ha="center",
        va="center",
        fontsize=16,
    )

    equations_ax = fig.add_subplot(outer[1, 0])
    equations_ax.axis("off")
    equation_lines = _build_equation_lines(n_polynomials, degree)
    equation_lines.append(r"$x_1 > \mathrm{threshold} \Rightarrow \mathrm{reset}$")
    equations_ax.text(
        0.02,
        0.5,
        "\n".join(equation_lines),
        ha="left",
        va="center",
        fontsize=15,
    )

    label_ax = fig.add_subplot(outer[2, 0])
    label_ax.axis("off")
    col_centers = [0.125, 0.375, 0.625, 0.875]
    for x in col_centers:
        label_ax.text(x, 0.5, "Survivors", ha="center", va="center", fontsize=12)

    grid_spec = outer[3, 0].subgridspec(rows, n_cols, wspace=0.35, hspace=0.35)
    axes = np.array([[fig.add_subplot(grid_spec[i, j]) for j in range(n_cols)] for i in range(rows)])

    n_gens = len(visualizer.manager.history)
    cmap = cm.inferno
    norm = Normalize(vmin=0, vmax=max(1, n_gens - 1))

    for idx, pair in enumerate(pairs):
        ax = axes.flat[idx]
        _plot_survivor_progression(ax, pair, series, cmap, norm, n_gens)

    for ax in axes.flat[len(pairs) :]:
        ax.axis("off")

    cbar_ax = fig.add_subplot(outer[3, 1])
    pos = cbar_ax.get_position()
    cbar_ax.set_position([pos.x0 - 0.03, pos.y0, pos.width, pos.height])
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("generation")
    if n_gens > 1:
        ticks = np.linspace(0, n_gens - 1, min(n_gens, 5))
        cbar.set_ticks(np.unique(np.round(ticks)).astype(int))

    fig.savefig(Path(visualizer.output_path) / "pmsn_clr_genome_slices_survivor_progression.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ga_file", type=str, help="Path to the ga pickle file")
    args = parser.parse_args()

    visualizer = GAVisualizer(args.ga_file)
    plot_pmsn_clr_survivor_progression(visualizer)
