"""Tests that the sparse Polynomial module matches a dense reference evaluation."""

import itertools
import torch
import pytest

from lib.neuron_eval.combined_networks.polynomial import Polynomial


@pytest.mark.parametrize("device", ["cuda:0" if torch.cuda.is_available() else "cpu"])
def test_polynomial_matches_naive(device):
    torch.manual_seed(123)

    degree = 4
    n_variables = 4
    n_polynomials = 5
    sparsity = 0.95
    population_size = 250
    batch_size = 16
    feature_size = 64   # reduced for faster tests

    # ---- Build monomial index list ----
    pre_combinations = list(
        itertools.chain.from_iterable(
            itertools.combinations_with_replacement(range(n_variables), d)
            for d in range(degree + 1)
        )
    )
    n_monomials = len(pre_combinations)

    # ---- Coefficients ----
    coeffs = torch.randn(
        (population_size, n_polynomials, n_monomials),
        device=device,
        requires_grad=False,
    )
    coeffs = coeffs * (torch.rand_like(coeffs) > sparsity).float()

    # ---- Module ----
    p = Polynomial(
        coeffs, n_variables, degree, batch_size, feature_size, device
    ).to(device)

    # ---- Inputs ----
    x = torch.randn(
        (batch_size, population_size, feature_size),
        device=device,
        requires_grad=False,
    )
    state = torch.randn(
        (batch_size, population_size, n_variables - 1, feature_size),
        device=device,
        requires_grad=False,
    )

    # ---- Naive implementation ----
    def naive_polynomial_evaluation(x, state, coeffs, pre_combinations, n_variables):
        x_unsq = x.unsqueeze(2)
        all_vars = torch.cat((x_unsq, state), dim=2)

        mon_vals = []
        for combo in pre_combinations:
            mval = torch.ones_like(x)
            for var_idx in combo:
                mval = mval * all_vars[:, :, var_idx, :]
            mon_vals.append(mval.unsqueeze(2))

        all_monomials = torch.cat(mon_vals, dim=2)
        out = torch.einsum("bpmf,pnm->bpnf", all_monomials, coeffs)
        return out

    naive = naive_polynomial_evaluation(
        x, state, coeffs, pre_combinations, n_variables
    )
    compiled = p(x, state)

    assert naive.shape == compiled.shape

    # ---- Numerical closeness ----
    abs_diff = torch.abs(naive - compiled)
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()

    # These tolerances are intentionally loose because floating-point noise
    # accumulates slightly in your compiled version.
    assert mean_diff < 5e-6, f"Mean difference too large: {mean_diff}"
    assert max_diff < 1e-3, f"Max difference too large: {max_diff}"

    # ---- Ratio sanity check (only for non-zero compiled values) ----
    mask = compiled.abs() > 1e-6
    if mask.any():
        ratio = (naive[mask].abs() / compiled[mask].abs())
        ratio_mean = ratio.mean().item()

        # Ratio should hover around 1.0
        assert 0.95 < ratio_mean < 1.05, f"ratio mean abnormal: {ratio_mean}"

