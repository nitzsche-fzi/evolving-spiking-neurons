"""Tests that DenseLayer initialization yields preactivations with mean/variance ~= 1."""

import numpy as np
import torch
import pytest

from lib.neuron_eval.combined_networks.dense import DenseLayer


VAR_TOL = 0.12
MEAN_TOL = 0.05

@pytest.mark.parametrize("mode", ["continuous", "spike"])
def test_dense_layer_output_is_normalized(mode):
    """
    Tests whether each individual dense layer produces dot-products
    with mean ≈ 1 and variance ≈ 1, given correct initialization.

    The test avoids invalid configurations (e.g., mu_x <= 0 or tau_sq <= 0).
    """

    torch.manual_seed(123)
    np.random.seed(123)

    population_size = 5
    input_size = 512     # big enough for law of large numbers
    output_size = 1      # test statistics for one output dim
    batch_size = 4000    # enough samples to estimate distribution

    if mode == "continuous":
        # Use benign input statistics to avoid tau_sq <= 0.
        in_mean = 1.0
        in_var = 0.5

        layer = DenseLayer(
            population_size=population_size,
            input_size=input_size,
            output_size=output_size,
            seed=42,
            in_mean=in_mean,
            in_var=in_var,
            spike_rates=None,
        )

        # Sample Gaussian inputs matching the assumed distribution
        x = torch.randn(batch_size, population_size, input_size)
        x = x * np.sqrt(in_var) + in_mean

    elif mode == "spike":
        # Choose moderate spike rates to avoid tau_sq <= 0
        spike_rates = np.linspace(0.2, 0.8, population_size).tolist()

        layer = DenseLayer(
            population_size=population_size,
            input_size=input_size,
            output_size=output_size,
            seed=42,
            spike_rates=spike_rates,
        )

        # Sample Bernoulli spikes per individual
        xs = []
        for r in spike_rates:
            xs.append(torch.bernoulli(torch.full((batch_size, input_size), r)))
        x = torch.stack(xs, dim=1)  # (batch, population, input_size)

    # Run layer
    with torch.no_grad():
        y = layer(x)  # (batch, population, 1)

    # Now test each individual separately
    y_np = y.squeeze(-1).numpy()  # (batch, population)

    for i in range(population_size):
        vals = y_np[:, i]

        mean = vals.mean()
        var = vals.var()

        assert np.isclose(mean, 1.0, atol=MEAN_TOL), \
            f"Individual {i}: mean={mean:.4f}, expected ≈1"

        assert np.isclose(var, 1.0, atol=VAR_TOL), \
            f"Individual {i}: var={var:.4f}, expected ≈1"
