"""Shared helper for tests that need a population of real neuron parameters."""

import copy

from esn.neurons.clr import N2D2


def create_test_neuron_params(population_size: int):
    """Returns ``population_size`` independent copies of the evolved N2D2 params."""
    base_neuron = N2D2()
    return [copy.deepcopy(base_neuron.params) for _ in range(population_size)]
