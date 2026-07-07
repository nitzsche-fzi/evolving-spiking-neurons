import torch
from typing import Tuple

class Integrator(torch.nn.Module):
    """
    A population-based leaky integrator neuron model.

    This module simulates a population of simple integrator neurons, where each neuron
    integrates its corresponding input feature over time. The entire population
    operates in parallel on a batch of data.
    """

    def __init__(self, population_size: int, batch_size: int, feature_size: int, device: torch.device):
        """
        Initializes the Integrator module.

        Args:
            population_size (int): The number of individual integrator neurons in the population.
            batch_size (int): The number of samples in a batch.
            feature_size (int): The dimensionality of the input and state for each neuron.
            device (torch.device): The device (e.g., 'cpu' or 'cuda') on which to perform computations.
        """
        super(Integrator, self).__init__()
        self.population_size = population_size
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.device = device

    def init_state(self) -> torch.Tensor:
        """
        Initializes the state for the entire population of integrator neurons.
        The state is initialized to zeros.

        Returns:
            torch.Tensor: A zero tensor representing the initial state, with a shape of
                          [batch_size, population_size, feature_size].
        """
        initial_state = torch.zeros(
            self.batch_size, 
            self.population_size, 
            self.feature_size, 
            device=self.device
        )
        return initial_state

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_state = state + x
        return new_state, new_state
