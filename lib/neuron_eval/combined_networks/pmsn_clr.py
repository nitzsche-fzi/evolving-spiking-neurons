import torch
from .polynomial import Polynomial

from esn.surrogate_gradient.Heaviside import get_heaviside

class PMSN_CLR(torch.nn.Module):
    """PMSN-CLR neuron model: Polynomial Multi-State Neuron with Constant
    threshold and Linear Reset.

    Simulates a whole population of PMSN-CLR neurons in parallel. Each neuron's
    state evolves according to a learned multivariate :class:`Polynomial`; a
    spike is emitted when the first (membrane-potential) state variable crosses
    a constant threshold, after which a second polynomial defines the reset.
    """

    def __init__(
            self,
            param_list,
            batch_size,
            feature_size,
            device,
            detach_spikes=True
        ):
        super(PMSN_CLR, self).__init__()
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.device = device
        self.detach_spikes = detach_spikes

        poly_coeffs = []
        reset_coeffs = []
        resting_states = []
        thresholds = []

        self.population_size = len(param_list)
        self.n_states = param_list[0].n_states
        self.degree = param_list[0].degree
        self.state_delta = param_list[0].state_delta
        self.threshold_fn = get_heaviside(param_list[0].surrogate_method, param_list[0].surrogate_alpha)
        self.surrogate_alpha = param_list[0].surrogate_alpha

        for params in param_list:
            assert params.n_states == self.n_states, "All parameters must have the same number of states"
            assert params.degree == self.degree, "All parameters must have the same polynomial degree"
            assert params.state_delta == self.state_delta, "All parameters must have the same state delta"
            assert params.surrogate_method == param_list[0].surrogate_method, "All parameters must have the same surrogate method"
            assert params.surrogate_alpha == self.surrogate_alpha, "All parameters must have the same surrogate alpha"
            poly_coeffs.append(params.polynomial_coeffs)
            reset_coeffs.append(params.reset_coeffs)
            thresholds.append(params.threshold)
            resting_states.append(_sanitize_resting_state(params.resting_state, self.device))
            
        #convert to tensors
        poly_coeffs = torch.stack(poly_coeffs, dim=0).to(device)
        reset_coeffs = torch.stack(reset_coeffs, dim=0).to(device)
        thresholds = torch.tensor(thresholds, dtype=torch.float32, device=self.device).view(1, self.population_size, 1)
        resting_states = torch.stack(resting_states, dim=0).to(device)

        #poly coeff shape: (population_size, n_states, n_monomials)
        #reset coeff shape: (population_size, n_states, n_reset_monomials) 
        #thresholds shape: (population_size,)
        #resting_states shape: (population_size, n_states)

        self.state_polynomial = Polynomial(poly_coeffs, self.n_states+1, self.degree, batch_size, feature_size, device)
        self.reset_polynomial = Polynomial(reset_coeffs, self.n_states+1, 1, batch_size, feature_size, device)
        self.thresholds = thresholds
        self.resting_states = resting_states
        self.register_buffer("resting_state", resting_states.clone())
        self.register_buffer("threshold", thresholds.clone())


    def init_state(self) -> torch.Tensor:
        """
        Initializes the neuron states based on the resting state for each neuron in the population.
        Returns a tensor of shape (batch_size, population_size, n_states, feature_size).
        """
        initial_state_view = self.resting_state.view(1, self.population_size, self.n_states, 1)
        return initial_state_view.expand(
            self.batch_size, self.population_size, self.n_states, self.feature_size
        ).clone()


    def forward(self, x, state):
        """
        Performs the forward pass for the population of PMSN-CLR neurons.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, population_size, feature_size).
            state (torch.Tensor): Current state tensor of shape (batch_size, population_size, n_states, feature_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - spikes (torch.Tensor): Output spikes of shape (batch_size, population_size, feature_size).
                - new_state (torch.Tensor): Updated state tensor of shape (batch_size, population_size, n_states, feature_size).
        """
        # 1. Update state based on the state evolution polynomial
        state_update = self.state_polynomial(x, state)
        new_state = state + state_update * self.state_delta

        # 2. Generate spikes based on the threshold
        membrane_potential = new_state[:, :, 0, :] # Shape: (batch_size, population_size, feature_size)
        spikes = self.threshold_fn(membrane_potential - self.threshold)

        # 3. Calculate the reset values
        reset_values = self.reset_polynomial(x, new_state)

        # 4. Apply the reset mechanism to neurons that spiked
        if self.detach_spikes:
            detached_spikes = spikes.detach()
        else:
            detached_spikes = spikes
        detached_spikes_expanded = detached_spikes.unsqueeze(2)
        new_state += (reset_values - new_state) * detached_spikes_expanded
        
        return spikes, new_state


def _sanitize_resting_state(x, device):
    return torch.as_tensor(x, dtype=torch.float32).to(device)
