import torch


class DenseLayer(torch.nn.Module):
    """
    A vectorized wrapper around a *population of dense layers*.

    The layer holds `population_size` independent dense layers
    (called *individuals*). Each individual has its own weight matrix
    W[i] ∈ R^{input_size × output_size} and receives inputs
    x[:, i, :] ∈ R^{batch × input_size}.

    The initializer computes, for each individual `i`, a statistically
    matched weight distribution such that the dot-product

        z = <x_i, w_i>

    has approximately mean 1 and variance 1 *under that individual's
    expected input statistics*.

    Input statistics can be specified in two ways:

        1. Continuous inputs:
           Provide `in_mean` and `in_var`.
           These statistics are shared across all individuals
           (e.g. the true data distribution at the input layer).

        2. Spike-based inputs:
           Provide `spike_rates` of length `population_size`.
           Each individual's input mean/variance is derived from its
           own spike rate r_i:  mean = r_i,  var = r_i(1 - r_i).

    This allows deeper layers to be initialized correctly even when
    different individuals fire at different rates.

    Result: every individual dense layer begins with comparable
    preactivation scales (≈ N(1, 1)) regardless of differences in
    firing statistics across the population.
    """


    def __init__(
        self,
        population_size,
        input_size,
        output_size,
        seed,
        in_mean=None,
        in_var=None,
        spike_rates=None,
    ):
        super().__init__()

        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size

        # Validate input specification mode
        if (bool(in_mean) != bool(in_var)):
            raise ValueError("Either both in_mean and in_var must be provided, or neither.")

        if (in_mean is not None and spike_rates is not None):
            raise ValueError(
                "Cannot use in_mean/in_var together with spike_rates. "
                "Choose either continuous or spike-based initialization."
            )

        # Resolve input statistics depending on mode
        if spike_rates is not None:
            if len(spike_rates) != population_size:
                raise ValueError(
                    f"Expected spike_rates length {population_size}, "
                    f"got {len(spike_rates)}."
                )
            inp_means = spike_rates
            inp_vars = [r * (1.0 - r) for r in spike_rates]
        else:
            inp_means = [in_mean] * population_size
            inp_vars = [in_var] * population_size

        # Ensure parameters exist
        assert inp_means is not None, "Input mean or spike rate must be provided."
        assert inp_vars is not None, "Input variance or spike rate must be provided."

        # Reproducibility
        torch.manual_seed(seed)

        # Allocate parameter storage: (pop, in, out)
        weight = torch.empty(
            population_size, input_size, output_size, dtype=torch.float32
        )

        # Initialize population blocks independently
        for i in range(population_size):
            n = input_size
            mu_x = float(inp_means[i])
            sigma_x_sq = float(inp_vars[i])

            if mu_x <= 0:
                raise ValueError(
                    f"mu_x must be > 0 for individual {i}, got {mu_x}."
                )

            # Weight mean so that E[ sum x_i w_i ] = 1
            mu_w = 1.0 / (n * mu_x)

            # Deviation variance so Var(sum x_i w_i) = 1
            tau_sq = (1.0 / (n * sigma_x_sq)) - (mu_w ** 2)
            if tau_sq <= 0:
                tau_sq = 1e-12
                print(
                    f"Warning: tau_sq <= 0 for individual {i}. "
                    "Falling back to small deviation. Var(sum x_i w_i) < 1."
                )

            tau = tau_sq ** 0.5

            # Initialize weights with zero-sum deviation across inputs (per output dim)
            w = torch.empty(n, output_size, dtype=torch.float32)
            for o in range(output_size):
                v = torch.randn(n, dtype=torch.float32) * tau
                v -= v.mean()  # zero-sum deviations -> controlled covariance structure
                w[:, o] = mu_w + v

            weight[i] = w

        self.weight = torch.nn.Parameter(weight, requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: (batch, population_size, input_size)

        Returns:
            (batch, population_size, output_size)
        """
        return torch.einsum("bpi,pio->bpo", x, self.weight)
