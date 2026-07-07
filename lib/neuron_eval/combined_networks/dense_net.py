import torch

from .dense import DenseLayer
from .pmsn_clr import PMSN_CLR
from .integrator import Integrator

class DenseNet(torch.nn.Module):
    """A population of feed-forward spiking networks evaluated in parallel.

    Each individual in the population is a fully connected network whose hidden
    layers alternate dense projections (:class:`DenseLayer`) with a
    spiking neuron model, followed by a non-spiking
    :class:`Integrator` read-out layer. All individuals share the same
    ``network_shape`` and are simulated together along the population dimension,
    which is what makes evaluating a whole GA population efficient.
    """

    def __init__(
            self,
            seed,
            neuron_params,
            neuron_type,
            network_shape,
            batch_size,
            device,
            in_mean,
            in_var,
            dropout_p=0.0
        ):
        """
        Args:
            seed: Base seed for reproducible dense-layer initialization.
            neuron_params: One parameter object per individual in the
                population; its length sets the population size.
            neuron_type: Spiking neuron model to use.
            network_shape: Layer sizes including input and output, e.g.
                ``[in, hidden, ..., out]``.
            batch_size: Number of samples processed per forward pass.
            device: Torch device the network runs on.
            in_mean: Mean of the (continuous) network input, used to
                statistically match the first dense layer.
            in_var: Variance of the network input, used likewise.
            dropout_p: Currently unused; kept for interface compatibility.
        """
        super(DenseNet, self).__init__()
        if neuron_type != "pmsn_clr":
            raise ValueError(f"Unsupported neuron_type: {neuron_type}")
        torch.set_float32_matmul_precision('high')
        self.neuron_params = neuron_params 
        self.neuron_type = neuron_type
        self.network_shape = network_shape 
        self.batch_size = batch_size 
        self.device = device 
        self.population_size = len(neuron_params) 

        self.dense_layers = torch.nn.ModuleList() 
        self.n_spiking_neurons = len(network_shape) - 2 #the first and last layer are not spiking neurons 

        self.spike_rates = [neuron.spike_rate for neuron in neuron_params]

        for i in range(1,len(network_shape)):
            in_shape  = network_shape[i-1] 
            out_shape = network_shape[i] 
            if i == 1:
                self.dense_layers.append(DenseLayer(
                    self.population_size, in_shape, out_shape, seed=hash((seed,i)) % 1000000,
                    in_mean=in_mean, in_var=in_var, 
                ).to(self.device))
            else:
                self.dense_layers.append(DenseLayer(
                    self.population_size, in_shape, out_shape, seed=hash((seed,i)) % 1000000,
                    spike_rates=self.spike_rates
                ).to(self.device))
        self.spiking_neurons = torch.nn.ModuleList()

        for i in range(1,len(network_shape) - 1):
            feature_size = network_shape[i] 
            self.spiking_neurons.append(PMSN_CLR(
                neuron_params, batch_size, feature_size, device
            ))
        self.spiking_neurons.append(Integrator(self.population_size,self.batch_size,network_shape[-1],self.device)) 

    def init_state(self):
        """Builds the initial recurrent state for one forward rollout.

        Returns a list with one state tensor per spiking/read-out layer, plus a
        final per-individual scalar that accumulates the total spike count.
        """
        ret = []
        for neuron in self.spiking_neurons:
            ret.append(neuron.init_state())
        ret.append(torch.zeros((self.population_size,), device=self.device, dtype=torch.float32))
        return ret

    def forward(self,x,states):
        """Advances every network in the population by one time step.

        Args:
            x: Input at this time step, shape
                ``(batch_size, population_size, input_size)``.
            states: Recurrent state from the previous step, as returned by
                :meth:`init_state` or a prior :meth:`forward` call.

        Returns:
            Tuple ``(output, new_states)`` where ``output`` is the read-out of
            the final layer and ``new_states`` carries the updated state list
            (its last element is the running spike count).
        """
        new_states = []
        spike_count = states[-1]
        for i,(layer,neuron,state) in enumerate(zip(self.dense_layers,self.spiking_neurons,states[:-1])):
            x = layer(x)
            x,new_state = neuron(x,state)
            #x has shape [batch_size, population_size, features]
            if i < self.n_spiking_neurons:
                spike_count = spike_count + x.sum(dim=2).mean(dim=0)
            new_states.append(new_state)
        new_states.append(spike_count)
        return x, new_states

    def get_spiking_neuron_count(self):
        """Returns the number of spiking neurons in a single network.

        This counts one individual network, i.e. it is not multiplied by the
        population size.
        """
        return sum(self.network_shape[1:-1])
