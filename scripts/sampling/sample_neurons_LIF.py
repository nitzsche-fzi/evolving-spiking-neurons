"""Samples a population of LIFBox neurons and saves it to disk.

Draws neurons with randomised parameters as a (LIF) baseline population for the
GA. The sampled ranges are:
- threshold in [0.5, 10.0]
- input_scale in [0.1, 10.0] (log scale)
- state_decay in [0.5, 0.999] (inverse log scale: closer to 1 more often, i.e.
  1 - log_uniform[0.001, 0.5])
- reset_value in [-10.0, 10.0] (log + sign scale; kept below threshold to avoid
  instant spiking)
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import os
import torch
import argparse
import pickle
import numpy as np
from tqdm import tqdm

from lib.neuron_eval.combined_networks.pmsn_clr import PMSN_CLR

from esn.neurons.clr import LIFBox

def get_random_LIFBox_params():
    threshold = np.random.uniform(0.5, 10.0)
    input_scale = 10 ** np.random.uniform(-1, 1)  # log scale
    state_decay = 1.0 - 10 ** np.random.uniform(np.log10(1-0.95), np.log10(1-0.7))  # inverse log scale
    reset_value_sign = np.random.choice([-1, 1])
    reset_value_magnitude = 10 ** np.random.uniform(0, 1) - 1.0 #to go from 0.0 to 9.0
    reset_value = reset_value_sign * reset_value_magnitude
    #ensure reset_value is never greater than threshold
    while reset_value >= threshold:
        reset_value_sign = np.random.choice([-1, 1])
        reset_value_magnitude = 10 ** np.random.uniform(0, 1) - 1.0
        reset_value = reset_value_sign * reset_value_magnitude

    return {
        "threshold": threshold,
        "input_scale": input_scale,
        "state_decay": state_decay,
        "reset_value": reset_value
    }


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Generate a population of LIFBox neurons with random parameters."
    )
    parser.add_argument("n_neurons", type=int, help="Number of neurons to generate")
    parser.add_argument("min_spike_rate", type=float, help="Minimum spike rate for neurons to be included")
    parser.add_argument("max_spike_rate", type=float, help="Maximum spike rate for neurons to be included")
    parser.add_argument("output_file", type=str, help="Path to output file to save neuron parameters")
    parser.add_argument("--verbose", action="store_true", default=False, help="Print selected neuron details")
    args = parser.parse_args()

    neuron_params_list = []
    spike_rate_list = []
    with tqdm(total=args.n_neurons, desc="Sampled neurons") as pbar:
        while len(neuron_params_list) < args.n_neurons:
            params = get_random_LIFBox_params()
            neuron = LIFBox(
                threshold=params["threshold"],
                input_scale=params["input_scale"],
                state_decay=params["state_decay"],
                reset_value=float(params["reset_value"])
            )
            #the resting state is always zero for LIFBox neurons
            resting_state = torch.tensor([0.0])
            neuron.params.resting_state = resting_state
            #run 1000 steps with N(1,1) input to estimate spike rate
            input_current = torch.randn(1000) + 1.0
            state = neuron.init_state(batch_size=1, n_neurons=1)
            spikes = []

            if args.verbose:
                print(".", end="", flush=True)
            for t in range(1000):
                spike, state = neuron.forward(input_current[t].unsqueeze(0), state)
                spikes.append(spike.item())
            spike_rate = np.mean(spikes)
            if (spike_rate >= args.min_spike_rate) and (spike_rate <= args.max_spike_rate):
                neuron.params.spike_rate = float(spike_rate)
                neuron_params_list.append(neuron.params)
                spike_rate_list.append(spike_rate)
                pbar.update(1)
                if args.verbose:
                    print(f"\nSelected neuron {len(neuron_params_list)}/{args.n_neurons}: params={params}, estimated spike rate={spike_rate:.3f}")
            
    import matplotlib.pyplot as plt
    plt.hist(spike_rate_list, bins=20)
    plt.title("Histogram of selected neuron spike rates")
    plt.xlabel("Spike Rate (spikes per 1000 steps)")
    plt.ylabel("Number of Neurons")
    plt.savefig("selected_neuron_spike_rates_histogram.png")
    print("\nSaved spike rate histogram to selected_neuron_spike_rates_histogram.png")

    output = {
        "params" : neuron_params_list,
        "n_neurons": args.n_neurons
    }

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output_file, "wb") as f:
        pickle.dump(output, f)
    print(f"Saved {len(neuron_params_list)} neuron parameters to {args.output_file}")

    #python sample_neurons_LIF.py 1000 0.01 0.2 results/neurons/sampled/1000x_lif.pkl && 
