"""Samples an initial population of PMSN-CLR neurons and saves it to disk.

Draws neurons from the distribution defined by a sampling config, keeping only
those that pass the validity filters in :func:`lib.sampling.pmsn_clr.sample`,
and pickles their parameters as the starting population for a GA run.

Example:
    python3 scripts/sampling/sample_neurons.py configs/sampling/n2d2.yaml 1000 \\
        results/sampled_neurons/1000x_n2d2.pkl
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))


import yaml
import argparse
import os
import pickle
import numpy as np

from lib.sampling.pmsn_clr import sample, get_initial_sampler

from esn.neuron_classes import PMSN_CLR

def load_neurons(neurons_file):
    """Loads the neurons from the given file."""
    with open(neurons_file, "rb") as f:
        data = pickle.load(f)
        neurons = [PMSN_CLR(params) for params in data["params"]]
    return {
        "neurons": neurons,
        "n_neurons": data["n_neurons"],
        "report": data["report"],
        "sampling_config": data["sampling_config"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample neurons for a task')
    parser.add_argument('sampling_config', type=str, help='Path to the sampling config yaml file')
    parser.add_argument('n_neurons', type=int, help='Number of neurons to sample. Size of the population of the genetic algorithm down the line')
    parser.add_argument('output_file', type=str, help='Path to the output file containing the sampled neurons')
    args = parser.parse_args()

    with open(args.sampling_config) as f:
        sampling_config = yaml.safe_load(f)

    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler = get_initial_sampler(
        sampling_config['n_states'],
        sampling_config['degree'],
        sampling_config['threshold_mean'], #10.0
        sampling_config['threshold_std'], #4.0
        sampling_config['state_delta'],
        "scaled_superspike",
        0.0,
        sampling_config['polynomial_std'],
        sampling_config['reset_std'],
        sampling_config['min_terms_per_polynomial'],
        sampling_config['max_terms_per_polynomial'],
        sampling_config['min_terms_per_reset'],
        sampling_config['max_terms_per_reset'],
        sampling_config['enforce_degree'])

    neurons, report = sample(
        sampler,
        n_neurons=args.n_neurons,
        n_states=sampling_config['n_states'], #2
        max_rs_steps=sampling_config['max_rs_steps'], #1000
        rs_delta_eps=sampling_config['rs_delta_eps'], #1e-4
        max_state_abs=sampling_config['max_state_abs'], #20.0
        n_trials = sampling_config['n_trials'], #100
        n_trial_steps = sampling_config['n_trial_steps'], #200
        min_spike_rate = sampling_config['min_spike_rate'], #0.01, #1% spiking
        max_spike_rate = sampling_config['max_spike_rate'], #0.05, #5% spiking
        max_it = sampling_config.get('max_it', 1000), #1000
        report_failed=True
    )

    data = {
        "n_neurons": args.n_neurons,
        "params": [neuron.params for neuron in neurons],
        "sampling_report": report, 
        "sampling_config": sampling_config
    }

    with open(args.output_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Sampled {len(neurons)} neurons and saved to {args.output_file}")
    print("Report:")
    for key, value in report.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.tolist()}")
        else:
            print(f"{key}: {value}")
