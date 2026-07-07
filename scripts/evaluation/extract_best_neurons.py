"""Extracts the fittest neurons from a completed GA run and saves them.

Selects the top ``n_best`` individuals by fitness and writes their parameters
to a pickle in the same ``{"n_neurons", "params"}`` format as the
``sample_neurons*.py`` scripts, so the result can be fed straight into
``evaluate_neurons.py`` for final performance metrics. With ``--write-config``
the run's GA config is also written alongside the output as ``config.yaml``.

Example:
    python3 scripts/evaluation/extract_best_neurons.py results/ga/n2d3_01.pkl 20 \\
        results/best_neurons/20x_n2d3_01.pkl
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import pickle
import argparse
import yaml
from pathlib import Path

if __name__ == "__main__":
    from esn.neuron_classes import PMSN_CLR
    from lib.neuron_eval.ga.manager import GAManager 
    from lib.neuron_eval.ga.genome import reconstruct_genome

    parser = argparse.ArgumentParser(description='Extract best neurons from a GA run.')
    parser.add_argument('input_file', type=str, help='Path to the input GA pickle file')
    parser.add_argument('n_best', type=int, help='Number of best neurons to extract')
    parser.add_argument('output_file', type=str, help='Path to the output pickle file')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode', default=False)
    parser.add_argument('--write-config', action='store_true', help='Also write GA config from the pickle to the output folder as config.yaml')
    args = parser.parse_args()

    manager = GAManager.load(args.input_file, args.input_file)

    neurons = [reconstruct_genome(g,i) for g,i in zip(*manager._select_top_individuals(args.n_best))]
    if args.verbose:
        for neuron in neurons:
            print(neuron)

    # Save in consistent format
    data = {
        "n_neurons": len(neurons),
        "params": neurons
    }

    # Make all the directories if they don't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, "wb") as f:
        pickle.dump(data, f)
    
    if args.write_config:
        config_out = Path(args.output_file).parent / "config.yaml"
        with open(config_out, "w") as f:
            yaml.safe_dump(manager.config, f, sort_keys=False)
    
    print(f"Saved {len(neurons)} best neurons to {args.output_file}")
