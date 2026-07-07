"""Resumes a GA run from a saved state pickle.

Continues evolving a previously started run, optionally overriding the GA config,
and writes the updated state back to the same file. The number of generations
can be passed on the command line or entered interactively.

Example:
    python3 scripts/ga/resume_ga.py results/ga/n2d2_01.pkl 10
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import yaml
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch

from lib.utils import mute_torch_jit

torch._dynamo.config.recompile_limit = 100000
torch._dynamo.config.accumulated_recompile_limit = 100000


def _query_generations_to_evaluate() -> int:
    while True:
        try:
            inp = input("Enter number of generations to evaluate: ").strip()
            n_generations = int(inp)
            if n_generations <= 0:
                print("Please enter a positive integer.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")
    return n_generations


if __name__ == "__main__":
    from esn.neuron_classes import PMSN_CLR
    from lib.neuron_eval.ga.manager import GAManager 

    parser = argparse.ArgumentParser(description='Resume the genetic algorithm')
    parser.add_argument('ga_file', type=str, help='Path to the ga pickle')
    parser.add_argument('n_generations', type=int, nargs='?', default=None, help='Number of generations to run')
    parser.add_argument('ga_config', type=str, nargs='?', default=None, help='Optional path to override config yaml file')
    args = parser.parse_args()
    # The run is resumed in place: the input pickle is also the output file.
    output_file = args.ga_file

    # Override GA config if provided
    if args.ga_config is not None:
        with open(args.ga_config, "rb") as f:
            ga_config = yaml.safe_load(f)
    else:
        ga_config = None

    mute_torch_jit()

    manager = GAManager.load(args.ga_file, output_file, ga_config)

    # Query generations to evaluate interactively if not provided
    if args.n_generations is None:
        args.n_generations = _query_generations_to_evaluate()

    manager.show_gpu_dialog()

    for i in range(args.n_generations):
        manager.sample_population()
        manager.eval_population()