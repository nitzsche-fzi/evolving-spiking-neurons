"""Prints configuration and summary information about a saved GA run.

Reads a GA state pickle and prints the GA configuration and basic run summary
(number of generations, the keys stored per generation, etc.).

Example:
    python3 scripts/ga/ga_info.py path/to/ga_run.pkl
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import argparse
import yaml

from lib.neuron_eval.ga.manager import GAManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract GA run information.')
    parser.add_argument('input_file', type=str, help='Path to the input GA pickle file')
    args = parser.parse_args()

    manager = GAManager.load(args.input_file, args.input_file)

    print("GA Configuration:")
    print(yaml.safe_dump(manager.config, sort_keys=False))

    n_generations = len(manager.history)

    history_keys = manager.history[0].keys() if n_generations > 0 else []
    print("History keys in generation 0:", list(zip(list(history_keys), [type(manager.history[0][k]) for k in history_keys])))
    if n_generations > 1:
        history_keys_2 = manager.history[1].keys()
        print("History keys in generation 1:", list(zip(list(history_keys_2), [type(manager.history[1][k]) for k in history_keys_2])))

    print(f"Number of generations: {n_generations}")