"""Truncates a saved GA run to keep only its first N generations.

Loads a GA state pickle, drops every generation after the first ``n_generations``,
and writes the result to a new output file (the input is left untouched).

Example:
    python3 scripts/ga/truncate_ga_file.py results/ga/dvs_n2d2.pkl 1 \\
        results/ga/dvs_n2d2_truncated.pkl
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import argparse
import os
import pickle
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from lib.neuron_eval.ga.manager import GAManager

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Truncate a saved GA run to the first N generations')
    parser.add_argument('ga_file', type=str, help='Path to the ga pickle file')
    parser.add_argument('n_generations', type=int, help='Number of generations to keep')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    args = parser.parse_args()
    print("Loading GA state from:", args.ga_file)

    with open(args.ga_file, "rb") as f:
        ga_state = pickle.load(f)
    print("GA state loaded successfully.")
    print("Number of generations:", len(ga_state["history"]))
    if args.n_generations >= len(ga_state["history"]):
        print("No truncation needed, keeping all generations.")
        with open(args.output_file, "wb") as f:
            pickle.dump(ga_state, f)
        print("Saved to:", args.output_file)
    else:
        ga_state["history"] = ga_state["history"][:args.n_generations]
        print(f"Truncated to {args.n_generations} generations.")
        with open(args.output_file, "wb") as f:
            pickle.dump(ga_state, f)
        print("Saved to:", args.output_file)
        print("Truncated GA file successfully.")
        print("Number of generations after truncation:", len(ga_state["history"]))
        print("Done.")