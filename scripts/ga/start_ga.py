"""Starts a new GA run from a config and initial population.

Interactively selects GPUs, then evolves the population for the requested number
of generations, saving the full GA state to the output pickle after each one.

Example:
    python3 scripts/ga/start_ga.py configs/ga/n2d3_01.yaml results/ga/n2d3_01.pkl 20
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

from lib.neuron_eval.ga.manager import GAManager 

from esn.neuron_classes import PMSN_CLR

torch._dynamo.config.recompile_limit = 100000
torch._dynamo.config.accumulated_recompile_limit = 100000

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Start the genetic algorithm')
    parser.add_argument('ga_config', type=str, help='Path to the ga config yaml file')
    parser.add_argument('output_file', type=str, help='Path to the output file')
    parser.add_argument('n_generations', type=int, nargs='?', default=100, help='Number of generations to run')
    args = parser.parse_args()

    print("Loading GA config from:", args.ga_config)
    with open(args.ga_config, "rb") as f:
        ga_config = yaml.safe_load(f)

    # Mute torch JIT
    import logging
    logging.getLogger("torch._inductor.select_algorithm").setLevel(logging.CRITICAL)
    logging.getLogger("torch._inductor").setLevel(logging.CRITICAL)

    manager = GAManager.init_new_run(ga_config, args.output_file)
    manager.show_gpu_dialog()

    for i in range(args.n_generations):
        manager.eval_population()
        manager.sample_population()
        
