"""Evaluates a set of neurons under a given config and saves the results.

Each neuron is evaluated ``n_repeats`` times to average out training noise in
its accuracy, while hardware energy (which is deterministic per neuron) is
evaluated only once and then broadcast across the repeats. Mean accuracy,
energy, spike rate and the combined fitness are written to
``evaluation_data.pkl`` in the output directory.

Example:
    python3 scripts/evaluation/evaluate_neurons.py results/best_neurons/20x_n2d1.pkl \\
        configs/ga/n2d1.yaml 10 --output_path n2d1
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import pickle
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import yaml
import numpy as np
from pathlib import Path
import datetime

import torch
torch._dynamo.config.recompile_limit = 100000
torch._dynamo.config.accumulated_recompile_limit = 100000

from lib.neuron_eval.hardware.hardware_evaluator import HardwareEvaluator
from lib.utils import get_root_dir


class MockHWQueue:
    """Minimal in-process stand-in for a multiprocessing queue.

    Lets :class:`HardwareEvaluator` be driven synchronously in this script
    without spawning a separate manager process.
    """

    def __init__(self):
        self.data = []

    def get(self):
        return self.data.pop(0) if self.data else None

    def put(self, item):
        self.data.append(item)

    def close(self):
        pass

if __name__ == "__main__":
    from esn.neuron_classes import PMSN_CLR
    from lib.neuron_eval.ga.manager import GAManager

    parser = argparse.ArgumentParser(description='Evaluate a set of neurons with a given config.')
    parser.add_argument('neuron_file', type=str, help='Path to the neuron pickle file')
    parser.add_argument('config_file', type=str, help='Path to the config yaml file')
    parser.add_argument('n_repeats', type=int, help='Number of repeats for evaluation')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the evaluation results (optional)')
    args = parser.parse_args()

    with open(args.neuron_file, "rb") as f:
        neuron_data = pickle.load(f)

    n_neurons = neuron_data["n_neurons"]
    neuron_params = neuron_data["params"]

    with open(args.config_file, "rb") as f:
        config = yaml.safe_load(f)

    #before we init the manager, we need to repeat the neuron params n_repeats times
    new_neuron_data = {"n_neurons": n_neurons * args.n_repeats, "params": []}
    for _ in range(args.n_repeats):
        new_neuron_data["params"].extend(neuron_params)

    new_neuron_data["results"] = {} #needed by ga manager
    
    print(f"population size for evaluation: {new_neuron_data['n_neurons']}")
    
    neuron_file_name = Path(args.neuron_file).stem
    if args.output_path:
        output_dir = get_root_dir() / "results" / "evaluate_neurons" / args.output_path
    else:
        output_dir = get_root_dir() / "results" / "evaluate_neurons" / neuron_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    acc_manager_output_file = output_dir / "acc_data.pkl"
    acc_manager = GAManager(config, new_neuron_data, [], acc_manager_output_file)
    acc_manager.show_gpu_dialog()

    # Hardware is evaluated separate from GAManager, since we evaluate each neuron n_repeats times 
    # but the hardware result would be the same everytime. Hence hardware is evaluated once per neuron
    # and only accuracy evaluation is repeated using the GAManager.
    hw_dir = output_dir / "hardware_eval"
    if config.get("enable_hardware_eval", False):
        hw_manager = HardwareEvaluator(
            run_dir_name="",
            output_main_dir=hw_dir,
            num_devices=1,
            generation=0,
        )
        hw_queue = MockHWQueue()
        hw_manager.evaluate_population(
            neuron_data=neuron_data,
            output_queue=hw_queue,
            max_parallel=config.get("max_hw_processes", 10)
        )
    
        hw_results_single = hw_queue.get()
        hw_results_repeated = {}

        for r in range(args.n_repeats):
            for idx in range(n_neurons):
                new_idx = r * n_neurons + idx
                hw_results_repeated[new_idx] = hw_results_single[idx]

        acc_manager.neuron_data["hw_results"] = hw_results_repeated

    acc_manager.hw_eval_enabled = False  # Disable hardware evaluation in accuracy manager
    acc_manager.include_hw_fitness = False  # Disable hardware fitness inclusion
    acc_manager.eval_population() # this populates manager.neuron_data with the results of the evaluation

    if config.get("enable_hardware_eval", False):
        # Re-enable hardware evaluation for fitness computation
        acc_manager.hw_eval_enabled = True
        acc_manager.include_hw_fitness = True

    all_genomes, all_infos, all_accs, all_energy = acc_manager._collect_all_individuals()
    # Next step: average energies, accuracies (and spike rates) over runs for the same neuron.
    num_tasks = all_accs.shape[0]
    num_individuals = len(neuron_data["params"])
    all_accs = all_accs.reshape(num_tasks, args.n_repeats, num_individuals)
    all_accs = np.transpose(all_accs, (0, 2, 1))  # -> (num_tasks, num_individuals, n_repeats)

    if all_energy is None:
        all_energy = np.zeros_like(all_accs)
    
    all_energy = all_energy.reshape(num_tasks, args.n_repeats, num_individuals)
    all_energy = np.transpose(all_energy, (0, 2, 1))

    # spike rates and eval_timesteps per task: collect from latest generation
    latest_gen = acc_manager.history[-1]  # single generation for this evaluation
    spike_rates_flat = acc_manager._collect_spike_rates(latest_gen)  # (num_tasks, n_repeats * num_individuals)
    all_spike_rates = spike_rates_flat.reshape(num_tasks, args.n_repeats, num_individuals)
    all_spike_rates = np.transpose(all_spike_rates, (0, 2, 1))  # -> (num_tasks, num_individuals, n_repeats)

    # eval_timesteps typically same for all individuals per task; take first per task for one value per task
    eval_timesteps_flat = np.array(acc_manager._collect_eval_total_timesteps(latest_gen))  # (num_tasks, pop_size)
    all_timesteps = np.array([eval_timesteps_flat[t, 0] for t in range(num_tasks)], dtype=float)

    mean_acc = np.mean(all_accs, axis=2)  # -> (num_tasks, num_individuals)
    mean_energy = np.mean(all_energy, axis=2)  # -> (num_tasks, num_individuals)
    mean_spike_rate = np.mean(all_spike_rates, axis=2)  # -> (num_tasks, num_individuals)

    fitness = acc_manager._compute_fitness(mean_acc, mean_energy)
    
    # Collect task configs referenced in the GA config
    task_configs = []
    for task_path in config.get("tasks", []):
        with open(task_path, "rb") as f:
            task_cfg = yaml.safe_load(f)
            task_configs.append(task_cfg)

    # Collect metadata
    metadata = {
        "n_neurons": n_neurons,
        "n_repeats": args.n_repeats,
        "num_tasks": mean_acc.shape[0],
        "config": config,
        "task_configs": task_configs,  # list aligned with config['tasks']
        "timestamp": datetime.datetime.now().isoformat(),
        "neuron_file": Path(args.neuron_file).resolve(),
        "config_file": Path(args.config_file).resolve(),
        "output_path": output_dir.resolve(),
    }

    # Consolidate everything to a single dict
    results_dict = {
        "metadata": metadata,
        "hw_results": acc_manager.neuron_data.get("hw_results", None),
        "all_accs": all_accs,
        "all_energy": all_energy,
        "all_spike_rates": all_spike_rates,
        "all_timesteps": all_timesteps,  # (num_tasks,) total timesteps per task, used for network energy
        "mean_acc": mean_acc,
        "mean_energy": mean_energy,
        "mean_spike_rate": mean_spike_rate,
        "fitness": fitness,
    }

    # Serialize to pickle
    output_file = output_dir / "evaluation_data.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"Saved full evaluation data to {output_file}")
