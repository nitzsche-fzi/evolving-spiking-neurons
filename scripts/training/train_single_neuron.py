#!/usr/bin/env python3
"""Trains a single neuron on one task and reports its final accuracy.

Loads a neuron's parameters, trains it on the given task with fixed
hyperparameters (no search), and records the final result.

Example:
    python3 scripts/training/train_single_neuron.py results/final/n2d1_01.pkl \\
        configs/task/shd.yaml cuda:0
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import yaml
import copy
import torch

torch._dynamo.config.recompile_limit = 100000
torch._dynamo.config.accumulated_recompile_limit = 100000

from lib.neuron_eval.tasks import get_task

from esn.neuron_classes import PMSN_CLR_Params

class MockMessageQueue:
    """
    Reports intermediate accuracy (ignored here) and records final result.
    """
    def __init__(self):
        self.test_accuracy = None
        self.n_steps = 1
        self.current_step = 0
        self.ma_accs = []

    def put(self, message):
        msg_type = message.get("type")
        if msg_type == "training started":
            self.n_steps = message["n_training_steps"]
        elif msg_type == "training progress":
            self.current_step = message["step"]
            self.accuracies = message.get("accuracies", [])
            if len(self.ma_accs) < len(self.accuracies):
                self.ma_accs = self.accuracies
            else:
                for i in range(len(self.accuracies)):
                    if self.accuracies[i] is not None:
                        self.ma_accs[i] = self.accuracies[i] * 0.1 + self.ma_accs[i] * 0.9
            print(f"\rTraining progress: {self.current_step}/{self.n_steps}  Accuracies: {self.ma_accs}", end="", flush=True)
        elif msg_type == "evaluation finished":
            self.test_accuracy = message["results"][0]

    def get_accuracy(self):
        return self.test_accuracy


def run_single_training(task_config, neuron_params, device):
    # Make sure these limits are set per-process
    torch._dynamo.config.recompile_limit = 100000
    torch._dynamo.config.accumulated_recompile_limit = 100000

    # Copy config
    cfg = copy.deepcopy(task_config)

    # Prepare task
    task = get_task(cfg)

    # Prepare message queue
    mq = MockMessageQueue()
    seed = 412851

    # Run training+evaluation
    task.evaluate(seed, [neuron_params], device, mq, do_compile=True)

    return mq.get_accuracy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one SNN instance with a given neuron and task configuration."
    )
    parser.add_argument("neuron_file", type=str, help="Path to PMSN_CLR_Params file")
    parser.add_argument("task_file", type=str, help="Path to task yaml")
    parser.add_argument("device", type=str, help="Device (e.g. cuda:0 or cpu)")
    args = parser.parse_args()

    # Load neuron params
    neuron_params = PMSN_CLR_Params.load(args.neuron_file)

    # Load task configuration
    with open(args.task_file, "r") as f:
        task_config = yaml.safe_load(f)

    # Choose device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Run training
    acc = run_single_training(task_config, neuron_params, device)

    # Report
    print("\nTraining run finished!")
    if acc is None:
        print("No accuracy reported (something failed inside task.evaluate).")
    else:
        print(f"Final accuracy: {acc:.6f}")
