"""Smoke test for the SHD task's train/evaluate pipeline."""

import numpy as np
import torch
import pytest
import os
import sys
import yaml

# ------------------------------------------------------------
# Imports / path setup
# ------------------------------------------------------------
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from esn.neurons.clr import Izhikevich
from lib.neuron_eval.combined_networks.pmsn_clr import PMSN_CLR as PMSN_CLR_Lib
from lib.neuron_eval.tasks import get_task

task_file = os.path.join(parent_dir, "configs", "task", "shd.yaml")

def test_shd_task():
    #we need to test that the shd task is working correctly
    torch.manual_seed(0)
    np.random.seed(0)
    device = "cpu"
    task_config = None
    with open(task_file, "r") as f:
        task_config = yaml.safe_load(f)
    task_config["n_epochs"] = 2 #only 2 epochs. why 2? because we want to see if it handles epoch boundaries correctly
    task_config["batch_size"] = 32
    task = get_task(task_config)
    train_dataloader = task.get_train_dataloader(num_workers=0) #keep everything on same process for testing
    test_dataloader = task.get_test_dataloader(num_workers=0)

    
    print("polling train dataloader:")
    for data, targets in train_dataloader:
        print(".", end="", flush=True)
    
    print("\npolling test dataloader:")
    for data, targets in test_dataloader:
        print(".", end="", flush=True)
    print("\nSHD task dataloaders work correctly.")

if __name__ == "__main__":
    test_shd_task()
