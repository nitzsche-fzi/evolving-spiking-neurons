"""Optimises a single neuron's training hyperparameters with Optuna.

Runs an Optuna study that repeatedly trains a given neuron on a task, searching
the hyperparameter space defined by an HPO config and reporting intermediate
accuracy for pruning.

Example:
    python3 scripts/hpo/optimize_hyperparams.py results/final/n2d1_01.pkl \\
        configs/task/shd.yaml configs/hpo/hpo_01.yaml 1000 1 cuda:1
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Ensure consistent GPU ordering
import yaml
import argparse
import copy
import torch

torch._dynamo.config.recompile_limit = 100000
torch._dynamo.config.accumulated_recompile_limit = 100000

import optuna
import pathlib
from optuna.samplers import GPSampler

from esn.neuron_classes import PMSN_CLR_Params
from lib.neuron_eval.tasks import get_task


class MockMessageQueue:
    """
    Reports intermediate accuracy to Optuna (for pruning),
    captures final accuracy, and tracks progress.
    """
    def __init__(self):
        self.test_accuracy = None
        self.n_steps = 1
        self.current_step = 0

    def put(self, message):
        msg_type = message.get("type")
        if msg_type == "training started":
            self.n_steps = message["n_training_steps"]
        elif msg_type == "training progress":
            self.current_step = message["step"]
        elif msg_type == "evaluation finished":
            self.test_accuracy = message["results"][0]

    def get_progress(self):
        return self.current_step, self.n_steps

    def get_accuracy(self):
        return self.test_accuracy

def objective(trial, task_config, neuron_params, device, hpo_config):
    #if we do multiple jobs in parallel, we need to set the recompile limits in each process
    torch._dynamo.config.recompile_limit = 100000 
    torch._dynamo.config.accumulated_recompile_limit = 100000  

    # 1. copy config & define hyper‐search space
    cfg    = copy.deepcopy(task_config)
    params = cfg["params"]
    params["layer1_size"] = params["layer_sizes"][1]
    params["layer2_size"] = params["layer_sizes"][2]
    params["time_scale_augmentation"] = 1.0 - params["random_time_scale"][0]
        
    for k, v in hpo_config.items():
        assert k in params, f"Hyperparameter '{k}' not found in base config."
        v_range = v["range"]
        v_type = v["type"]
        if v_type == "int":
            params[k] = trial.suggest_int(k, v_range[0], v_range[1])
        elif v_type == "float":
            v_log = v.get("log", False)
            params[k] = trial.suggest_float(k, v_range[0], v_range[1], log=v_log)
        elif v_type == "categorical":
            params[k] = trial.suggest_categorical(k, v_range)
        else:
            raise ValueError(f"Unsupported hyperparameter type '{v_type}'")

    input_dim  = params["layer_sizes"][0]
    output_dim = params["layer_sizes"][-1]
    l1 = params.pop("layer1_size")
    l2 = params.pop("layer2_size")
    params["layer_sizes"] = [input_dim, l1, l2, output_dim]
    tsa = params.pop("time_scale_augmentation")
    params["random_time_scale"] = [1.0 - tsa, 1.0 + tsa]

    # 2. prepare task & queue
    task = get_task(cfg)
    mq   = MockMessageQueue()
    seed = 412851
    spike_rate = neuron_params.spike_rate

    task.evaluate(seed, [neuron_params], device, mq, do_compile=False)

    acc = mq.get_accuracy()
    if acc is None:
        raise optuna.exceptions.TrialPruned()
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize SNN hyperparameters using Optuna."
    )
    parser.add_argument("neuron_file", type=str, help="Path to PMSN_CLR_Params file")
    parser.add_argument("task_file", type=str, help="Path to task yaml")
    parser.add_argument("hpo_file",  type=str, help="Path to hpo yaml")
    parser.add_argument("n_trials",    type=int, help="Number of trials")
    parser.add_argument("jobs",     type=int, help="Number of parallel jobs (default: 1)")
    parser.add_argument("device",     type=str, help="Device to use (e.g., 'cuda:0' or 'cpu'). GPUs in the same order as in `nvidia-smi`.")
    args = parser.parse_args()

    # load params & config
    neuron_params = PMSN_CLR_Params.load(args.neuron_file)
    with open(args.task_file, "r") as f:
        task_config = yaml.safe_load(f)
    with open(args.hpo_file, "r") as f:
        hpo_config = yaml.safe_load(f)
    
    #the study_name consists of the name of the neuron, the task, and the hpo config
    neuron_name = os.path.splitext(os.path.basename(args.neuron_file))[0]
    task_name   = os.path.splitext(os.path.basename(args.task_file))[0]
    hpo_name    = os.path.splitext(os.path.basename(args.hpo_file))[0]
    study_name  = f"{neuron_name}_{task_name}_{hpo_name}"

    print(f"The name of the study is: {study_name}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    #storage in results/optuna/{study_name}.db

    results_dir = pathlib.Path("results/optuna")
    results_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{results_dir}/{study_name}.db"
    if not os.path.exists(storage):
        print(f"Creating new Optuna study at {storage}")
    else:
        print(f"Loading existing Optuna study from {storage}")

    sampler = GPSampler(n_startup_trials=max(10, args.jobs))
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler
    )

    if not study.trials:
        init = task_config["params"]
        keys = list(hpo_config.keys())

        kwargs = {k: init[k] for k in keys if k in init}

        # layer_sizes must come from init, not kwargs
        layer_sizes = init["layer_sizes"]
        kwargs["layer1_size"] = layer_sizes[1]
        kwargs["layer2_size"] = layer_sizes[2]
        kwargs["time_scale_augmentation"] = 1.0 - init["random_time_scale"][0]

        study.enqueue_trial(kwargs)

    # run optimization
    study.optimize(
        lambda t: objective(t, task_config, neuron_params, device, hpo_config),
        n_trials=args.n_trials,
        n_jobs=args.jobs,
        show_progress_bar=True
    )

    # report results
    best = study.best_trial
    print("\nOptimization finished!")
    print(f"Best accuracy: {best.value:.6f}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
