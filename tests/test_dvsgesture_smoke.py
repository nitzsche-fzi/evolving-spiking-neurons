"""Smoke test for the DVSGesture task's train/evaluate pipeline."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

parent_dir = Path(__file__).resolve().parents[1]
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
helpers_path = Path(__file__).resolve().parent / "izhikevich_test_utils.py"
helpers_spec = importlib.util.spec_from_file_location(
    "tests.izhikevich_test_utils", helpers_path
)
if helpers_spec is None or helpers_spec.loader is None:
    raise ImportError("Unable to load izhikevich_test_utils.py")
helpers_module = importlib.util.module_from_spec(helpers_spec)
helpers_spec.loader.exec_module(helpers_module)
create_test_neuron_params = helpers_module.create_test_neuron_params

from lib.neuron_eval.combined_networks.dense_net import DenseNet
from lib.neuron_eval.combined_networks.loss import PopulationCELoss, accumulate_spike_loss
from lib.neuron_eval.combined_networks.clip_grad_norm import clip_population_grad_norm


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/task/dvsgesture.yaml"


def _load_dvsgesture_params():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return config["params"]


def _synthetic_batch(task_params, population_size, device):
    n_steps = task_params["n_steps"]
    batch_size = task_params["batch_size"]
    feature_size = task_params["layer_sizes"][0]
    n_classes = task_params["layer_sizes"][-1]

    std = float(task_params["inp_var"]) ** 0.5 if task_params["inp_var"] > 0 else 1.0
    inputs = torch.randn(
        n_steps,
        batch_size,
        population_size,
        feature_size,
        device=device,
    )
    inputs = inputs * std + float(task_params["inp_mean"])
    targets = torch.randint(0, n_classes, (batch_size,), device=device)
    return inputs, targets


def test_single_izhikevich_neuron_trains_with_dvsgesture_config():
    torch.manual_seed(0)
    device = torch.device("cpu")
    task_params = _load_dvsgesture_params()
    population_size = 1

    neuron_params = create_test_neuron_params(population_size)
    for params in neuron_params:
        params.surrogate_method = task_params["surrogate_method"]
        params.surrogate_alpha = task_params["surrogate_alpha"]

    net = DenseNet(
        seed=123,
        neuron_params=neuron_params,
        neuron_type="pmsn_clr",
        network_shape=task_params["layer_sizes"],
        batch_size=task_params["batch_size"],
        device=device,
        in_mean=task_params["inp_mean"],
        in_var=task_params["inp_var"],
    )
    net.train()

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=task_params["lr"],
        weight_decay=task_params["weight_decay"],
    )
    criterion = PopulationCELoss(
        task_params["batch_size"],
        population_size,
        task_params["layer_sizes"][-1],
    )

    first_weight = net.dense_layers[0].weight.detach().clone()
    losses = []
    n_updates = 3
    n_steps = task_params["n_steps"]

    for _ in range(n_updates):
        inputs, targets = _synthetic_batch(task_params, population_size, device)
        optimizer.zero_grad(set_to_none=True)
        states = net.init_state()
        outputs = None
        for step in range(n_steps):
            outputs, states = net(inputs[step], states)

        classification_loss, accuracies = criterion(outputs, targets)
        spike_loss = accumulate_spike_loss(states[-1])
        total_loss = classification_loss + task_params["spike_rate_reg"] * spike_loss
        assert torch.isfinite(total_loss), "Loss exploded during smoke training"
        total_loss.backward()
        clip_population_grad_norm(net, task_params["grad_clip_norm"])
        optimizer.step()

        assert torch.isfinite(states[-1]).all(), "Spike counts became NaN/Inf"
        assert accuracies.shape == (population_size,)
        losses.append(total_loss.detach().item())

    final_weight = net.dense_layers[0].weight.detach()
    assert not torch.allclose(first_weight, final_weight)
    assert len(losses) == n_updates
    assert all(math.isfinite(value) for value in losses)
