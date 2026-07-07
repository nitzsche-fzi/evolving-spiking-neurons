"""Smoke test that DenseNet runs a forward pass with the real neuron components."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

helpers_path = Path(__file__).resolve().parent / "izhikevich_test_utils.py"
helpers_spec = importlib.util.spec_from_file_location(
    "tests.izhikevich_test_utils", helpers_path
)
if helpers_spec is None or helpers_spec.loader is None:
    raise ImportError("Unable to load izhikevich_test_utils.py")
helpers_module = importlib.util.module_from_spec(helpers_spec)
helpers_spec.loader.exec_module(helpers_module)

from lib.neuron_eval.combined_networks import dense_net
from lib.neuron_eval.combined_networks.clip_grad_norm import clip_population_grad_norm
IZHI_PARAMS = helpers_module.IZHI_PARAMS
_create_test_neuron_params = helpers_module.create_test_neuron_params


def test_dense_net_forward_runs_with_real_components():
    device = torch.device("cpu")
    batch_size = 2
    population_size = 3
    network_shape = [4, 5, 3]
    neuron_params = _create_test_neuron_params(population_size)

    net = dense_net.DenseNet(
        seed=123,
        neuron_params=neuron_params,
        neuron_type="pmsn_clr",
        network_shape=network_shape,
        batch_size=batch_size,
        device=device,
        in_mean=1.0,
        in_var=0.5,
    )

    states = net.init_state()
    assert len(states) == len(net.spiking_neurons) + 1
    assert states[-1].shape == (population_size,)

    inputs = torch.randn(5, batch_size, population_size, network_shape[0], device=device)
    spike_count = states[-1].clone()
    last_output = None
    for t in range(inputs.shape[0]):
        last_output, states = net(inputs[t], states)

    assert last_output is not None
    assert last_output.shape == (batch_size, population_size, network_shape[-1])
    assert states[-1].shape == (population_size,)
    assert torch.all(states[-1] >= spike_count)
    assert net.get_spiking_neuron_count() == sum(network_shape[1:-1])


@pytest.mark.longrun
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_dense_net_gradients_match_with_compile():
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile is unavailable")

    device = torch.device("cpu")
    torch.manual_seed(42)
    batch_size = 2
    population_size = 3
    network_shape = [4, 5, 3]
    time_steps = 4

    neuron_params = _create_test_neuron_params(population_size)
    inputs = torch.randn(
        time_steps, batch_size, population_size, network_shape[0], device=device
    )

    def build_net():
        return dense_net.DenseNet(
            seed=123,
            neuron_params=neuron_params,
            neuron_type="pmsn_clr",
            network_shape=network_shape,
            batch_size=batch_size,
            device=device,
            in_mean=1.0,
            in_var=0.5,
        )

    net_eager = build_net()
    net_compiled = build_net()
    net_compiled.load_state_dict(net_eager.state_dict())
    compiled_net = compile_fn(net_compiled, mode="max-autotune", fullgraph=True, dynamic=False)

    def run_sequence(callable_net, owning_net):
        owning_net.zero_grad(set_to_none=True)
        states = owning_net.init_state()
        loss = torch.zeros((), device=device)
        for step in range(time_steps):
            x_t = inputs[step]
            output, states = callable_net(x_t, states)
            loss = loss + output.square().mean()
        loss.backward()
        grads = []
        for name, param in owning_net.named_parameters():
            if not param.requires_grad:
                continue
            grads.append((name, None if param.grad is None else param.grad.detach().clone()))
        return grads

    eager_grads = run_sequence(net_eager, net_eager)
    compiled_grads = run_sequence(compiled_net, net_compiled)

    assert [name for name, _ in eager_grads] == [name for name, _ in compiled_grads]

    for (name, eager_grad), (_, compiled_grad) in zip(eager_grads, compiled_grads):
        if eager_grad is None or compiled_grad is None:
            assert eager_grad is compiled_grad is None
            continue
        torch.testing.assert_close(
            eager_grad, compiled_grad, rtol=1e-5, atol=1e-6, msg=f"Gradient mismatch for {name}"
        )

@pytest.mark.longrun
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_population_grad_norm_clip_matches_with_compile():
    """
    Non-visual: ensure that applying clip_population_grad_norm on compiled vs uncompiled paths
    yields identical gradients.
    """
    from lib.neuron_eval.combined_networks.clip_grad_norm import clip_population_grad_norm

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        pytest.skip("torch.compile unavailable")

    device = torch.device("cpu")
    batch_size = 8
    population_size = 4
    network_shape = [16, 32, 8]
    time_steps = 10
    max_norm = 1.0

    torch.manual_seed(123)
    neuron_params = _create_test_neuron_params(population_size)
    inputs = torch.randn(
        time_steps, batch_size, population_size, network_shape[0], device=device
    )

    def build():
        return dense_net.DenseNet(
            seed=777,
            neuron_params=neuron_params,
            neuron_type="pmsn_clr",
            network_shape=network_shape,
            batch_size=batch_size,
            device=device,
            in_mean=1.0,
            in_var=0.5,
        )

    # ----------------------------
    # Build eager + compiled copies
    # ----------------------------
    eager = build()
    compiled_owner = build()
    compiled_owner.load_state_dict(eager.state_dict())
    compiled = compile_fn(compiled_owner, mode="max-autotune", fullgraph=True)

    # ----------------------------
    # Helper that runs, backprops, clips
    # ----------------------------
    def run(net_callable, owner):
        owner.zero_grad(set_to_none=True)
        states = owner.init_state()
        loss = torch.zeros((), device=device)
        for t in range(time_steps):
            out, states = net_callable(inputs[t], states)
            loss = loss + out.square().mean()

        loss.backward()
        clip_population_grad_norm(owner, max_norm=max_norm)

        # Extract all grads into dict
        result = {}
        for name, p in owner.named_parameters():
            if p.grad is not None:
                result[name] = p.grad.detach().clone()
        return result

    grads_eager = run(eager, eager)
    grads_comp  = run(compiled, compiled_owner)

    assert grads_eager.keys() == grads_comp.keys()

    # The clipping must produce identical per-parameter gradients
    for name in grads_eager.keys():
        torch.testing.assert_close(
            grads_eager[name],
            grads_comp[name],
            rtol=1e-6,
            atol=1e-6,
            msg=f"Clipped gradient mismatch for {name}",
        )


@pytest.mark.visual
def test_dense_net_gradients_visual():
    ROOT = Path(__file__).resolve().parents[1]
    BASE = ROOT / "tmp" / "visual_tests" / "dense_net"
    BASE.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    torch.manual_seed(99)
    batch_size = 8
    population_size = 4
    network_shape = [16, 32, 8]
    time_steps = 10

    neuron_params = _create_test_neuron_params(population_size)
    spike_rates = [float(getattr(p, "spike_rate", float("nan")))
                   for p in neuron_params]

    def build_model():
        return dense_net.DenseNet(
            seed=999,
            neuron_params=neuron_params,
            neuron_type="pmsn_clr",
            network_shape=network_shape,
            batch_size=batch_size,
            device=device,
            in_mean=1.0,
            in_var=0.5,
        )

    # ----------------------------------------------------------
    # Build eager + compiled twins
    # ----------------------------------------------------------
    net_eager = build_model()
    net_compiled_owner = build_model()
    net_compiled_owner.load_state_dict(net_eager.state_dict())

    compiled_net = torch.compile(net_compiled_owner)

    inputs = torch.randn(
        time_steps, batch_size, population_size, network_shape[0], device=device
    )

    # ----------------------------------------------------------
    # Helper to run the network while collecting gradients
    # BUT: always read parameters from the “owner” model.
    # ----------------------------------------------------------
    def run_and_collect(callable_net, owning_net):
        owning_net.zero_grad(set_to_none=True)
        states = owning_net.init_state()
        loss = torch.zeros((), device=device)

        last_output = None
        for step in range(time_steps):
            last_output, states = callable_net(inputs[step], states)
            loss = loss + last_output.square().mean()

        loss.backward()

        grads = {}
        for name, param in owning_net.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grads[name] = (
                param.detach().cpu(),
                param.grad.detach().cpu(),
            )

        return (
            loss.detach().cpu(),
            last_output.detach().cpu(),
            states[-1].detach().cpu(),
            grads,
        )

    eager_loss, eager_output, eager_spikes, eager_grads = run_and_collect(
        net_eager, net_eager
    )
    compiled_loss, compiled_output, compiled_spikes, compiled_grads = run_and_collect(
        compiled_net, net_compiled_owner
    )

    # ----------------------------------------------------------
    # Build diagnostic text file
    # ----------------------------------------------------------
    diag_lines = [
        "DenseNet Gradient Visual Report",
        f"time_steps={time_steps}",
        f"input_mean={inputs.mean().item():.6f}, input_std={inputs.std().item():.6f}",
        "spike_rates=" + ", ".join(f"{rate:.4f}" for rate in spike_rates),
        f"loss_eager={eager_loss.item():.6f}, loss_compiled={compiled_loss.item():.6f}",
        (
            "output_stats_eager="
            f"mean={eager_output.mean().item():.6f}, "
            f"std={eager_output.std(unbiased=False).item():.6f}, "
            f"min={eager_output.min().item():.6f}, "
            f"max={eager_output.max().item():.6f}"
        ),
        (
            "output_stats_compiled="
            f"mean={compiled_output.mean().item():.6f}, "
            f"std={compiled_output.std(unbiased=False).item():.6f}, "
            f"min={compiled_output.min().item():.6f}, "
            f"max={compiled_output.max().item():.6f}"
        ),
        (
            "spike_count_stats_eager="
            f"mean={eager_spikes.mean().item():.6f}, "
            f"std={eager_spikes.std(unbiased=False).item():.6f}, "
            f"min={eager_spikes.min().item():.6f}, "
            f"max={eager_spikes.max().item():.6f}"
        ),
        (
            "spike_count_stats_compiled="
            f"mean={compiled_spikes.mean().item():.6f}, "
            f"std={compiled_spikes.std(unbiased=False).item():.6f}, "
            f"min={compiled_spikes.min().item():.6f}, "
            f"max={compiled_spikes.max().item():.6f}"
        ),
        "",
        "Parameter summaries:",
    ]

    lines = diag_lines

    # Same names now, so no "missing" spam
    param_names = sorted(set(eager_grads.keys()) | set(compiled_grads.keys()))
    for name in param_names:
        eager_weight, eager_grad = eager_grads[name]
        _, comp_grad = compiled_grads[name]

        weight_norm = eager_weight.norm().item()
        grad_norm = eager_grad.norm().item()
        comp_grad_norm = comp_grad.norm().item()
        grad_diff_norm = (eager_grad - comp_grad).norm().item()

        sample_n = min(10, eager_grad.numel())
        sample_vals = ", ".join(
            f"{v:.5f}" for v in eager_grad.flatten()[:sample_n].tolist()
        )
        if eager_grad.numel() > sample_n:
            sample_vals += ", ..."

        lines.append(
            f"[{name}] shape={tuple(eager_weight.shape)} "
            f"weight_norm={weight_norm:.6f} "
            f"grad_norm={grad_norm:.6f} "
            f"compiled_grad_norm={comp_grad_norm:.6f} "
            f"grad_diff_norm={grad_diff_norm:.6f}"
        )
        lines.append(f" sample (eager grad): {sample_vals}")

        # ----------------------------------------------------------
    # Apply per-individual gradient clipping and log results
    # ----------------------------------------------------------
    max_norm = 1.0

    # Make deep copies of grads to keep original diagnostics intact
    eager_clip_net = build_model()
    eager_clip_net.load_state_dict(net_eager.state_dict())
    compiled_clip_net = build_model()
    compiled_clip_net.load_state_dict(net_compiled_owner.state_dict())

    # We need to re-run forward/backward to populate .grad on the new nets
    def rerun_and_clip(owner, call):
        owner.zero_grad(set_to_none=True)
        states = owner.init_state()
        loss = torch.zeros((), device=device)
        for step in range(time_steps):
            out, states = call(inputs[step], states)
            loss = loss + out.square().mean()
        loss.backward()
        clip_population_grad_norm(owner, max_norm=max_norm)
        # Collect clipped grads (per layer, per individual)
        clipped = {}
        for name, p in owner.named_parameters():
            if p.grad is not None:
                clipped[name] = p.grad.detach().cpu()
        return clipped

    eager_clipped = rerun_and_clip(eager_clip_net, eager_clip_net)
    compiled_clipped = rerun_and_clip(compiled_clip_net, compiled_clip_net)

    lines.append("")
    lines.append("Population-clipped gradient statistics:")
    lines.append(f"max_norm={max_norm}")
    lines.append("")

    for name in sorted(eager_clipped.keys()):
        g = eager_clipped[name]
        # If shape is e.g. [population_size, ...], we compute population-wide norms
        if g.ndim >= 2:
            # flatten per individual
            g_flat = g.view(g.shape[0], -1)
            norms = g_flat.norm(dim=1)
            sample_norms = ", ".join(f"{v:.5f}" for v in norms.tolist())
            lines.append(f"[{name}] per-individual norms (eager): {sample_norms}")
        else:
            lines.append(f"[{name}] scalar or bias, norm={g.norm().item():.6f}")

    lines.append("")
    lines.append("Compiled clipped gradients (for completeness):")
    for name in sorted(compiled_clipped.keys()):
        g = compiled_clipped[name]
        if g.ndim >= 2:
            g_flat = g.view(g.shape[0], -1)
            norms = g_flat.norm(dim=1)
            sample_norms = ", ".join(f"{v:.5f}" for v in norms.tolist())
            lines.append(f"[{name}] per-individual norms (compiled): {sample_norms}")
        else:
            lines.append(f"[{name}] scalar or bias, norm={g.norm().item():.6f}")


    output_path = BASE / "dense_net_gradients.txt"
    output_path.write_text("\n".join(lines))
    print(f"Gradient snapshot written to {output_path}")
