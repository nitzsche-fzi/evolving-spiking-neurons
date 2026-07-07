"""Visual check of the per-task input mean/std used for dense-layer initialization."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from esn.augmentations import AudioPad, FrameTransform
import tonic

# Ensure repo root on path
parent_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(parent_dir))

from lib.neuron_eval.tasks import get_task


@pytest.mark.visual
def test_task_meanstd_computation():
    """
    Long-running visual-style test:
    Loads SHD and DVS Gesture configs, computes activation-filtered mean/var,
    and writes the results to tmp/visual_tests/task_stats/.

    Extended version: the output text file explains each reduction step,
    includes dataset statistics, and describes the meaning of each reported value.
    """

    ROOT = parent_dir
    OUTDIR = ROOT / "tmp" / "visual_tests" / "task_stats"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    outfile = OUTDIR / "mean_std_results.txt"

    config_paths = {
        "shd": ROOT / "configs" / "task" / "shd.yaml",
        "dvsgesture": ROOT / "configs" / "task" / "dvsgesture.yaml",
    }

    activation_fraction = 0.2

    lines = []
    lines.append("==============================================================")
    lines.append("     Spiking Dataset Mean/Var Report (Activation-Filtered)")
    lines.append("==============================================================")
    lines.append(f"Activation fraction = {activation_fraction} (fraction of per-timestep max activity)")
    lines.append("")
    lines.append("This file explains how each statistic was computed.\n")
    lines.append("Definitions:")
    lines.append("  • A 'step' means a single time slice of shape (features,).")
    lines.append("  • For each sequence: (B,T,F) → reshaped to (B*T, F) = all time-steps flat.")
    lines.append("  • 'Activity' per step is the sum over all features: activity = sum(step_features).")
    lines.append("  • max_activation: maximum activity over all (B*T) steps in the dataset.")
    lines.append("  • threshold: activation_fraction * max_activation.")
    lines.append("  • A step is 'selected' if its activity >= threshold.")
    lines.append("  • mean/var: computed across ALL selected steps, over ALL features (flattened).")
    lines.append("")

    def compute_stats(task_name, config_path):
        with open(config_path, "r") as f:
            task_config = yaml.safe_load(f)

        task_config["n_epochs"] = 1  # prevent training loops
        task = get_task(task_config)

        # ----------------------------------------
        # Dataset + dataloader configuration
        # ----------------------------------------
        if task_name == "shd":
            dataloader = DataLoader(
                task.training_dataset,
                collate_fn=AudioPad(batch_first=False, noise=task.noise),
                pin_memory=False,
                num_workers=4,
                shuffle=False,
            )
        else:
            dataloader = DataLoader(
                task.training_dataset,
                collate_fn=tonic.collation.PadTensors(batch_first=False),
                pin_memory=False,
                num_workers=4,
                shuffle=False,
            )

        dataset_size = len(task.training_dataset)

        # Gather shapes info
        example_shapes = []
        for i, batch in enumerate(dataloader):
            x, y = batch
            example_shapes.append(tuple(x.shape))
            if i >= 2:
                break

        all_activities = []
        all_steps = []

        # ----------------------------------------
        # Build flattened step matrix
        # ----------------------------------------
        for batch in dataloader:
            inputs, _ = batch  # (B, T, F or channels flattened)
            inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)

            x_np = inputs.cpu().numpy()
            steps = x_np.reshape(-1, x_np.shape[-1])  # (B*T, F)
            activities = steps.sum(axis=1)  # (B*T,)

            all_activities.append(activities)
            all_steps.append(steps)

        all_activities = np.concatenate(all_activities, axis=0)
        all_steps = np.concatenate(all_steps, axis=0)

        num_steps, feature_dim = all_steps.shape

        # ----------------------------------------
        # Thresholding
        # ----------------------------------------
        max_act = float(np.max(all_activities))
        threshold = activation_fraction * max_act
        mask = all_activities >= threshold
        selected = all_steps[mask]

        inp_mean = float(selected.mean())
        inp_var = float(selected.var(ddof=0))

        return {
            "dataset_size": dataset_size,
            "example_shapes": example_shapes,
            "num_steps": num_steps,
            "feature_dim": feature_dim,
            "threshold": threshold,
            "n_selected": selected.shape[0],
            "mean": inp_mean,
            "var": inp_var,
            "max_act": max_act,
        }

    # ---------------------------------------------------------------
    # Run tasks
    # ---------------------------------------------------------------
    for task_name, config_path in config_paths.items():
        stats = compute_stats(task_name, config_path)
        lines.append("--------------------------------------------------------------")
        lines.append(f"[{task_name.upper()}]")
        lines.append("--------------------------------------------------------------")
        lines.append(f"Dataset samples           : {stats['dataset_size']}")
        lines.append("Example batch shapes      : " + ", ".join(str(s) for s in stats["example_shapes"]))
        lines.append(f"Flattened total steps     : {stats['num_steps']} (each of dim {stats['feature_dim']})")
        lines.append("")
        lines.append("Reduction process:")
        lines.append("  • Each step x ∈ ℝ^F produces activity = sum(x).")
        lines.append("  • Activities across all steps give a global maximum.")
        lines.append(f"  • max_activation         : {stats['max_act']:.6f}")
        lines.append(f"  • threshold (act_frac)   : {stats['threshold']:.6f}")
        lines.append(f"  • selected steps count   : {stats['n_selected']}")
        lines.append("")
        lines.append("Final statistics over SELECTED steps only:")
        lines.append("  • mean(value over all features of all selected steps)")
        lines.append("  • var(value over all features of all selected steps)")
        lines.append(f"    mean = {stats['mean']:.6f}")
        lines.append(f"    var  = {stats['var']:.6f}")
        lines.append("")

    # ---------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------
    outfile.write_text("\n".join(lines))
    print(f"[task mean/std] Results written to: {outfile}")
