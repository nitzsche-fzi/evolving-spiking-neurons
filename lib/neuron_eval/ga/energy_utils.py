"""Helpers for turning hardware measurements into per-neuron energy estimates."""

import numpy as np


def collect_hardware_metrics_from_generation(generation: dict):
    """Collects hardware energy metrics from a GA generation dict.

    Returns:
        Array of shape ``(2, pop_size)`` holding ``[idle_energy,
        spike_out_energy]``, or ``None`` if the generation has no hardware
        results.
    """
    hw_results = generation.get("hw_results")
    params = generation.get("params", [])
    pop_size = len(params)
    if not hw_results or pop_size == 0:
        return None
    idle = np.full(pop_size, np.nan, dtype=float)
    spike_out = np.full(pop_size, np.nan, dtype=float)
    for idx, hw in hw_results.items():
        energy = hw.get("energy", {}) if isinstance(hw, dict) else {}
        idle[idx] = float(energy.get("idle", np.nan))
        spike_out[idx] = float(energy.get("spike_out", np.nan))
    return np.stack([idle, spike_out], axis=0)


def compute_energy(spike_rates: np.ndarray, hw_metrics: np.ndarray, idle_scale: float = 1.0, report_missing: bool = False):
    """Computes per-task, per-neuron energy from spike rates and hardware metrics.

    Args:
        spike_rates: Array of shape ``(n_tasks, pop_size)`` or ``(pop_size,)``.
        hw_metrics: Array of shape ``(2, pop_size)`` holding ``[idle, spike_out]``.
        idle_scale: Scale factor applied to idle energy to model clock/power
            gating (default 1.0).
        report_missing: If True, prints the indices with missing/NaN inputs.

    Returns:
        Energy array of shape ``(n_tasks, pop_size)``, or ``None`` if either
        input is ``None``.
    """
    if hw_metrics is None or spike_rates is None:
        return None

    # Ensure 2D for uniform handling
    sr = spike_rates
    if sr.ndim == 1:
        sr = sr[None, :]

    idle = hw_metrics[0]
    spike_out = hw_metrics[1]

    # Optional reporting of missing inputs (NaNs / inf)
    if report_missing:
        # indices where any input is non-finite across tasks
        bad_idle = ~np.isfinite(idle)
        bad_spike_out = ~np.isfinite(spike_out)
        bad_sr_any = ~np.isfinite(sr)
        if bad_sr_any.ndim == 2:
            bad_sr_any = bad_sr_any.any(axis=0)
        bad_any = bad_idle | bad_spike_out | bad_sr_any
        if np.any(bad_any):
            bad_idx = np.where(bad_any)[0].tolist()
            print(f"[energy_utils] Warning: {len(bad_idx)} individuals have missing inputs (indices): {bad_idx}")

    return (1.0 - sr) * (idle_scale * idle)[None, :] + sr * spike_out[None, :]
