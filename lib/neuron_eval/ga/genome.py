"""Conversion between neuron parameter objects and flat genome arrays.

The genetic algorithm operates on 1D float genomes, while the rest of the code
works with typed PMSN-CLR parameter objects.
``flatten_genome`` encodes a parameter object into a numpy array plus a
``reconstruct_info`` dict of everything not stored in the genome (shapes,
non-evolved metadata), and ``reconstruct_genome`` inverts that mapping.
"""

import numpy as np
import torch
from esn.neuron_classes import PMSN_CLR_Params


def _flatten_pmsn_clr(pmsn_clr_params):
    """Flattens PMSN-CLR params into ``(genome, reconstruct_info)``."""
    ret = []
    indices = [0]
    shapes = []
    ret += pmsn_clr_params.polynomial_coeffs.flatten().tolist()
    indices.append(len(ret))
    shapes.append(pmsn_clr_params.polynomial_coeffs.shape)

    threshold = pmsn_clr_params.threshold
    if isinstance(threshold, torch.Tensor):
        threshold = threshold.item()
    threshold = float(threshold)
    ret.append(threshold)
    indices.append(len(ret))

    ret += pmsn_clr_params.reset_coeffs.flatten().tolist()
    indices.append(len(ret))
    shapes.append(pmsn_clr_params.reset_coeffs.shape)

    reconstruct_info = {
        "params_type": "pmsn_clr",
        "indices": indices,
        "shapes": shapes,
        "n_states": pmsn_clr_params.n_states,
        "degree": pmsn_clr_params.degree,
        "state_delta": pmsn_clr_params.state_delta,
        "surrogate_method": pmsn_clr_params.surrogate_method,
        "surrogate_alpha": pmsn_clr_params.surrogate_alpha,
        "spike_rate": pmsn_clr_params.spike_rate,
        "resting_state": pmsn_clr_params.resting_state,
    }

    return np.array(ret, dtype=np.float32), reconstruct_info

def flatten_genome(params, neuron_type: str = None):
    """Encodes a neuron parameter object into a flat genome array.

    Args:
        params: A ``PMSN_CLR_Params`` object.
        neuron_type: Optional explicit ``"pmsn_clr"`` override; inferred from
            ``params`` when omitted.

    Returns:
        Tuple ``(genome, reconstruct_info)`` suitable for
        :func:`reconstruct_genome`.
    """
    neuron_type = neuron_type or getattr(params, "params_type", None)
    if neuron_type is None:
        if hasattr(params, "polynomial_coeffs"):
            neuron_type = "pmsn_clr"
    if neuron_type == "pmsn_clr":
        return _flatten_pmsn_clr(params)
    raise ValueError(f"Unsupported neuron type for genome flattening: {neuron_type}")

def reconstruct_genome(genome, reconstruct_info, neuron_type: str = None):
    """Rebuilds a neuron parameter object from a genome and its metadata.

    Inverse of :func:`flatten_genome`. If ``reconstruct_info`` is flagged as
    ``is_initial``, the evolved-only fields (spike rate, resting state) are left
    at their defaults rather than restored.
    """
    params_type = reconstruct_info.get("params_type", neuron_type or "pmsn_clr")
    if params_type == "pmsn_clr":
        polynomial_coeffs = torch.tensor(
            genome[reconstruct_info["indices"][0]:reconstruct_info["indices"][1]],
            dtype=torch.float32
        ).reshape(reconstruct_info["shapes"][0])
        threshold = genome[reconstruct_info["indices"][1]]
        reset_coeffs = torch.tensor(
            genome[reconstruct_info["indices"][2]:reconstruct_info["indices"][3]],
            dtype=torch.float32
        ).reshape(reconstruct_info["shapes"][1])
        ret = PMSN_CLR_Params(
            polynomial_coeffs=polynomial_coeffs,
            threshold=threshold,
            reset_coeffs=reset_coeffs,
            state_delta=reconstruct_info["state_delta"],
            n_states=reconstruct_info["n_states"],
            degree=reconstruct_info["degree"],
            surrogate_method=reconstruct_info["surrogate_method"],
            surrogate_alpha=reconstruct_info["surrogate_alpha"]
        )
        if "is_initial" in reconstruct_info and reconstruct_info["is_initial"]:
            pass
        else:
            ret.set_spike_rate(reconstruct_info["spike_rate"])
            ret.set_resting_state(reconstruct_info["resting_state"])
        return ret

    raise ValueError(f"Unsupported params_type for genome reconstruction: {params_type}")
