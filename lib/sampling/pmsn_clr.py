"""Rejection sampling of valid PMSN-CLR neurons for the initial GA population.

Candidate neurons are drawn by sampling random state/reset polynomials, then
filtered so only well-behaved ones survive: their input must reach every state,
the first state must depend on all variables, the resting state must converge,
and the spike rate under random input must fall within configured bounds.
"""
import torch
import pickle
import numpy as np
from tqdm import tqdm

from .polynomial import PolynomialSampler
from ..bootstrapping.pmsn_clr import compute_resting_state, check_influence_of_input, find_influencers_of_state1, input_trials

from esn.neuron_classes import PMSN_CLR, PMSN_CLR_Params

def get_initial_sampler(
        n_states,
        degree,
        threshold_mean,
        threshold_std,
        state_delta,
        surrogate_method,
        surrogate_alpha,
        polynomial_std,
        reset_std,
        min_terms_per_polynomial,
        max_terms_per_polynomial,
        min_terms_per_reset,
        max_terms_per_reset,
        enforce_degree = True):
    """Builds an endless generator of random (unfiltered) PMSN-CLR parameters.

    Returns a generator yielding ``PMSN_CLR_Params`` drawn from the configured
    threshold distribution and polynomial/reset samplers. Validity filtering is
    applied separately by :func:`sample`.
    """
    polynomial_sampler = PolynomialSampler(
        n_states,
        degree,
        polynomial_std,
        min_terms_per_polynomial,
        max_terms_per_polynomial,
        enforce_degree,
        True
    )
    reset_sampler = PolynomialSampler(
        n_states,
        1,
        reset_std,
        min_terms_per_reset,
        max_terms_per_reset,
        False,
        False
    )
    def sample_fn():
        while True:
            yield sample_params(
                n_states,
                degree,
                threshold_mean,
                threshold_std,
                state_delta,
                surrogate_method,
                surrogate_alpha,
                polynomial_sampler,
                reset_sampler
            )
    return sample_fn()

def sample(
        sampler, 
        n_neurons,
        n_states,
        max_rs_steps,
        rs_delta_eps,
        max_state_abs,
        n_trials,
        n_trial_steps,
        min_spike_rate,
        max_spike_rate,
        max_it = 1000,
        report_failed = False):
    """Draws ``n_neurons`` valid neurons from ``sampler`` by rejection sampling.

    Each candidate is checked for input influence, resting-state convergence,
    zero self-spiking, and a random-input spike rate within
    ``[min_spike_rate, max_spike_rate]``; failing candidates are discarded. Up
    to ``max_it`` attempts are made per accepted neuron before raising.

    Args:
        sampler: Generator of candidate ``PMSN_CLR_Params`` (see
            :func:`get_initial_sampler`).
        n_neurons: Number of valid neurons to collect.
        n_states: Number of state variables per neuron.
        max_rs_steps, rs_delta_eps, max_state_abs: Resting-state search settings.
        n_trials, n_trial_steps: Random-input trial settings.
        min_spike_rate, max_spike_rate: Accepted random-input spike-rate range.
        max_it: Maximum attempts per accepted neuron.
        report_failed: If True, also return a dict counting rejection reasons.

    Returns:
        The list of neurons, or ``(neurons, report)`` when ``report_failed``.
    """
    ret_neurons = []
    if report_failed:
        report = {
            "uninfluenced_states": 0,
            "uninfluencing_states": 0,
            "resting_state_exploded": 0,
            "trials_exploded": 0,
            "nonzero_self_spiking": 0,
            "low_random_spiking": 0,
            "high_random_spiking": 0
        }

    with tqdm(total=n_neurons, desc="Sampling neurons", unit="neurons") as pbar:
        while len(ret_neurons) < n_neurons:
            found = False
            for it in range(max_it):
                params = next(sampler)
                neuron = PMSN_CLR(params)
                influences = check_influence_of_input(neuron)
                if not all(influences):
                    if report_failed:
                        report["uninfluenced_states"] += 1
                    continue
                influencers = find_influencers_of_state1(neuron)
                if len(influencers) < (n_states + 1):
                    if report_failed:
                        report["uninfluencing_states"] += 1
                    continue
                resting_state, dead = compute_resting_state(neuron, max_rs_steps, rs_delta_eps, max_state_abs)
                if dead:
                    if report_failed:
                        report["resting_state_exploded"] += 1
                    continue
                neuron.params.set_resting_state(resting_state)
                spike_rate_0, spike_rate_rand, dead = input_trials(neuron, n_trials, n_trial_steps, max_state_abs)
                if dead:
                    if report_failed:
                        report["trials_exploded"] += 1
                    continue
                if spike_rate_0 > 0.0:
                    if report_failed:
                        report["nonzero_self_spiking"] += 1
                    continue
                if spike_rate_rand < min_spike_rate:
                    if report_failed:
                        report["low_random_spiking"] += 1
                    continue
                if spike_rate_rand > max_spike_rate:
                    if report_failed:
                        report["high_random_spiking"] += 1
                    continue
                neuron.params.set_spike_rate(spike_rate_rand)
                ret_neurons.append(neuron)
                pbar.update(1)
                found = True
                break
            if not found:
                if report_failed:
                    print("Failure report:")
                    for key, value in report.items():
                        if isinstance(value, np.ndarray):
                            print(f"{key}: {value.tolist()}")
                        else:
                            print(f"{key}: {value}")
                raise RuntimeError(f"Failed to sample a neuron after {max_it} iterations. Try increasing max_it or change sampling parameters.")

    if report_failed:
        return ret_neurons, report

    return ret_neurons


def sample_params(
    n_states,
    degree,
    threshold_mean,
    threshold_std,
    state_delta,
    surrogate_method,
    surrogate_alpha,
    polynomial_sampler,
    reset_sampler
):
    """Samples one set of raw PMSN-CLR parameters (no validity filtering)."""
    polynomial_coeffs, dense_polynomial_coeffs, polynomial_mask = polynomial_sampler.sample_coeffs()
    threshold = torch.randn(1) * threshold_std + threshold_mean
    reset_coeffs, dense_reset_coeffs, reset_mask = reset_sampler.sample_coeffs()
    return PMSN_CLR_Params(polynomial_coeffs, threshold, reset_coeffs, state_delta, n_states, degree, surrogate_method, surrogate_alpha)

def save_neurons(neurons, filename):
    """Pickles a list of neurons to ``filename``."""
    with open(filename, "wb") as f:
        pickle.dump(neurons, f)
    print(f"Saved neurons to {filename}")

