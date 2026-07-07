#!/usr/bin/env python3
"""Prints the best-trial summary of a single-neuron training study.

Loads an Optuna study from a database and displays its best trial, optionally
including auto-discovered hardware evaluation and energy results.

Examples:
    python3 scripts/visualization/visualize_single_neuron.py \\
        results/train/lifbox_shd_train.db

    # Include hardware eval and energy (auto-discovered):
    python3 scripts/visualization/visualize_single_neuron.py \\
        results/train/lifbox_shd_train.db --hw_eval
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import argparse
import os
import json
import pickle
from pathlib import Path
import optuna


def display_results_summary(study_db_path: str, include_hw_eval: bool = False):
    """
    Display a summary of training and optional hardware/energy evaluation results.
    
    Paths are inferred from the study db path:
    - results_dir = parent(study_db_path)
    - study_name  = basename(study_db_path).replace('.db','')
    - GA state pickle = results_dir / f"{study_name}.pkl"
    - energy json = results_dir / f"{study_name}_energy_analysis.json"
    
    Args:
        study_db_path: Path to the Optuna study database
        include_hw_eval: Whether to search for and display hardware and energy results
    """
    study_db = Path(study_db_path)
    results_dir = study_db.parent
    study_name = study_db.stem
    energy_json_path = results_dir / f"{study_name}_energy_analysis.json"
    ga_pickle_path = results_dir / f"{study_name}.pkl"

    # Load the study
    study = optuna.load_study(storage=f"sqlite:///{study_db}", study_name=None)
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Training results
    print(f"Study database: {study_db}")
    print(f"Best accuracy: {study.best_value:.6f}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best parameters: {study.best_params}")

    # Hardware and energy (auto-discovered)
    if include_hw_eval:
        # Network energy (from JSON)
        if energy_json_path.exists():
            try:
                with open(energy_json_path, "r") as f:
                    energy_data = json.load(f)
                print("\nNetwork energy (from JSON):")
                task_names = energy_data.get("task_names") or []
                energy_per_task = energy_data.get("energy_per_task") or []
                total_energy = energy_data.get("total_energy")
                spike_rates = energy_data.get("spike_rates_per_task") or []
                if total_energy is not None:
                    print(f"  Total energy: {float(total_energy):.2f} pJ")
                if task_names and energy_per_task and len(task_names) == len(energy_per_task):
                    for i, name in enumerate(task_names):
                        e = float(energy_per_task[i])
                        sr = spike_rates[i] if i < len(spike_rates) else None
                        if sr is None:
                            print(f"  {name}: {e:.2f} pJ")
                        else:
                            print(f"  {name}: {e:.2f} pJ (spike_rate={float(sr):.4f})")
            except Exception as e:
                print(f"Could not load energy JSON ({energy_json_path}): {e}")
        else:
            print(f"\nEnergy JSON not found (expected): {energy_json_path}")

        # Hardware results from GA pickle
        if ga_pickle_path.exists():
            try:
                with open(ga_pickle_path, "rb") as f:
                    ga_state = pickle.load(f)
                # Expect keys: {"config", "history"}
                history = ga_state.get("history", [])
                latest = history[-1] if history else None
                if latest and "hw_results" in latest:
                    hw_results = latest["hw_results"]
                    print(f"\nHardware results from GA pickle: {ga_pickle_path.name}")
                    # Show metrics for individual 0 if present (single-neuron run), otherwise list first entry
                    idx = 0 if 0 in hw_results else (next(iter(hw_results.keys())) if hw_results else None)
                    if idx is not None:
                        metrics = hw_results[idx]
                        logic = metrics.get("logic")
                        latency = metrics.get("latency")
                        energy = metrics.get("energy", {})
                        if logic is not None:
                            print(f"  Logic usage: {float(logic):.2f} LUTs")
                        if latency is not None:
                            print(f"  Latency: {float(latency):.2f} ns")
                        if energy:
                            idle = energy.get('idle')
                            spike_in = energy.get('spike_in')
                            spike_out = energy.get('spike_out')
                            if idle is not None:
                                print(f"  Energy (idle): {float(idle):.2f} pJ")
                            if spike_in is not None:
                                print(f"  Energy (spike_in): {float(spike_in):.2f} pJ")
                            if spike_out is not None:
                                print(f"  Energy (spike_out): {float(spike_out):.2f} pJ")
                    else:
                        print("  No hardware metrics found in pickle.")
                else:
                    print("\nNo hardware results in GA pickle history.")
            except Exception as e:
                print(f"Could not load GA pickle ({ga_pickle_path}): {e}")
        else:
            print(f"\nGA pickle not found (expected): {ga_pickle_path}")
    
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display results summary for an Optuna study."
    )
    parser.add_argument("study_db", type=str, help="Path to the Optuna study database (.db file)")
    parser.add_argument("--hw_eval", action="store_true", help="Include hardware evaluation and energy (auto-discovered)")
    args = parser.parse_args()

    # Check if study database exists
    if not os.path.exists(args.study_db):
        print(f"Error: Study database not found: {args.study_db}")
        exit(1)

    display_results_summary(args.study_db, include_hw_eval=args.hw_eval)
