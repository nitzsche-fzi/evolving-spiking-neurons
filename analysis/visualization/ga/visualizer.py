"""Shared data-access layer for the GA visualisation scripts."""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lib.neuron_eval.ga.manager import GAManager

class GAVisualizer:
    """Loads a saved GA run and exposes its metrics for plotting.

    Wraps a :class:`GAManager` loaded from a pickle and provides convenience
    accessors used by the ``plot_*`` modules: per-generation population accuracy,
    spike rate and energy, the best individual per metric, consistent plot
    styles, and small axis/legend helpers.
    """

    def __init__(self, ga_file):
        self.ga_file = ga_file
        self.output_path = self._build_output_path()
        self.temp_ga_file = Path(self.output_path) / Path(ga_file).name
        self.manager = GAManager.load(ga_file, self.temp_ga_file)
        self.config = self.manager.config
        self.tasks = self._get_tasks()
        self.n_generations = len(self.manager.history)
        self.has_hardware = "hw_results" in self.manager.history[0]
        self.relevant_metrics = ["fitness", "accuracy", "energy"] if self.has_hardware else ["fitness", "accuracy"]
        self.top_individuals = self._find_top_individuals(consider_past_gens=True)
        self.pop_task_accuracy = self.get_pop_accuracy_all_gens()
        self.styles = self._define_metric_styles()


    def get_best_individual_property(self, prop: str, metric: str) -> np.ndarray:
        """
        Returns the specified property of the best individual per task per generation,
        where best is determined by the given metric. Property can be one of 'accuracies' or 'spike_rate'.
        Metric can be one of 'fitness', 'accuracy', or 'energy'.
        shape (n_tasks, n_generations)
        """
        return np.array([
            [self.top_individuals[gen_idx][metric][prop][task] for gen_idx in range(self.n_generations)] for task in self.tasks
        ])

    # For all generations
    def get_pop_accuracy_all_gens(self) -> np.ndarray:
        """
        Return the population accuracy per task for all generations.
        shape (n_tasks, n_generations)
        """
        return np.stack(
            [self.get_pop_accuracy(gen_idx) for gen_idx in range(self.n_generations)], axis=1
        )

    def get_pop_spike_rate_all_gens(self) -> np.ndarray:
        """
        Return the population spike rate per task for all generations.
        shape (n_tasks, n_generations)
        """
        return np.stack(
            [self.get_pop_spike_rate(gen_idx) for gen_idx in range(self.n_generations)], axis=1
        )

    def get_pop_energy_all_gens(self, normalize: bool = False) -> np.ndarray:
        """
        Return the population energy per task for all generations.
        shape (n_tasks, n_generations)
        """
        if normalize:
            return np.stack(
                [self.get_normalized_energy(gen_idx).mean(axis=1) for gen_idx in range(self.n_generations)], axis=1
            )
        else:
            return np.stack(
                [self.get_pop_energy(gen_idx) for gen_idx in range(self.n_generations)], axis=1
            )

    # Per generation
    def get_pop_accuracy(self, gen_idx: int) -> np.ndarray:
        """Return the population accuracy per task for a given generation."""
        generation = self.manager.history[gen_idx]
        task_accuracies = self.manager._collect_accuracies(generation) # shape (n_tasks, pop_size)
        return np.mean(task_accuracies, axis=1)

    def get_pop_spike_rate(self, gen_idx: int) -> np.ndarray:
        """Return the population spike rate per task for a given generation."""
        generation = self.manager.history[gen_idx]
        spike_rates = self.manager._collect_spike_rates(generation) # shape (n_tasks, pop_size)
        return np.mean(spike_rates, axis=1)

    def get_pop_energy(self, gen_idx: int) -> np.ndarray:
        """Return the population energy per task for a given generation."""
        generation = self.manager.history[gen_idx]
        spike_rates = self.manager._collect_spike_rates(generation)
        hw_metrics  = self.manager._collect_hardware_metrics(generation)
        energy      = self.manager._compute_energy(spike_rates, hw_metrics) # shape (n_tasks, pop_size)
        return np.ma.masked_invalid(energy).mean(axis=1)

    def get_normalized_energy(self, gen_idx: int) -> np.ndarray:
        gens = range(gen_idx + 1)
        _, _, _, energy = self.manager._collect_individuals(gens)
        normalized_energy = self.manager._normalize_energy(energy)
        # Return the most recent generation's normalized energy (last row).
        return normalized_energy[-1]


    def select_survivors(self, fitness: np.ndarray, survivor_factor: float, non_survivor_factor: float = 0.4):
        """
        Select survivors based on given fitness.
        Uses survivor_factor to determine the number of survivors (survivor_factor * population_size)
        and next best (non_survivor_factor * population_size) as non-survivors.
        """
        n_survivors = int(self.manager.population_size * survivor_factor)
        n_non_survivors = int(self.manager.population_size * non_survivor_factor) 
        sorted_indices = np.argsort(fitness)  # Sort in ascending order (best first)
        survivor_indices = sorted_indices[:n_survivors]  
        non_survivor_indices = sorted_indices[n_survivors:n_survivors+n_non_survivors]
        return survivor_indices, non_survivor_indices


    def prettify_task_name(self, task_name: str) -> str:
        # Remove number suffix (e.g., dvsgesture_00 -> dvsgesture)
        base_name = task_name.split('_')[0]
        # Convert to prettier names
        pretty_names = {
            'dvsgesture': 'DVSGesture',
            'shd': 'SHD',
            'nmnist': 'N-MNIST',
        }
        return pretty_names.get(base_name, base_name.title())

    def smooth_training_data(self, data: np.ndarray, smoothing: int = 20) -> np.ndarray:
        """Apply simple moving average across the second axis with cold start."""
        if smoothing <= 0 or data.ndim != 2:
            return data
        smoothed = []
        for i in range(data.shape[1]):
            window_size = min(i + 1, smoothing)
            window = data[:, max(0, i - window_size + 1):i + 1]
            current_val = np.mean(window, axis=1)
            smoothed.append(current_val)
        return np.array(smoothed).T

    def get_y_axis_ticks(self, values: list[np.ndarray], n_ticks_y: int = 11, step: float = 10) -> tuple[np.ndarray, float]:
        def _floor_to_nearest(n: float, step: float) -> float:
            return np.floor(n / step) * step
        def _ceil_to_nearest(n: float, step: float) -> float:
            return np.ceil(n / step) * step

        min_value = 0
        max_value = _ceil_to_nearest(np.nanmax(values), step)
        steps = np.empty(0)
        while len(steps) < n_ticks_y:
            step_size = _ceil_to_nearest((max_value - min_value) / (n_ticks_y - 1), step)
            steps = np.arange(min_value, max_value + step_size, step_size)
            step /= 2
        return steps, step_size

    def combine_legends(self, axes: list[plt.Axes]) -> tuple[list[plt.Line2D], list[str]]:
        """Combine matplotlib legends from multiple axes, removing the original legends."""
        handles, labels = list(), list()
        for ax in axes:
            handles_ax, labels_ax = ax.get_legend_handles_labels()
            handles.extend(handles_ax)
            labels.extend(labels_ax)
            ax.legend().remove()
        return handles, labels

    
    def _load_ga_state(self, ga_file: str):
        """Load GA state pickle and return the dict."""
        print("Loading GA state from:", ga_file)
        with open(ga_file, "rb") as f:
            ga_state = pickle.load(f)
        print("GA state loaded successfully.")
        print("Number of generations:", len(ga_state.get("history", [])))
        return ga_state


    def _get_tasks(self):
        """Return a list of task names."""
        latest = self.manager.history[-1]
        return list(latest.get("results", {}).keys())


    def _build_output_path(self) -> str:
        """Build and create the visualization output path for a GA pickle path."""
        ga_path = Path(self.ga_file)
        output_path = ga_path.parent / "visualizations" / ga_path.stem
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving visualizations to: {output_path}")
        return str(output_path)


    def _define_metric_styles(self) -> dict:
        """Define consistent colors, markers and line styles for all metrics."""
        styles = {
            "accuracy": {
                "mean":       {"color": "black",      "marker": "s", "linestyle": "--"},
                "DVSGesture": {"color": "tab:blue",   "marker": "o", "linestyle": "-"},
                "SHD":        {"color": "tab:orange", "marker": "o", "linestyle": "-"},
            },
            "accuracy_pop": {
                "mean":       {"color": "grey",      "marker": "s", "linestyle": ":"},
                "DVSGesture": {"color": "skyblue", "marker": "o", "linestyle": ":"},
                "SHD":        {"color": "sandybrown",    "marker": "o", "linestyle": ":"},
            },
            "spike_rate": {
                "DVSGesture": {"color": "lightskyblue",  "marker": "d", "linestyle": "-."}, #dodgerblue
                "SHD":        {"color": "goldenrod",    "marker": "d", "linestyle": "-."},
            },
            "energy": {
                "total":      {"color": "tab:red",     "marker": "x", "linestyle": "-"},
                "hw_low":     {"color": "dimgray",   "marker": "x", "linestyle": ":"},
                "hw_so":      {"color": "tab:brown", "marker": "x", "linestyle": "--"},
                "DVSGesture": {"color": "tab:cyan",  "marker": "^", "linestyle": "--"},
                "SHD":        {"color": "tab:olive", "marker": "^", "linestyle": "--"},
            },
        }
        return styles


    def _find_top_individuals(self, consider_past_gens: bool = True) -> list[dict]:
        """Find the top individuals per generation for fitness (overall and per task), accuracy, energy and hardware energy."""
        top_individuals = list()
        all_spike_rates, all_hardware_metrics = np.array([]).reshape(2, 0), np.array([]).reshape(2, 0)
        for gen_idx, gen in enumerate(self.manager.history):
            # Gather all data for relevant generations
            relevant_gens = list(range(gen_idx + 1)) if consider_past_gens else [gen_idx]
            all_genomes, all_infos, all_accs, all_energy = self.manager._collect_individuals(relevant_gens)
            if consider_past_gens and all_energy is not None:
                all_spike_rates      = np.concatenate([all_spike_rates, self.manager._collect_spike_rates(gen)], axis=1)
                all_hardware_metrics = np.concatenate([all_hardware_metrics, self.manager._collect_hardware_metrics(gen)], axis=1)
            elif all_energy is not None:
                all_spike_rates      = self.manager._collect_spike_rates(gen)
                all_hardware_metrics = self.manager._collect_hardware_metrics(gen)

            # Find best individual by metrics: fitness, accuracy, energy and hardware energy
            scores = dict()
            scores["fitness"]  = self.manager._compute_fitness(all_accs, all_energy, method="worst_rank")
            scores["accuracy"] = self.manager._compute_fitness(all_accs, None,       method="worst_rank", verbose=False)
            if all_energy is not None:
                scores["energy"] = np.sum(all_energy, axis=0)  # Sum energy across all tasks
                scores["energy_hw"] = all_hardware_metrics[1, :] # Hardware energy for 'spike out' operation
            # Add per-task fitness categories (best per single task)
            for task_idx, task_name in enumerate(self.tasks):
                task_accs = all_accs[task_idx:task_idx+1, :]
                task_energy = all_energy[task_idx:task_idx+1, :] if all_energy is not None else None
                scores[f"fitness_{task_name}"] = self.manager._compute_fitness(task_accs, task_energy, method="worst_rank")
            top_in_gen = dict()
            for metric, fitness in scores.items():
                stats = self._get_best_individual_stats(fitness, all_accs, all_energy, relevant_gens)
                stats["spike_rate"] = {task: all_spike_rates[i, stats["global_index"]] for i, task in enumerate(self.tasks)} if all_spike_rates.size > 0 else None
                stats["hw_metrics"] = all_hardware_metrics[:, stats["global_index"]] if all_hardware_metrics.size > 0 else None
                stats["fitness"]    = scores["fitness"][stats["global_index"]]
                stats["genome"]     = all_genomes[stats["global_index"]]
                stats["info"]       = all_infos[stats["global_index"]]
                top_in_gen[metric] = stats
            top_individuals.append(top_in_gen)
        return top_individuals

    def _get_best_individual_stats(self, fitness, all_accs, all_energy, relevant_gens):
        individual_idx = np.argmin(fitness)  # Lower fitness is better
        accuracies = {task: all_accs[i, individual_idx] for i, task in enumerate(self.tasks)}
        energy = {task: all_energy[i, individual_idx] for i, task in enumerate(self.tasks)} if all_energy is not None else None
        gen, local_idx = self.__find_gen_and_local_idx(individual_idx, relevant_gens)
        return {"global_index": individual_idx, "generation": gen, "local_index": local_idx, "accuracies": accuracies, "energy": energy}

    def __find_gen_and_local_idx(self, global_idx, relevant_gens):
        """Find which generation and local index an individual's global index corresponds to."""
        cumulative_size = 0
        for g in relevant_gens:
            gen_size = len(self.manager.history[g]["params"])
            if global_idx < cumulative_size + gen_size:
                local_idx = global_idx - cumulative_size
                return g, local_idx
            cumulative_size += gen_size
        return None, None