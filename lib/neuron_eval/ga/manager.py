import time
from pathlib import Path

import torch.multiprocessing as mp
import yaml
from queue import Empty
import pickle
import random
import itertools

from pathlib import Path
from queue import Empty
from tqdm import tqdm

import numpy as np
import torch.multiprocessing as mp

from lib.neuron_eval.tasks import get_task
from lib.utils import get_average_smi_data
from lib.neuron_eval.hardware.hardware_evaluator import HardwareEvaluator

from .genome import flatten_genome, reconstruct_genome
from lib.sampling.pmsn_clr import sample

from sparse_data_modeling import SparseDataModel

# Known limitation: task configs are referenced by path and re-read when a GA
# state is loaded, rather than being stored inside the saved GA file. Loading a
# saved run therefore assumes the task config files still match the originals.

class GAManager:
    """Drives one genetic-algorithm run that evolves a population of neurons.

    A run is a sequence of generations. Each generation evaluates every
    individual on all configured tasks (training a shared-architecture network
    per individual and, optionally, running a hardware energy evaluation in
    parallel), computes a fitness that combines accuracy across tasks with an
    optional hardware-energy penalty, selects the top survivors, and samples a
    new population from them. The full history of generations is pickled to
    ``output_file`` after every generation so a run can be resumed.

    Evaluation is distributed across GPUs: individuals are packed into batches
    sized to each device's memory and evaluated in worker subprocesses that
    report progress back over a multiprocessing queue.
    """

    @staticmethod
    def init_new_run(config: dict, output_file: str):
        """Creates a fresh GA run from a config and an initial neuron population."""
        print("Loading neurons from:", config["neurons"])
        # load the neurons
        with open(config["neurons"], "rb") as f:
            neuron_data = pickle.load(f)
            neuron_data["results"] = {} # Initialize results dictionary
            history = [] 
            ret = GAManager(config, neuron_data, history, output_file)
        print(f"Loaded {len(neuron_data['params'])} neurons")
        return ret

    @staticmethod
    def load(input_file: str, output_file: str, config: dict = None):
        """Resumes a GA run from a pickled state file.

        The most recent generation in the saved history becomes the current
        population. An explicit ``config`` overrides the saved one (the caller
        is responsible for keeping it compatible).
        """
        print("Loading genetic algorithm state from:", input_file)
        if config is not None:
            print("  Overriding saved config with provided config. Make sure it is compatible!")
        # the file is a pickle of a dict containing "config" and "history"
        with open(input_file, "rb") as f:
            data = pickle.load(f)
            if config is None:
                config = data["config"]
            history = data["history"]
            ret = GAManager(config, history[-1], history, output_file)
        print(f"  Found {len(history)} generations.")
        print(f"  Loaded {len(history[-1]['params'])} neurons from history")
        return ret

    def __init__(self, config: dict, neuron_data: dict, history: list, output_file: str):
        mp.set_start_method('spawn', force=True)  # Ensure the spawn method is used for multiprocessing
        self.train_message_queue = mp.Queue() # the message queue where subprocesses send their training information
        self.tasks = {}
        for i, task_path in enumerate(config["tasks"]):
            with open(task_path, "rb") as f:
                task_config = yaml.safe_load(f)
                task_name = f"{task_config['name']}_{i:02d}"
                self.tasks[task_name] = get_task(task_config)
                self.tasks[task_name].set_name(task_name) #set the name of the task
        self.config = config
        self.neuron_data = neuron_data
        self.neuron_type = config.get("neuron_type", "pmsn_clr")
        self.population_size = len(self.neuron_data["params"])
        #neuron data is of the currently most recent generation.
        self.history = history
        self.random_seed = config.get("random_seed", random.randint(0, 1000000))  # Use a random seed if not provided
        self.output_file = output_file
        self.individuals_per_gb = config["individuals_per_gb"]
        self.devices = []
        self.max_num_individuals = {} #map from (task_name, gpu_index) to maximal number of individuals to evaluate on that GPU
        self.n_survivors = config["n_survivors"]
        self.process_counters = {}
        print("Random seed: ", self.random_seed)

        if output_file is not None:
            self.output_dir_name = Path(output_file).stem
            self.results_hw_dir  = Path(output_file).resolve().parent / "hardware_eval"
        self.hw_eval_enabled    = config.get("enable_hardware_eval", False)
        self.include_hw_fitness = config.get("include_hardware_fitness", False)
        self.max_hw_processes   = config.get("max_hw_processes", 10)
        self.hw_fitness_factor  = config.get("hardware_fitness_factor", 0.1)
        self.hw_idle_energy_factor = config.get("hardware_idle_energy_factor", 1.0)
        self.hw_results_queue = mp.Queue()

    def eval_population(self):
        """Evaluates the current population on every task and saves the result.

        When hardware evaluation is enabled it runs concurrently with task
        evaluation. The finished generation is appended to the history and the
        whole run is persisted to disk.
        """
        print(f"Starting evaluation of generation {len(self.history)}")
        time_eval_start = time.time()
        
        if self.hw_eval_enabled:
            # Run hardware evaluation in parallel to task evaluation
            hw_manager_process = self._eval_population_hw_start(self.neuron_data)
            
        # Run task evaluation 
        time_task_start = time.time()
        for task_name, task in self.tasks.items():
            self._evaluate_task(task_name, task)
        time_task_end = time.time() - time_task_start
        print(f"  Task evaluation took: {time_task_end:.1f} seconds")

        if self.hw_eval_enabled:
            print("  Waiting for hardware evaluation to complete...")
            self.neuron_data["hw_results"] = self._eval_population_hw_finish(hw_manager_process)
        
        time_eval_end = time.time() - time_eval_start
        print(f"  Total evaluation time: {time_eval_end:.1f} seconds")

        self.history.append(self.neuron_data)  # Append the latest generation's data to history
        self.save() # then save the results to the output file


    def _evaluate_task(self, task_name, task):
        self._initialize_task_state(task_name)
        batches = self._create_batches(task_name)
        self._validate_batches_cover_population(batches)

        for batch_idx, batch in enumerate(batches, start=1):
            print(f"  Evaluating task {task_name} - batch {batch_idx}/{len(batches)}")
            processes, individual_ranges, active_devices = self.start_task(task_name, task, batch)
            self.listen_to_messages(task_name, processes, individual_ranges, active_devices)

    def _initialize_task_state(self, task_name):
        self.accuracies = [[] for _ in range(self.population_size)]
        self.ma_accuracies = [0.0 for _ in range(self.population_size)]
        self.eval_results = [None for _ in range(self.population_size)]
        self.eval_spike_counts = [None for _ in range(self.population_size)]
        self.eval_timesteps = [None for _ in range(self.population_size)]
        self.spike_rates = [[] for _ in range(self.population_size)]
        self.last_mean_train = 0.0
        self.last_mean_eval = 0.0
        self.max_grad = []
        self.n_training_steps = 0
        self.n_spiking_neurons = {}
        self.process_counters[task_name] = 0

    def _create_batches(self, task_name):
        capacities = []
        for device in self.devices:
            max_individuals = self.max_num_individuals.get((task_name, device), 0)
            if max_individuals > 0:
                capacities.append((device, max_individuals))
        if not capacities:
            raise RuntimeError(f"No evaluation capacity available for task {task_name}.")

        total_capacity_per_batch = sum(cap for _, cap in capacities)
        n_batches = (self.population_size + total_capacity_per_batch - 1) // total_capacity_per_batch
        
        # Create batches with balanced distribution
        batches = []
        start_index = 0
        remaining = self.population_size
        
        for batch_idx in range(n_batches):
            if remaining <= 0:
                break
            
            # Per-batch capacities reset each batch
            per_batch_caps = {device: cap for device, cap in capacities}
            batch_size = min(total_capacity_per_batch, remaining)
            batch = []
            batch_start = start_index
            # Sort devices by capacity (high to low) for fair distribution
            sorted_devices = sorted(per_batch_caps.items(), key=lambda x: x[1], reverse=True)
            n_gpus = len(sorted_devices)
            total_available = sum(cap for _, cap in sorted_devices if cap > 0)
            effective_batch_size = min(batch_size, total_available)
            if effective_batch_size <= 0:
                break
            per_gpu_target = effective_batch_size // n_gpus
            remainder = effective_batch_size % n_gpus

            allocated = 0
            # First pass: allocate targets
            for i, (device, cap) in enumerate(sorted_devices):
                if allocated >= effective_batch_size or remaining <= 0:
                    break
                target = per_gpu_target + (1 if i < remainder else 0)
                num = min(target, cap, effective_batch_size - allocated, remaining)
                if num > 0:
                    batch.append((device, batch_start, batch_start + num))
                    batch_start += num
                    allocated += num
                    per_batch_caps[device] -= num
                    remaining -= num

            # Second pass: if still need to fill effective_batch_size, redistribute leftover to devices with spare cap
            if allocated < effective_batch_size and remaining > 0:
                remaining_to_allocate = min(effective_batch_size - allocated, remaining)
                for device, cap in sorted(per_batch_caps.items(), key=lambda x: x[1], reverse=True):
                    if remaining_to_allocate <= 0 or cap <= 0:
                        continue
                    # Find this device in batch and add more individuals
                    for j, (d, start, end) in enumerate(batch):
                        if d == device and remaining_to_allocate > 0 and cap > 0:
                            additional = min(remaining_to_allocate, cap)
                            batch[j] = (d, start, end + additional)
                            per_batch_caps[device] -= additional
                            remaining_to_allocate -= additional
                            remaining -= additional
                            batch_start += additional
                            break
            if batch:
                batches.append(batch)
                start_index = batch_start
        return batches

    def _validate_batches_cover_population(self, batches):
        # Flatten and sort all (start, end) ranges across all batches
        all_ranges = []
        for batch in batches:
            for _, start, end in batch:
                all_ranges.append((start, end))
        if not all_ranges:
            raise RuntimeError("Batch planning produced no ranges.")
        all_ranges.sort(key=lambda r: r[0])
        # Check contiguous, non-overlapping, full coverage
        total = 0
        expected_start = 0
        for i, (start, end) in enumerate(all_ranges):
            if start != expected_start:
                raise RuntimeError(
                    f"Batch planning error: gap or overlap at segment {i}. "
                    f"Expected start {expected_start}, found {start}. "
                    f"First few ranges: {all_ranges[:5]}"
                )
            if end <= start:
                raise RuntimeError(
                    f"Batch planning error: non-positive length range ({start}, {end})."
                )
            seg_len = end - start
            total += seg_len
            expected_start = end
        if total != self.population_size:
            raise RuntimeError(
                f"Batch planning error: total covered {total} != population_size {self.population_size}. "
                f"Last range ends at {all_ranges[-1][1]}."
            )


    def _eval_population_hw_start(self, neuron_data) -> mp.Process:
        hw_evaluator = HardwareEvaluator(
            self.output_dir_name, len(self.devices), len(self.history), self.results_hw_dir)
        hw_manager_process = mp.Process(
            target=hw_evaluator.evaluate_population,
            args=(neuron_data, self.hw_results_queue, self.max_hw_processes)
        )
        hw_manager_process.start()
        return hw_manager_process

    def _eval_population_hw_finish(self, hw_manager_process: mp.Process) -> dict:
        hw_results = self.hw_results_queue.get()
        # Rejoin and clean up the hardware evaluation process
        hw_manager_process.join(timeout=10)
        if hw_manager_process.is_alive():
            print("[WARN] Hardware evaluator process did not exit after results retrieval. Terminating...")
            hw_manager_process.terminate()
            hw_manager_process.join()
        return hw_results
    

    def _collect_all_individuals(self):
        return self._collect_individuals(range(len(self.history)))

    def _collect_individuals(self, generations: list):
        all_genomes, all_infos, all_accs, all_energy = [], [], [], []

        for gen_idx in generations:
            gen = self.history[gen_idx]
            acc_matrix  = self._collect_accuracies(gen)
            spike_rates = self._collect_spike_rates(gen)
            hw_metrics  = self._collect_hardware_metrics(gen)
            energy      = self._compute_energy(spike_rates, hw_metrics)
            genomes, infos = self._extract_genomes(gen["params"], range(len(gen["params"])))

            for i, g in enumerate(genomes):
                all_genomes.append(g)
                all_infos.append(infos[i])
                all_accs.append(acc_matrix[:, i])
                if energy is not None:
                    all_energy.append(energy[:, i])

        all_accs = np.stack(all_accs, axis=1)
        all_energy = np.stack(all_energy, axis=1) if all_energy else None

        if all_energy is None:
            print("[WARN] No hardware energy data available, proceeding without energy in fitness calculation.")

        return all_genomes, all_infos, all_accs, all_energy

    def _select_top_individuals(self, n, generations: list = None):
        generations = range(len(self.history)) if generations is None else generations
        all_genomes, all_infos, all_accs, all_energy = self._collect_individuals(generations)

        fitness = self._compute_fitness(all_accs, all_energy, method="worst_rank")
        top_idx = np.argsort(fitness)[:n]

        return [all_genomes[i] for i in top_idx], [all_infos[i] for i in top_idx]

    def sample_population(self):
        """Selects the fittest survivors and samples the next generation from them.

        The survivors' genomes are fitted with a :class:`SparseDataModel` and
        the sampler draws new neurons from that distribution, applying any
        neuron-type-specific validity constraints. The resulting population
        replaces ``self.neuron_data``.
        """
        survivors, survivor_infos = self._select_top_individuals(self.n_survivors)

        cfg = self.history[-1].get("sampling_config")
        if cfg is None:
            cfg_path = self.config.get("sampling_config")
            if cfg_path is None:
                raise KeyError("No sampling configuration found in history or config.")
            with open(cfg_path, "rb") as f:
                cfg = yaml.safe_load(f)
        survivors = np.array(survivors)
        new_neurons, report = self._sample_new_population(survivors, survivor_infos, cfg)

        self._update_neuron_data(new_neurons, cfg, report)


    def _collect_accuracies(self, parents: dict):
        return np.array([
            parents["results"][task]["eval_results"]
            for task in parents["results"]
        ])  # shape (n_tasks, pop_size)

    def _collect_spike_rates(self, parents: dict) -> np.ndarray:
        try:
            eval_spike_counts = self._collect_eval_spike_counts(parents)
            eval_timesteps = self._collect_eval_total_timesteps(parents)
            
            # Calculate per-neuron spike rates for each task
            spike_rates = []
            for task in parents["results"]:
                n_spiking_neurons = parents["results"][task]["n_spiking_neurons"]
                task_spike_counts = eval_spike_counts[len(spike_rates)]
                task_timesteps = eval_timesteps[len(spike_rates)]
                task_spike_rate = task_spike_counts / (n_spiking_neurons * task_timesteps)
                spike_rates.append(task_spike_rate)
            
            return np.array(spike_rates)
        except (KeyError, TypeError):
            # Fall back to training spike rates if evaluation data not available
            print("[WARN:_collect_spike_rates] No evaluation data available, using training spike rates.")
            return np.array([
                np.array(parents["results"][task]["spike_rates"])[:, -1]
                for task in parents["results"]
            ])  # shape (n_tasks, pop_size)

    def _collect_eval_spike_counts(self, parents: dict):
        return np.array([
            parents["results"][task]["eval_spike_counts"]
            for task in parents["results"]
        ])  # shape (n_tasks, pop_size)

    def _collect_eval_total_timesteps(self, parents: dict):
        return np.array([
            parents["results"][task]["eval_timesteps"]
            for task in parents["results"]
        ])  # shape (n_tasks, pop_size)

    def _collect_hardware_metrics(self, parents: dict):
        if not self.hw_eval_enabled:
            return None
        if "hw_results" not in parents:
            print("[WARN: _collect_hardware_metrics] Hardware evaluation results missing.")
            return None
        
        hw_metrics = []
        for i in range(len(parents["params"])):
            hw_result = parents["hw_results"][i]
            # Ignore logic, spike_in energy and latency for now, just track them in the GA history
            metrics = [hw_result["energy"]["idle"], hw_result["energy"]["spike_out"]]
            hw_metrics.append(metrics)
        return np.array(hw_metrics).T  # shape (n_hw_metrics, pop_size)

    def _compute_energy(self, spike_rates: np.ndarray, hw_metrics: np.ndarray) -> np.ndarray:
        """Compute per-task energy from spike rates and hardware energy metrics. This is heavily
        influenced by the SNN architecture used to measure spike rates.

        Args:
            spike_rates: Array of shape (n_tasks, pop_size) with spike rates per neuron and task.
            hw_metrics: Array of shape (2, pop_size) with [energy_idle, energy_spike_out].

        Returns:
            np.ndarray: Energy per task and neuron with shape (n_tasks, pop_size) or None if hw_metrics is None.
        """
        if hw_metrics is None or spike_rates is None:
            print("[WARN:_compute_energy] Missing hardware metrics or spike rates, cannot compute energy.")
            return None
        
        # This is a approximation: it does not account for how many spikes
        # a neuron receives on average (which would use the per-spike_in energy).
        # On modern neuromorphic hardware, multiple logical neurons share one 
        # physical neuron core and are processed using time-multiplexing.
        # Hence idle energy is typically neglectable. However, it can still be 
        # used for energy calculation if desired. The relevance is chosen by hw_idle_energy_factor.
        idle_energy = hw_metrics[0]      # (pop_size,)
        spike_out_energy = hw_metrics[1] # (pop_size,)
        return (1.0 - spike_rates) * idle_energy[None, :] * self.hw_idle_energy_factor + spike_rates * spike_out_energy[None, :]

    def _compute_fitness(self, accs, energy = None, method = "worst_rank", verbose = True):
        # Optionally incorporate normalized energy by subtracting a weighted penalty per task/neuron
        # the fitness is computed such that **LOWER IS BETTER**
        adjusted_accs = accs
        if energy is not None and self.include_hw_fitness:
            energy_normalized = self._normalize_energy(energy)
            adjusted_accs = accs - self.hw_fitness_factor * energy_normalized
        elif verbose:
            print("[INFO: _compute_fitness] Energy is not included in fitness calculation.")

        if method == "worst_rank":
            ranks = np.argsort(np.argsort(-adjusted_accs, axis=1), axis=1)
            fitness = ranks.max(axis=0)
        elif method == "sum_rank":
            ranks = np.argsort(np.argsort(-adjusted_accs, axis=1), axis=1)
            fitness = ranks.sum(axis=0)
        elif method == "sum_fitness":
            fitness = adjusted_accs.sum(axis=0) * -1  
        else:
            raise ValueError(f"Unknown fitness method '{method}'")         
        return fitness

    def _normalize_energy(self, energy: np.ndarray) -> np.ndarray:
        # Filter out inf values from failed evaluations
        finite_mask = np.isfinite(energy)
        finite_metrics = np.where(finite_mask, energy, np.nan)

        min_vals = np.nanmin(finite_metrics, axis=1, keepdims=True)  # min of finite values
        max_vals = np.nanmax(finite_metrics, axis=1, keepdims=True)  # max of finite values
        
        # Handle case where max = min (all values are the same)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        return np.where(finite_mask, (energy - min_vals) / range_vals, 1.0)

    def _extract_genomes(self, params_list, indices):
        genomes = []
        infos = []
        for idx in indices:
            g, info = flatten_genome(params_list[idx], self.neuron_type)
            genomes.append(g)
            infos.append(info)
        return np.array(genomes), infos

    def _sample_new_population(self, genomes, reconstruct_infos, cfg):
        """Builds sampler with reconstruct_info, runs sampling."""
        model = SparseDataModel(genomes)
        neutral_reconstruct_info = reconstruct_infos[0].copy()
        neutral_reconstruct_info["is_initial"] = True
        sampler = self._build_sampler(model, neutral_reconstruct_info, cfg)
        n_neurons = self.population_size

        if self.neuron_type != "pmsn_clr":
            neurons = [next(sampler) for _ in range(n_neurons)]
            report = {"sampled": len(neurons)}
            return neurons, report

        neurons, report = sample(
            sampler,
            n_neurons=n_neurons,
            n_states=cfg["n_states"],
            max_rs_steps=cfg["max_rs_steps"],
            rs_delta_eps=cfg["rs_delta_eps"],
            max_state_abs=cfg["max_state_abs"],
            n_trials=cfg["n_trials"],
            n_trial_steps=cfg["n_trial_steps"],
            min_spike_rate=cfg["min_spike_rate"],
            max_spike_rate=cfg["max_spike_rate"],
            report_failed=True,
        )
        return neurons, report
    
    def _build_sampler(self, data_model, reconstruct_info, cfg):
        """Returns a generator that reconstructs each sampled genome via reconstruct_info."""
        if self.neuron_type != "pmsn_clr":
            sampling_report = {}

            def sampler_fn():
                while True:
                    batch = data_model(2000)
                    for flat in batch:
                        yield reconstruct_genome(flat, reconstruct_info, self.neuron_type)

            self._sampling_report = sampling_report
            return sampler_fn()

        # PMSN-CLR specific constraints
        pre_combos = list(
            itertools.chain.from_iterable(
                itertools.combinations_with_replacement(range(cfg["n_states"] + 1), d)
                for d in range(cfg["degree"] + 1)
            )
        )
        max_degree = max(len(c) for c in pre_combos)
        idx_max_deg = {i for i, c in enumerate(pre_combos) if len(c) == max_degree}
        sampling_report = {"not_max_degree": 0, "term_violations": 0}

        def sampler_fn():
            while True:
                batch = data_model(2000)
                for flat in batch:
                    neuron = reconstruct_genome(flat, reconstruct_info, self.neuron_type)
                    coeffs = neuron.polynomial_coeffs
                    if cfg["enforce_degree"]:
                        coeffs = neuron.polynomial_coeffs
                        if not any((coeffs[:, i] != 0).any().item() for i in idx_max_deg):
                            sampling_report["not_max_degree"] += 1
                            continue
                    num_nonzero_terms = (coeffs != 0).sum(dim=1)  # per polynomial
                    if not ((num_nonzero_terms >= cfg["min_terms_per_polynomial"]) &
                            (num_nonzero_terms <= cfg["max_terms_per_polynomial"])).all():
                        sampling_report.setdefault("term_violations", 0)
                        sampling_report["term_violations"] += 1
                        continue
                    yield neuron

        # keep report accessible externally
        self._sampling_report = sampling_report
        return sampler_fn()

    def _update_neuron_data(self, neurons, cfg, report):
        report.update(self._sampling_report)
        params_list = []
        for neuron in neurons:
            params_list.append(neuron.params if hasattr(neuron, "params") else neuron)
        self.neuron_data = {
            "params": params_list,
            "sampling_report": report,
            "sampling_config": cfg,
            "results": {},
        }

    def start_task(self, task_name, task, batch):
        """Launches one worker subprocess per device range in ``batch``.

        Returns the started processes together with their individual index
        ranges and active devices, for :meth:`listen_to_messages` to track.
        """
        processes = []
        individual_ranges = []
        active_devices = []

        for device, start_index, end_index in batch:
            num_individuals = end_index - start_index
            if num_individuals <= 0:
                continue
            params = [params for params in self.neuron_data["params"][start_index:end_index]]
            process_seed = (self.random_seed + hash(task_name) + self.process_counters[task_name]) % 1000000
            self.process_counters[task_name] += 1
            process = mp.Process(
                target=task.evaluate,
                args=(process_seed, params, self.neuron_type, device, self.train_message_queue),
            )
            process.start()
            processes.append(process)
            individual_ranges.append((start_index, end_index))
            active_devices.append(device)
            print(
                f"Started evaluation for task {task_name} on device {device} with {num_individuals} individuals (indices {start_index}-{end_index - 1})."
            )

        return processes, individual_ranges, active_devices

    def listen_to_messages(self, task_name, processes, individual_ranges, active_devices):
        """Consumes worker messages until every device has finished this task.

        Updates the training/evaluation progress bars, collects per-individual
        results as they arrive, and raises if a worker process dies. The
        aggregated results are stored under ``self.neuron_data["results"]``.
        """
        if not active_devices:
            return

        self.individual_ranges = individual_ranges
        self._active_devices = list(active_devices)
        self._active_device_to_range_idx = {device: idx for idx, device in enumerate(self._active_devices)}
        self.progresses = {device: 0.0 for device in self._active_devices}
        self.eval_progresses = {device: 0.0 for device in self._active_devices}
        self.process_states = {device: "training" for device in self._active_devices}
        self.last_mean_train = 0.0
        self.last_mean_eval = 0.0

        self._initialize_progress_bars(self._active_devices)

        while self.process_states:
            try:
                message = self.train_message_queue.get()
                device = message["device"]
                if device not in self._active_device_to_range_idx:
                    continue
                msg_type = message["type"]

                if msg_type == "training started":
                    if self.n_training_steps == 0:
                        self.n_training_steps = message["n_training_steps"]
                        self.max_grad = [0.0 for _ in range(self.n_training_steps)]
                    self.n_spiking_neurons[task_name] = message["n_spiking_neurons"]

                if msg_type == "training progress":
                    self._update_training_progress(device, message)

                elif msg_type == "training finished":
                    if device in self.process_states:
                        self.process_states[device] = "eval"

                elif msg_type == "evaluation progress":
                    self._update_eval_progress(device, message["progress"])

                elif msg_type == "evaluation finished":
                    range_idx = self._active_device_to_range_idx.get(device)
                    if range_idx is None:
                        continue
                    start_index, end_index = self.individual_ranges[range_idx]
                    self.eval_results[start_index:end_index] = message["results"]
                    self.eval_spike_counts[start_index:end_index] = message["total_spike_counts"]
                    self.eval_timesteps[start_index:end_index] = [message["total_timesteps"]] * (end_index - start_index)
                    if device in self.process_states:
                        del self.process_states[device]

            except Empty:
                for process, device in zip(processes, self._active_devices):
                    if not process.is_alive() and device in self.process_states:
                        print(f"[ERROR] Process for device {device} died unexpectedly.")
                        raise RuntimeError("Evaluation subprocess crashed.")

            except Exception as e:
                raise e

        for process in processes:
            process.join()

        self._finalize_bars()

        self.neuron_data["results"][task_name] = {
            "accuracies": self.accuracies,
            "spike_rates": self.spike_rates,
            "eval_results": self.eval_results,
            "eval_spike_counts": self.eval_spike_counts,
            "eval_timesteps": self.eval_timesteps,
            "n_spiking_neurons": self.n_spiking_neurons.get(task_name),
            "max_grad": self.max_grad,
        }
    
    def _update_training_progress(self, device, message):
        if self.n_training_steps == 0:
            return

        progress = message["step"] / self.n_training_steps
        accuracies = message["accuracies"]
        spike_rate = message["spike_rate"]
        step = message["step"]

        self.max_grad[step - 1] = max(self.max_grad[step - 1], message["max_grad"])

        range_idx = self._active_device_to_range_idx.get(device)
        if range_idx is None:
            return
        index_range = self.individual_ranges[range_idx]

        for i in range(index_range[0], index_range[1]):
            acc = accuracies[i - index_range[0]]
            self.accuracies[i].append(acc)
            self.ma_accuracies[i] = self.ma_accuracies[i] * 0.9 + acc * 0.1
            self.spike_rates[i].append(spike_rate[i - index_range[0]])

        new_progress = progress * 100.0
        delta = new_progress - self.progresses[device]
        self.progresses[device] = new_progress
        self.training_bars[device].update(delta)

        mean = sum(self.progresses.values()) / len(self.progresses)
        mean_delta = mean - self.last_mean_train
        self.last_mean_train = mean
        self.mean_training_bar.update(mean_delta)

        best_ma_acc = max(self.ma_accuracies) if self.ma_accuracies else 0.0
        self.mean_training_bar.set_postfix(best_ma_acc=f"{best_ma_acc:.3f}")

    def _update_eval_progress(self, device, progress):
        range_idx = self._active_device_to_range_idx.get(device)
        if range_idx is None:
            return

        new_progress = progress * 100.0
        delta = new_progress - self.eval_progresses[device]
        self.eval_progresses[device] = new_progress
        self.eval_bars[device].update(delta)

        mean = sum(self.eval_progresses.values()) / len(self.eval_progresses)
        mean_delta = mean - self.last_mean_eval
        self.last_mean_eval = mean
        self.mean_eval_bar.update(mean_delta)


    def _initialize_progress_bars(self, active_devices):
        self.training_bars = {}
        self.eval_bars = {}

        for i, device in enumerate(active_devices):
            self.training_bars[device] = tqdm(
                total=100,
                position=i,
                desc=f"[Train] {device}",
                leave=True,
                unit="%",
                bar_format="{l_bar}{bar}|{n:.1f}%•{rate_fmt}•{elapsed}<{remaining}",
                dynamic_ncols=True
            )

        self.mean_training_bar = tqdm(
            total=100,
            position=len(active_devices),
            desc="[Train] Mean",
            leave=True,
            unit="%",
            bar_format="{l_bar}{bar}|{n:.1f}%•{rate_fmt}•{elapsed}<{remaining}•{postfix}",
            dynamic_ncols=True
        )

        for i, device in enumerate(active_devices):
            self.eval_bars[device] = tqdm(
                total=100,
                position=len(active_devices) + 1 + i,
                desc=f"[Eval] {device}",
                leave=True,
                unit="%",
                bar_format="{l_bar}{bar}|{n:.1f}%•{rate_fmt}•{elapsed}<{remaining}",
                dynamic_ncols=True
            )

        self.mean_eval_bar = tqdm(
            total=100,
            position=2 * len(active_devices) + 1,
            desc="[Eval] Mean",
            leave=True,
            unit="%",
            bar_format="{l_bar}{bar}|{n:.1f}%•{rate_fmt}•{elapsed}<{remaining}",
            dynamic_ncols=True
        )

    def _finalize_bars(self):
        for bar in list(self.training_bars.values()) + list(self.eval_bars.values()):
            bar.n = 1.0
            bar.refresh()
            bar.close()
        self.mean_training_bar.n = 1.0
        self.mean_training_bar.refresh()
        self.mean_training_bar.close()
        self.mean_eval_bar.n = 1.0
        self.mean_eval_bar.refresh()
        self.mean_eval_bar.close()

    def compute_individuals_per_gpu(self, task_name):
        """Splits the population across GPUs for one task.

        The split is weighted by each GPU's per-task capacity so that more
        capable devices receive proportionally more individuals; any rounding
        remainder is distributed round-robin so the totals match the population
        size exactly.
        """
        num_gpus = len(self.devices)
        if num_gpus == 0:
            print("No GPUs available for evaluation. Exiting.")
            exit(1)
        max_individuals_per_gpu = [self.max_num_individuals.get((task_name, device), 0) for device in self.devices]
        gpu_weights = [max_individuals / sum(max_individuals_per_gpu) for max_individuals in max_individuals_per_gpu]
        # Calculate the number of individuals per GPU based on the population size and the weights
        individuals_per_gpu = [int(self.population_size * weight) for weight in gpu_weights]
        # Ensure that the total number of individuals matches the population size
        total_individuals = sum(individuals_per_gpu)
        if total_individuals < self.population_size:
            # Distribute the remaining individuals to the GPUs in a round-robin fashion
            for i in range(self.population_size - total_individuals):
                individuals_per_gpu[i % num_gpus] += 1
        elif total_individuals > self.population_size:
            # Reduce the number of individuals per GPU to match the population size
            for i in range(total_individuals - self.population_size):
                individuals_per_gpu[i % num_gpus] -= 1
        print(f"Individuals per GPU for task {task_name}: {individuals_per_gpu}")
        return individuals_per_gpu

    def show_gpu_dialog(self):
        """Interactively selects GPUs and records their per-task capacity.

        Queries ``nvidia-smi``, prompts the user to include each GPU, and for
        every accepted device computes how many individuals fit per task given
        its free memory (minus the configured headroom).
        """
        print("Querying GPU information via nvidia-smi...")
        smi_avg = get_average_smi_data(20)
        if not smi_avg:
            print("No GPUs detected by nvidia-smi. Exiting.")
            exit(1)
        indices = sorted(smi_avg.keys())
        print(f"Found {len(indices)} GPUs:")
        for i in indices:
            info = smi_avg[i]
            util = info.get('utilization_gpu', -1)
            print(f"--- Device {i} ---")
            print(f"  Name: {info['name']}")
            print(f"  Capability: {info.get('compute_cap', 'unknown')}")
            print(f"  Total Memory: {info['memory_total_gb']:.2f} GB")
            print(f"  Free Memory: {info['memory_free_gb']:.2f} GB")
            print(f"  Used Memory: {(info['memory_total_gb'] - info['memory_free_gb']):.2f} GB")
            print(f"  Utilization: {util}%")
            print(f"  PCI Bus ID: {info['pci_bus_id']}")
            print("do you want to use this GPU? (y/n)")
            choice = input(f"Use GPU {i} (y/n): ").strip().lower()
            choice = choice[0] if choice else choice
            if choice == 'y':
                free_memory_gb = max(0.0, info['memory_free_gb'] - self.config["gpu_headroom_gb"])
                if free_memory_gb <= 0:
                    print(f"Not enough free memory on GPU {i} after headroom. Skipping this GPU.")
                    continue
                for j, (task_name, task) in enumerate(self.tasks.items()):
                    # Calculate the maximum number of individuals for this task on this GPU
                    max_individuals = int(free_memory_gb * self.individuals_per_gb[j])
                    if max_individuals <= 0:
                        print(f"Not enough memory for task {task_name} on GPU {i}. Skipping this task.")
                        continue
                    self.max_num_individuals[(task_name, f"cuda:{i}")] = max_individuals
                    print(f"  Task {task_name}, GPU {i}: Max individuals = {max_individuals}")
                self.devices.append(f"cuda:{i}")
            elif choice == 'n':
                print(f"Skipping GPU {i}.")
            else:
                print("Invalid input, skipping this GPU.")
        if not self.devices:
            print("No GPUs selected. Exiting.")
            exit(1)
        print(f"Selected devices: {self.devices}")
        # For each task, print the maximum number of individuals per GPU
        for task_name in self.tasks.keys():
            print(f"Task {task_name}:")
            sum_max_individuals = 0
            for gpu_index in indices:
                max_individuals = self.max_num_individuals.get((task_name, f"cuda:{gpu_index}"), 0)
                sum_max_individuals += max_individuals
            print(f"  Total max individuals across all GPUs: {sum_max_individuals}")
            if sum_max_individuals < self.population_size:
                print(
                    f"  [INFO] Population size {self.population_size} exceeds single-round capacity. "
                    "Will evaluate in multiple batches."
                )

    def save(self):
        """Pickles the full run (config + generation history) to ``output_file``."""
        if self.output_file is not None:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "wb") as f:
                pickle.dump({
                    "config": self.config,
                    "history": self.history
                }, f)
            print(f"Saved GA state to {self.output_file}")
        else:
            print(f"[WARN] Could not save GA state since output file is not given.")
