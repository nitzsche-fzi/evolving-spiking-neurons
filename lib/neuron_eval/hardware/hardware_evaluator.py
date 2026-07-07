from pathlib import Path
import subprocess
import json
import time
import shutil
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty

from esn.neuron_classes import PMSN_CLR


class HardwareEvaluator:
    """Estimates FPGA cost metrics for neurons via an HLS/synthesis toolchain.

    For each neuron the evaluator generates C++/H sources, runs them through
    Xilinx Vitis HLS and Vivado (synthesis, simulation and power estimation)
    and returns logic usage, latency and per-state energy. A whole population
    is evaluated in a pool of worker subprocesses, and every neuron's build
    artifacts are written under a per-generation output directory; failed
    builds are copied aside for debugging.
    """

    def __init__(self, run_dir_name: str, num_devices: int = 0, generation: int = 0, output_main_dir: Path = None):
        self.script_dir = Path(__file__).resolve().parent
        self.root_dir = self.script_dir / ".." / ".." / ".."
        if output_main_dir is None:
            self.output_dir = self.root_dir / "results" / "ga" / "hardware_eval"
        else:
            self.output_dir = output_main_dir
        self.output_dir = self.output_dir / run_dir_name / f"gen_{generation:02d}"
        self.error_dir  = self.output_dir / "errors"
        self.error_dir.mkdir(parents=True, exist_ok=True)
        print(f"HardwareEvaluator - Output directory: {self.output_dir}")
        print(f"HardwareEvaluator - Error directory: {self.error_dir}")
        print(f"HardwareEvaluator - run_dir_name: {run_dir_name}, generation: {generation}, num_devices: {num_devices} ")
        self.fixed_point = False
        self.fixed_point_W = 32  # word length
        self.fixed_point_I = 12  # Integer bits incliding sign
        self.num_devices = num_devices
        self.generation = generation

    def evaluate_population(self, neuron_data: dict, output_queue: mp.Queue, max_parallel: int = 50):
        """Run hardware evaluation for all neurons in neuron_data in separate processes.
        
        Args:
            neuron_data: Dictionary containing 'params' list with neuron parameters
            output_queue: Queue to put the final results dictionary on
            max_parallel: Maximum number of parallel evaluation processes
        """
        population_size = len(neuron_data["params"])
        max_hw_processes = max(1, min(int(max_parallel), population_size))
        print(f"HardwareEvaluator - Manager started (max {max_hw_processes} parallel processes, {population_size} individuals total)")
        
        # Create temporary queue for collecting results
        temp_queue = mp.Queue()
        hw_processes = []
        completed_count = 0
        # Collect results as they come in to avoid blocking producers
        hw_results = {}
        error_count = 0
        failed_individuals = list()

        def _accumulate_result(message: dict):
            nonlocal error_count
            individual_idx = message["individual_idx"]
            hw_results[individual_idx] = message["hw_metrics"]
            error_count += message.get("error", 0)
            if message.get("error", 0):
                name = message.get("name")
                failed_individuals.append(name)

        def _drain_results_nonblocking():
            """Drain all available results from temp_queue without blocking."""
            while True:
                try:
                    message = temp_queue.get_nowait()
                    _accumulate_result(message)
                except Empty:
                    break

        def _cleanup_finished_processes():
            """Join finished processes, update progress, and drain any available results."""
            nonlocal completed_count
            jobs_bar.refresh()
            for p in hw_processes[:]:
                if not p.is_alive():
                    p.join()
                    hw_processes.remove(p)
                    completed_count += 1
                    jobs_bar.update(1)
            _drain_results_nonblocking()
        
        # Start evaluation for all individuals
        tqdm_position = 2 * self.num_devices + 2 if self.num_devices > 0 else 0 # Lower positions are used by SNN train/eval bars
        jobs_bar = tqdm(total=population_size, desc="[HW] Jobs finished", leave=True, unit="job", position=tqdm_position)
        for individual_idx in range(population_size):
            # Wait for a free slot if needed
            while len(hw_processes) >= max_hw_processes:
                _cleanup_finished_processes()
                time.sleep(1)
            # Start new worker
            p = mp.Process(
                target=self._evaluate_neuron_process,
                args=(individual_idx, neuron_data["params"][individual_idx], temp_queue)
            )
            p.start()
            hw_processes.append(p)

        print(f"HardwareEvaluator - All HW processes started, waiting for {len(hw_processes)} processes to finish...")
        while len(hw_processes) > 0:
            _cleanup_finished_processes()
            time.sleep(1)
        jobs_bar.close()
        print(f"HardwareEvaluator - All HW processes finished, collecting any remaining results...")

        # Final drain for any remaining results (block until all results are collected)
        while len(hw_results) < population_size:
            message = temp_queue.get()
            _accumulate_result(message)
        
        print(f"HardwareEvaluator - Manager finished - processed {completed_count} individuals from which {error_count} failed.")
        if error_count > 0:
            print(f"                    Failed individuals are: {', '.join(failed_individuals)}")
        output_queue.put(hw_results)
        # Ensure the writer end is closed in this process so the parent can fully drain and the feeder thread can terminate cleanly
        output_queue.close()


    def _evaluate_neuron_process(self, individual_idx: int, neuron_params: dict, message_queue: mp.Queue):
        """Evaluate hardware metrics for a single individual and put result on queue."""
        neuron = self._create_neuron(neuron_params)
        try:
            hw_metrics = self.evaluate_neuron(neuron)
            
            # Clean up temporary files after successful evaluation
            neuron_out_dir = self.output_dir / neuron.get_identifier()
            self._cleanup_temp_files(neuron_out_dir)
            
            message_queue.put({
                "individual_idx": individual_idx,
                "hw_metrics": hw_metrics,
                "error": 0
            })
        except Exception as e:
            # Common failure modes are running out of disk space ([Errno 28]) or
            # the synthesis tools running out of RAM ([Errno 139]). On any error
            # we report the worst possible metrics so the individual is penalised
            # under the minimization objectives.
            print(f"HardwareEvaluator - Error while evaluting individual {individual_idx}, name: {neuron.get_identifier()}: {str(e)}")
            print(f"                    If the error happened during Vivado and the error message is weird, this is likely due to insufficient RAM.")
            print(f"                    Try to either reduce max_hw_processes in the config or add more RAM.")
            
            # Copy the output folder of the failed neuron to the error directory
            neuron_out_dir = self.output_dir / neuron.get_identifier()
            self._copy_failed_neuron_to_error_dir(neuron_out_dir, neuron.get_identifier())
            
            # Possible improvement: detect OOM errors here and reduce max_parallel
            # for subsequent runs.
            message_queue.put({
                "individual_idx": individual_idx,
                "hw_metrics": {
                    "logic": float('inf'),
                    "latency": float('inf'),
                    "energy": {
                        "idle": float('inf'),
                        "spike_in": float('inf'),
                        "spike_out": float('inf')
                    }
                },
                "error": 1,
                "name": neuron.get_identifier()
            })
    def _create_neuron(self, neuron_params):
        """Instantiate the correct neuron class for hardware evaluation."""
        if hasattr(neuron_params, "polynomial_coeffs"):
            return PMSN_CLR(neuron_params)
        raise ValueError("Unsupported neuron params for hardware evaluation")


    def evaluate_neuron(self, neuron) -> dict:
        """
        Evaluate the hardware metrics for a given neuron by spawning a subprocess that runs Xilinx Vitis HLS and Vivado.
        Returns a dictionary:
        - "logic": float, the logic usage of the neuron in LUT equivalents
        - "latency": float, the latency of the neuron in cycles (1 cycle = 10ns)
        - "energy": dict with floats, the energy consumption for idle, spike_in and spike_out states in pJ
        """
        # Create folder structure for output files
        name = neuron.get_identifier()
        neuron_out_dir = self.output_dir / name
        neuron_src_dir = neuron_out_dir / "src"
        neuron_src_dir.mkdir(parents=True, exist_ok=True)
        scores_path = neuron_out_dir / "reports" / "scores.json"
        log_dir = neuron_out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run_hls.log"
        # Generate cpp/h files for this neuron to use with HLS
        self._store_c_code(neuron, neuron_src_dir)

        # Run HLS, synthesis/implementation, simulation and power estimation
        with open(log_file, 'w') as f:
            result = subprocess.run(["bash", "run_hls.sh", name, neuron_src_dir, self.output_dir], cwd=self.script_dir, stdout=f, stderr=f)
        if result.returncode != 0:
            raise RuntimeError(f"run_hls.sh failed with return code {result.returncode}")
            
        # Load hardware results and process them
        with open(scores_path) as f:
            scores_dict = json.load(f)
        power = scores_dict.get("power")
        scores_dict["energy"] = dict()
        for key in power.keys():
            scores_dict["energy"][key] = scores_dict["latency"] * power[key]
        if scores_dict["latency"] == 0:
            print(f"HardwareEvaluator - Latency is 0 for {name}, this should not happen. Investigate!")
        return scores_dict

    def _store_c_code(self, neuron, target_dir: Path):
        forward_code, state_init_code, forward_signature, state_init_signature = neuron.get_c_code()
        c_file = target_dir / "neuron.cpp"
        h_file = target_dir / "neuron.h"

        with c_file.open("w") as f:
            f.write(
                f"""\
// File generated by hardware_evaluator.py

#include "neuron.h"

{state_init_code}
{forward_code}
            """
            )

        with h_file.open("w") as f:
            f.write(
                f"""\
// File generated by hardware_evaluator.py

#ifndef SPIKING_NEURONS_H
#define SPIKING_NEURONS_H

{"//" if not self.fixed_point else ""} #define USE_FIXED_POINT

#ifdef USE_FIXED_POINT
  #include <ap_fixed.h>
  typedef ap_fixed<{self.fixed_point_W}, {self.fixed_point_I}> data_t;
#else
  typedef float data_t;
#endif

#include <stdbool.h>
#include <math.h>
#include <cmath>

{forward_signature};
{state_init_signature};  

#endif
"""
            )

    def _copy_failed_neuron_to_error_dir(self, neuron_out_dir: Path, neuron_name: str):
        """
        Copy the hardware output folder of a failed neuron to the error directory for debugging.
        """
        try:
            error_neuron_dir = self.error_dir / neuron_name
            if error_neuron_dir.exists():
                # If directory already exists, add timestamp to avoid conflicts
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                error_neuron_dir = self.error_dir / f"{neuron_name}_{timestamp}"
            
            shutil.copytree(neuron_out_dir, error_neuron_dir)
            print(f"HardwareEvaluator - Copied output of failed neuron {neuron_name} to: {error_neuron_dir}")
        except Exception as e:
            print(f"HardwareEvaluator - Error copying failed neuron {neuron_name} to error directory: {str(e)}")

    def _cleanup_temp_files(self, neuron_out_dir: Path):
        """
        Clean up temporary files after successful evaluation, keeping only essential files:
        - src folder
        - ip.zip
        - results/scores.json, results/parsed_report.json
        - logs/run_hls.log, logs/vitis_warn.txt and logs/vivado_warn.txt
        - vivado/neuron_config.vh, vivado/tb_neuron.sv and vivado/vivado.log
        """
        try:        
            # Clean up ip/ip subdirectory (temporary IP core files)
            ip_subdir = neuron_out_dir / "ip" / "ip"
            if ip_subdir.exists():
                shutil.rmtree(ip_subdir)
            
            # Clean up reports folder - keep only scores.json and parsed_report.json
            reports_dir = neuron_out_dir / "reports"
            if reports_dir.exists():
                for item in reports_dir.iterdir():
                    if item.is_file() and item.name not in ["scores.json", "parsed_report.json"]:
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            
            # Clean up logs folder - keep only run_hls.log, vitis_warn.txt and vivado_warn.txt
            logs_dir = neuron_out_dir / "logs"
            if logs_dir.exists():
                for item in logs_dir.iterdir():
                    if item.is_file() and item.name not in ["run_hls.log", "vitis_warn.txt", "vivado_warn.txt"]:
                        item.unlink()
            
            # Clean up vivado folder - keep only neuron_config.vh and tb_neuron.sv
            vivado_dir = neuron_out_dir / "vivado"
            if vivado_dir.exists():
                for item in vivado_dir.iterdir():
                    if item.is_file() and item.name not in ["neuron_config.vh", "tb_neuron.sv", "vivado.log"]:
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            
            # Remove vitis folder entirely (temporary build files)
            vitis_dir = neuron_out_dir / "vitis"
            if vitis_dir.exists():
                shutil.rmtree(vitis_dir)
            
            # Remove hls_config.tcl (temporary config file)
            config_file = neuron_out_dir / "hls_config.tcl"
            if config_file.exists():
                config_file.unlink()
        except Exception as e:
            print(f"HardwareEvaluator - Warning: Failed to cleanup temp files for {neuron_out_dir.name}: {str(e)}")
