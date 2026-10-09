# Evolving Spiking Neurons

A repository for exploring spiking neuron models using a evolutionary genetic algorithm with various tasks and hardware evaluation.

## Repository layout

- `lib/` — core library (GA engine, tasks, hardware evaluation, neuron networks, sampling)
- `analysis/` — plotting and inspection helpers
- `configs/` — YAML configs grouped by purpose (see [Configs](#configs)).
- `scripts/` — runnable entry-point scripts, grouped by pipeline stage:
  - `scripts/sampling/` — `sample_neurons.py`, `sample_neurons_LIF.py`
  - `scripts/ga/` — `start_ga.py`, `resume_ga.py`, `resume_ga_fill_hw.py`, `ga_info.py`, `truncate_ga_file.py`
  - `scripts/evaluation/` — `extract_best_neurons.py`, `evaluate_neurons.py`, `aggregate_eval_results.py`, `extract_all_best_neurons.sh`
  - `scripts/hpo/` — `optimize_hyperparams.py`, `plot_hpo_slices.py`
  - `scripts/visualization/` — `visualize_ga.py`, `visualize_single_neuron.py`, `visualize_all_ga.sh`
  - `scripts/training/` — `train_single_neuron.py`
- `tests/` — pytest suite, run via `run_tests.sh` at the repo root.

Scripts are meant to be invoked from the repo root (so relative paths like `configs/...` and `results/...` resolve), e.g. `python3 scripts/ga/start_ga.py ...`.

## Setup

### Prerequisites

This repository assumes a Python environment with PyTorch stack and a working Git installation. A CUDA-capable PyTorch setup is strongly recommended for GA runs and task evaluation, although some scripts can run on CPU for small tests.

Hardware evaluation is optional. If you want to include hardware metrics in the GA fitness, you also need an AMD/Xilinx Vivado + Vitis installation with valid licenses. The hardware flow currently uses Vitis/Vivado `2024.1` by default.

### Python environment

We recommend to use a virtual environment, but this is optional of course. Either way, install requirements from requirements.txt:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The project also depends on the external `esn` package, which is imported as `esn` and is not part of this repository. Install it from the companion repository. 

```bash
pip install git+<https://github.com/nitzsche-fzi/spiking-neurons> 
```

For editable development before publication, download the `spiking-neurons`
repository as a zip from the anonymous artifact page, extract it next to this
repository, and install the extracted folder in editable mode:

```bash
pip install -e ../spiking-neurons
```

### Local paths to configure

Task configs define where datasets are stored. By default they use `./datasets`, relative to the repository root:

- `configs/task/shd.yaml`: `params.dataset_path`
- `configs/task/dvsgesture.yaml`: `params.dataset_path`

The datasets are loaded through `tonic`. If they are not present in the target folder, tonic may download them on first use. Change these paths if you keep
datasets on shared storage or outside the repository. 

Hardware evaluation uses Xilinx, configured in `lib/neuron_eval/hardware/run_hls.sh`:
```bash
XILINX_VERSION="2024.1"
XILINX_SETTINGS="/tools/Xilinx/Vitis/${XILINX_VERSION}/settings64.sh"
```

Adjust `XILINX_VERSION` and `XILINX_SETTINGS` to match your local Vivado/Vitis installation. For example, on some Windows installations the settings file is under `C:/Xilinx/Vitis/<version>/settings64.sh`.

### Disabling hardware evaluation

If you do not have Vivado/Vitis or the required licenses, disable hardware evaluation in the GA config you are using:

```yaml
enable_hardware_eval: false
include_hardware_fitness: false
```

These flags live in `configs/ga/*.yaml`. With both disabled, GA runs use task accuracy only (with spike rate regularization) and do not launch the hardware flow.

### Resource usage

GA runs can be resource intensive. The population size is determined by the number of neurons in the sampled neuron file, so larger populations increase GPU memory use, evaluation time, and hardware-evaluation work. GPU batching is controlled in each GA config with:

```yaml
individuals_per_gb: [15, 15]
gpu_headroom_gb: 0.5
```

`individuals_per_gb` has one entry per task and determines how many individuals are evaluated per GB of available GPU memory. Reduce these values if you run out of GPU memory; increase them cautiously on larger GPUs.
If the total amount of GPU memory (i.e., sum of GPU memory across all selected GPUs) is not sufficient, to evaluate the whole population at once, the GA manager tries to slice the population and evaluate it sequentially.

Hardware evaluation runs Vivado/Vitis jobs in parallel. Configure the number of parallel hardware processes in the GA config:

```yaml
max_hw_processes: 55
```

Each hardware process can require several GB of RAM and significant CPU time. If Vivado fails with out-of-memory errors, the machine becomes unresponsive, or jobs compete too heavily for CPU cores, reduce `max_hw_processes`.
We recommend one CPU core and ~4 GB of RAM per parallel hardware process. The default values are chosen in a way that hardware evaluation roughly takes the same time as task evaluation / SNN training on sufficiently powerful machines.

## Quick start

After completing the setup, you can run the commands below from the repository root to evolve new spiking neurons. This example samples a population of `n2d1` neurons and evolves them with the matching GA config.

1. Sample an initial neuron population:

```bash
python3 scripts/sampling/sample_neurons.py configs/sampling/n2d1.yaml 1000 results/neurons/sampled/1000x_n2d1.pkl
```

2. If you do not have Vivado/Vitis configured, first edit `configs/ga/n2d1.yaml`
and set:

```yaml
enable_hardware_eval: false
include_hardware_fitness: false
```

3. Start a short GA run:

```bash
python3 scripts/ga/start_ga.py configs/ga/n2d1.yaml results/ga/n2d1.pkl 5
```

The last argument is the number of generations. Increase it for a real run, good results can be achieved with 15-30 generations. Intermediate GA state is saved to `results/ga/n2d1.pkl` after each generation, so the run can be continued later:

```bash
python3 scripts/ga/resume_ga.py results/ga/n2d1.pkl 5
```

4. Extract the best evolved neurons:

```bash
python3 scripts/evaluation/extract_best_neurons.py results/ga/n2d1.pkl 20 results/neurons/best/20x_n2d1.pkl --write-config
```

5. Re-evaluate the extracted neurons with repeated trials:

```bash
python3 scripts/evaluation/evaluate_neurons.py results/neurons/best/20x_n2d1.pkl results/neurons/best/config.yaml 10 --output_path n2d1_best
```

6. Optionally, generate diagnostic plots for the GA run:

```bash
python3 scripts/visualization/visualize_ga.py results/ga/n2d1.pkl
```

The visualizer writes plots next to the GA file under `results/ga/visualizations/n2d1/`. It creates progress, population, fitness, energy, survivor, spike-rate, and genome-slice plots. You can pass an optional smoothing window as the second argument, for example:

```bash
python3 scripts/visualization/visualize_ga.py results/ga/n2d1.pkl 20
```

7. You could now go ahead and implement the best neuron in the Evolved Spiking Neurons package, using the already implemented neurons there as blueprint, and use it just like any other neuron in your preferred SNN development flow.

Results are written under `results/`. The first run may also download datasets to the `dataset_path` configured in `configs/task/*.yaml`.

## Advanced usage

### Changing the evaluated tasks

GA configs decide which tasks are evaluated. Edit the `tasks` list in `configs/ga/*.yaml`:

```yaml
tasks: ["configs/task/dvsgesture.yaml", "configs/task/shd.yaml"]
```

To evaluate only one task, remove the other entry. To add a new task, add a new task YAML under `configs/task/` and register its loader in `lib/neuron_eval/tasks/` if it is not already supported. Keep `individuals_per_gb` aligned with the number of tasks, because each entry controls the GPU batch allocation for the corresponding task:

```yaml
tasks: ["configs/task/shd.yaml"]
individuals_per_gb: [15]
```

Task-specific settings such as `batch_size`, `n_epochs`, `layer_sizes`, augmentation parameters, and `dataset_path` live in the task YAML files.

### Changing the sampled neuron family

The PMSN-CLR sampling configs are named by state count and polynomial degree:

- `configs/sampling/n2d1.yaml`: 1 state, degree 2
- `configs/sampling/n2d1.yaml`: 2 states, degree 1
- `configs/sampling/n3d2.yaml`: 3 states, degree 2

To try another family, sample neurons with a different config and make the GA config point to both the sampled population and the matching sampling config:

```bash
python3 scripts/sampling/sample_neurons.py configs/sampling/n3d2.yaml 1000 results/neurons/sampled/1000x_n3d2.pkl
```

```yaml
neurons: "results/neurons/sampled/1000x_n3d2.pkl"
sampling_config: "configs/sampling/n3d2.yaml"
```

The sampling YAML controls the random polynomial distribution, threshold/reset statistics, resting-state search, and accepted spike-rate range. Increasing `n_states`, `degree`, or the maximum number of terms generally increases neuron
expressiveness, but also makes sampling and hardware evaluation more expensive. For example, one could set `enforce_degree` in the YAML to false, allowing the algorithm to find any type of neuron up to the maximum `degree` that is suited best for the given tasks.


# Repository Overview

## Main scripts overview

The scripts under `scripts/` serve different purposes in the exploration workflow. Below is a brief overview of the key scripts. They are explained in the order they would typically be used.

### `sample_neurons.py`
Samples fresh neuron parameters and saves them to disk. This is used to generate an initial population of neurons for evolutionary algorithms. 
Config: sampling config (`configs/sampling/*.yaml`) that defines the neuron family and sampling constraints.

Example call:

```
python3 scripts/sampling/sample_neurons.py configs/sampling/n1d1.yaml 1000 results/sampled_neurons/1000x_n1d1.pkl
```

### `start_ga.py`
Starts a genetic algorithm (GA) to evolve neuron parameters saves a pickle file containing the entire GA state to disk after each generation.
Config: GA config (`configs/ga/*.yaml`) that references tasks and the sampling config.

Example call:

```
python3 scripts/ga/start_ga.py configs/ga/n2d1.yaml results/ga/n2d1.pkl 20
```

### `resume_ga.py`
Resumes a previously started GA from a saved pickle file.
Config: GA pickle from `start_ga.py`, with optional GA config override (`configs/ga/*.yaml`).

Example call:

```
python3 scripts/ga/resume_ga.py results/ga/n2d1.pkl 10
```

### `extract_best_neurons.py`
Extracts the best neurons from a saved GA state pickle file and saves them to disk for further evaluation. Same format as `sample_neurons.py`, but the neurons are selected based on their fitness instead of being randomly generated.
Config: reads GA config from the input pickle; `--write-config` can dump it alongside the output.

Example call:

```
python3 scripts/evaluation/extract_best_neurons.py results/ga/n2d1.pkl 20 results/best_neurons/20x_n2d1.pkl --write-config
```

### `evaluate_neurons.py`
Evaluates a set of neurons saved to disk with a given ga config and saves the results to disk. Evaluates each neuron multiple times to get an average accuracy. This is used to evaluate the best neurons extracted from a GA run, but can in principle be used to evaluate any set of neurons, also randomly sampled ones.
Config: GA config (`configs/ga/*.yaml`), or a copied `config.yaml` from `extract_best_neurons.py --write-config`.

Example call:

```
python3 scripts/evaluation/evaluate_neurons.py results/best_neurons/20x_n2d1.pkl configs/ga/n2d1.yaml 10 results/final/n2d1
```

## Configs

Config files live under `configs/` and are grouped by purpose:
- `configs/sampling/*.yaml`: sampling hyperparameters for neuron population generation (used by `sample_neurons.py`, referenced by GA configs).
- `configs/ga/*.yaml`: GA run settings, including task list, population source, and hardware evaluation toggles.
- `configs/task/*.yaml`: dataset/model/training settings per task (e.g., `configs/task/shd.yaml`).

## Test Suite Runner (`run_tests.sh`)

The repository includes a unified test runner script located at the project root:

```
./run_tests.sh
```

This script ensures a consistent environment for all tests by automatically adjusting the working directory and configuring the `PYTHONPATH` so imports like `from lib...` behave as expected.

The script supports three modes:

#### **1. Standard Tests (default)**

Runs all **non-visual** tests inside `tests/`:

```
./run_tests.sh
```

This excludes anything marked with the `@pytest.mark.visual` decorator.

#### **2. Visual Tests Only**

Runs *only* tests marked as visual:

```
./run_tests.sh --visual
```

Visual tests typically generate plots or state visualizations. All visual outputs are saved automatically into:

```
tmp/
```

(relative to the repository root)

#### **3. Full Test Suite (Including Visual)**

Runs every test in the `tests/` directory:

```
./run_tests.sh --all
```

This mode is helpful before major commits or releases to ensure reproducibility across both analytical and visual components.

---

Visual output files are always written to the `tmp/` directory. Make sure it exists or is git-ignored depending on your workflow.
