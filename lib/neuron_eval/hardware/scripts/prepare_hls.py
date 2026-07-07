"""Sets up an HLS project directory for a single neuron.

Creates the output subdirectories, writes an HLS config file that can be sourced
in Vitis, copies the Vivado testbench/constraints, and generates a Verilog
header (``neuron_config.vh``) whose parameters and port mappings are derived
from the generated C sources so the testbench matches the neuron's interface.

Usage: ``python prepare_hls.py <name> <source_path> <output_dir>``
"""

from pathlib import Path
import shutil
import sys
import re

# Get arguments from command line
if len(sys.argv) != 4:
    print("Usage: python prepare_hls.py <name> <source_path> <output_dir>")
    sys.exit(1)

name = sys.argv[1]
source_path = Path(sys.argv[2]).resolve()
output_dir  = Path(sys.argv[3]).resolve()

print(f"Preparing HLS project for neuron '{name}'")
print(f"Source path: {source_path}")
print(f"Output directory: {output_dir}")

# Prepare folders
script_dir = Path(__file__).resolve().parent
root_dir   = script_dir/".."
output_dir.mkdir(parents=True, exist_ok=True)
logs_dir   = output_dir/"logs"
logs_dir.mkdir(parents=True, exist_ok=True)
vivado_dir = output_dir/"vivado"
vivado_dir.mkdir(parents=True, exist_ok=True)
vitis_dir  = output_dir/"vitis"
vitis_dir.mkdir(parents=True, exist_ok=True)
design_source_path = output_dir/"src"
# Note: the source files are already copied into output_dir/src by
# hardware_evaluator.py, so no copy is performed here.

# Create config in output folder
config_path = output_dir/"hls_config.tcl"
with config_path.open("w") as f:
    f.write(f"set name {{{name}}}\n")
    f.write(f"set source_path {{{design_source_path}}}\n")

# Make a local copy of testbench, headers and constraints
vivado_src_dir = root_dir/"assets"/"vivado"
shutil.copy2(vivado_src_dir/"tb_neuron.sv",    vivado_dir)
shutil.copy2(vivado_src_dir/"assertions.vh",   vivado_dir)
shutil.copy2(vivado_src_dir/"constraints.xdc", vivado_dir)

# Create testbench header based on C code function definition
def count_state_pointers(c_code) -> int:
    """Counts the pointer (state) arguments in the neuron's forward prototype."""
    func_match = re.search(r'bool\s+\w+\s*\(([^)]*)\)', c_code)
    if not func_match:
        return 0

    params = func_match.group(1)
    state_params = [p.strip() for p in params.split(',')
                   if re.search(r'(float|data_t)\s*\*\s*\w+', p)]  # Match float*

    return len(state_params)

def extract_state_init_values(c_code):
    """Maps each state pointer to its initial value from ``init_state``.

    Returns ``(pointer_names, init_values)`` in signature order, defaulting a
    pointer to ``"0.0"`` if ``init_state`` does not assign it.
    """
    init_sig = re.search(r'void\s+init_state\s*\(([^)]*)\)', c_code)
    init_body = re.search(r'void\s+init_state\s*\([^)]*\)\s*{(.*?)}', c_code, re.DOTALL)
    if not init_sig or not init_body:
        raise ValueError("init_state function not found")

    params = init_sig.group(1)
    pointer_names = [
        m.group(1)
        for m in re.finditer(r'(?:float|data_t)\s*\*\s*(\w+)', params)
    ]

    assignments = {}
    for match in re.finditer(r'\*\s*(\w+)\s*=\s*([^;]+);', init_body.group(1)):
        name = match.group(1)
        value = match.group(2).strip()
        data_t_match = re.match(r'data_t\s*\((.*)\)', value)
        if data_t_match:
            value = data_t_match.group(1).strip()
        assignments[name] = value

    init_values = []
    for name in pointer_names:
        if name not in assignments:
            print(f"Warning: init_state does not assign {name}; defaulting to 0.0")
            init_values.append("0.0")
        else:
            init_values.append(assignments[name])

    return pointer_names, init_values

def extract_input_name(c_code):
    """Returns the name of the scalar (non-pointer) input argument, or None."""
    func_match = re.search(r'bool\s+\w+\s*\(([^)]*)\)', c_code)
    if not func_match:
        return None
    params = [p.strip() for p in func_match.group(1).split(',') if p.strip()]
    for param in params:
        if '*' in param:
            continue
        name_match = re.search(r'(\w+)\s*$', param)
        if name_match:
            return name_match.group(1)
    return None

def parse_c_function(c_file):
    """Parses a neuron C source into ``(num_states, pointer_names, init_values, input_name)``."""
    with open(c_file, 'r') as f:
        code = f.read()
    num_states = count_state_pointers(code)
    pointer_names, init_values = extract_state_init_values(code)
    input_name = extract_input_name(code)
    if pointer_names:
        num_states = len(pointer_names)
    return num_states, pointer_names, init_values, input_name



# Generate Verilog header with parameters and port mappings
source_code = list(source_path.glob("*.c*"))[0]
state_count, pointer_names, init_values, input_name = parse_c_function(source_code)
neuron_config_file = vivado_dir/"neuron_config.vh"
with open(neuron_config_file, "w") as f:
    f.write(f"`define STATE_VARIABLES {state_count}\n")
    f.write("// Auto-generated port mappings:\n")
    # Input state ports (s0_i_0, s1_i_0, ...)
    if pointer_names:
        s_i_ports = ", ".join([f".{name}_i_0(uut_state_i_r[{i}])" for i, name in enumerate(pointer_names)])
    else:
        s_i_ports = ", ".join([f".s{i}_i_0(uut_state_i_r[{i}])" for i in range(state_count)])
    f.write(f"`define ALL_S_I_PORTS {s_i_ports}\n")
    # Output state ports (s0_o_0, s1_o_0, ...)
    if pointer_names:
        s_o_ports = ", ".join([f".{name}_o_0(uut_state_o[{i}]), .{name}_o_ap_vld_0(uut_state_valid_o[{i}])"
                              for i, name in enumerate(pointer_names)])
    else:
        s_o_ports = ", ".join([f".s{i}_o_0(uut_state_o[{i}]), .s{i}_o_ap_vld_0(uut_state_valid_o[{i}])"
                              for i in range(state_count)])
    f.write(f"`define ALL_S_O_PORTS {s_o_ports}\n")
    if input_name:
        f.write(f"`define INPUT_PORT .{input_name}_0(uut_weight_sum_i_r)\n")
    else:
        f.write("`define INPUT_PORT .x_0(uut_weight_sum_i_r)\n")
    f.write("// Initial state values:\n")
    f.write(f"parameter real INIT_VALUES [0:{state_count - 1}] = '{{\n")
    for i, value in enumerate(init_values):
        if i == 0:
            f.write(f"  {value}")
        else:
            f.write(f",\n  {value}")
    f.write("\n};\n")
