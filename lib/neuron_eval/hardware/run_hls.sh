#!/bin/bash
# Usage: Call this from the hardware dir as working dir!
# Example: run_hls.sh n2d2_1cc8c973 "results/n2d2_1cc8c973/src" "results/ga/hardware_eval/n2d2_hw_01"

XILINX_VERSION="2024.1"
# XILINX_SETTINGS="C:/Xilinx/Vitis/${XILINX_VERSION}/settings64.sh"
XILINX_SETTINGS="/tools/Xilinx/Vitis/${XILINX_VERSION}/settings64.sh"
#TODO, not important: check path and prompt user to fix xilinx path otherwise

source $XILINX_SETTINGS

#####
## Arg parsing
#####
name=$1
source_path=$2
results_dir=$3

SECONDS=0
total_seconds=0

# Dirs
work_dir=$(pwd)
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scripts_dir="${root_dir}/scripts"
reports_flow_dir="${scripts_dir}/create_reports"
output_dir="${results_dir}/$name"
logs_dir="${output_dir}/logs"

# Write args to temporary config file since new Vitis does not support tcl args.
# Also create subdirectory structure in outputs dir and adjust testbench for Vivado.
# Do this in Python to have easy cross-platform support (creating dirs in bash on Windows is a pain)
python3 $scripts_dir/prepare_hls.py "$name" "$source_path" "$output_dir"
if [ $? -ne 0 ]; then
    echo "Python preparation script failed with error code $?"
    exit 1
fi
echo ""
echo "HLS preparation finished. Elapsed time: ${SECONDS}s"

#####
## Generate neuron logic via HLS
#####
# Note: The working dir in Vitis tcl mode will be determined by the location of the tcl file,
#       not the --work_dir param passed below. Make sure to change to the correct folder using
#       'cd' inside the tcl script!
# Run Vitis and filter INFO lines, but check vitis-run's actual exit code so we don't continue with Vivado if Vitis failed.
echo ""
echo "Running Vitis HLS with: vitis-run --mode hls --tcl ${root_dir}/vitis_generate_hw.tcl --work_dir '$output_dir/vitis'"
total_seconds=$SECONDS
SECONDS=0
vitis_log="$logs_dir/vitis.txt"
vitis_log_warn="$logs_dir/vitis_warn.txt"
OUTPUT_DIR="$output_dir" vitis-run --mode hls --tcl ${root_dir}/vitis_generate_hw.tcl --work_dir "$output_dir/vitis" > $vitis_log
exit_code_vitis=$?
grep -e "^WARNING: " -e "^ERROR: " $vitis_log > $vitis_log_warn
if [ $exit_code_vitis -ne 0 ]; then
    echo "VITIS HLS failed with error code $exit_code_vitis"
    echo "See $vitis_log_warn for details"
    exit 1
fi
echo "VITIS HLS finished. Elapsed time: ${SECONDS}s"
echo ""


# Place & route logic and generate reports via Vivado
((total_seconds+=SECONDS))
SECONDS=0
echo "Running Vivado with: vivado_main.tcl -tclargs $name $output_dir '-mode out_of_context'"
# Note: We cannot call Vivado from git-bash on Windows, as the reported system architecture is wrong in that case.
#       Instead we call powershell from bash first, but this breaks cross-platform compatibility.
# FIXME: Find a way to differentiate between git-bash on Windows and actual Linux system.
cd "$output_dir/vivado"
vivado_log="$logs_dir/vivado.txt"
vivado_log_warn="$logs_dir/vivado_warn.txt"
# powershell.exe -Command "vivado.bat -mode batch -source ../../../vivado_main.tcl -tclargs $name" > $vivado_log
vivado -mode batch -source ${root_dir}/vivado_main.tcl -tclargs $name $output_dir '-mode out_of_context' > $vivado_log
exit_code_vivado=$?
grep -e "^WARNING: " -e "^ERROR: " $vivado_log > $vivado_log_warn
if [ $exit_code_vivado -ne 0 ]; then
    echo "Vivado failed with error code $exit_code_vivado"
    echo "See $vivado_log_warn for details"
    exit 1
fi
echo "Vivado finished. Elapsed time: ${SECONDS}s"
echo ""

# copy log to reports directory, so it can be analyzed
cp $vivado_log "$output_dir/reports/vivado.log"

#####
## Analyze Vivado reports to get logic usage and power estimations
#####
cd $work_dir
((total_seconds+=SECONDS))
SECONDS=0
report_parser_log="$logs_dir/report_parser.txt"
echo ""
python3 $reports_flow_dir/output_parser.py "design_1_wrapper" 100 "$output_dir/reports"
python3 $scripts_dir/process_parsed_report.py "$output_dir/reports/parsed_report.json"
echo "Report parsing finished. Elapsed time: ${SECONDS}s"
echo ""
echo "run_hls finished. Total elapsed titme: ${total_seconds}"