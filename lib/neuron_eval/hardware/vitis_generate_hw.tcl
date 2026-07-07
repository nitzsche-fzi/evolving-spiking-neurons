# argv 0 are Vitis command flags, argv 1 is the script name
# Get output_dir from environment variable
if {[info exists env(OUTPUT_DIR)]} {
    set output_dir $env(OUTPUT_DIR)
} else {
    puts "vitis_generate_hw.tcl - ERROR: OUTPUT_DIR environment variable is not set"
    puts "This script must be called with OUTPUT_DIR set to the correct output directory"
    exit 1
}

set config_path [file join $output_dir hls_config.tcl]
# Source HLS config created by run_hls.sh. This sets the following variables:
# - $name           Name of current design
# - $source_path    Path to C source files
source $config_path

set work_dir    $output_dir
set project_dir [file join $work_dir vitis]
set ip_path     [file join $work_dir ip ip]

set part_number "xck26-sfvc784-2LV-c"
set top_name    "forward"

# Note: The working dir in Vitis tcl mode will be determined by the location of the tcl file,
#       not the --work_dir param. Make sure to change to folder containing the vitis project
#       using 'cd' inside this script!
cd $project_dir
puts "vitis_generate_hw.tcl: project dir is $project_dir"
puts "vitis_generate_hw.tcl: source path is $source_path"

open_project -reset $project_dir
set source_files [glob -nocomplain -directory $source_path *.c *.cpp]
add_files $source_files
# add_files -cflags "-DUSE_FIXED_POINT" neuron.cpp
#puts "$project_dir/src/neuron.c"
#add_files $project_dir/src/neuron.c
set_top $top_name
open_solution $top_name -flow_target vivado
set_part $part_number
create_clock -period 10 -name default

# Synthesis configuration
#config_interface -m_axi_alignment_byte_size 64
#config_compile -name_max_length 64
#config_schedule -effort medium -relax_ii_for_timing=0

# Run synthesis and export
#cd outputs/neuron_test_2
#puts [pwd]
csynth_design
# This will generate a .zip file containing the IP
export_design -format ip_catalog -output $ip_path
exit
