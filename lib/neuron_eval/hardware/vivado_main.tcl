## Imports
set script_path [file normalize [info script]]
set script_dir  [file dirname $script_path]
set root_dir    $script_dir
set reports_flow_dir [file join $root_dir scripts create_reports]

source [file join $reports_flow_dir launch_synth.tcl]
source [file join $reports_flow_dir report_gerneral.tcl]
source [file join $reports_flow_dir report_power.tcl]
source [file join $script_dir vivado_project_flow.tcl]

## Configuration
set run_implementation 0
set save_checkpoints   0
set name         [lindex $argv 0]
set outputs_dir  [lindex $argv 1]
set part_number  "xck26-sfvc784-2LV-c"
set sim_fileset  "sim_1"
set tb_name      "tb_neuron"

## Create Vivado project
::vivado::create_vivado_project $name $part_number $outputs_dir

## Configure synthesis
# Get -tclargs for Vivado synthesis. For example '-mode out_of_context -flatten_hierarchy none -max_dsp 0'
set more_options [lindex $argv 2]
::vivado::configure_synthesis $more_options

## Start synthesis (and implementation)
if {$run_implementation} {
    launch_synth_and_impl $::vivado::top_module $::vivado::synth_run_name $::vivado::impl_run_name
} else {
    launch_synth          $::vivado::top_module $::vivado::synth_run_name
}

## Wait for synthesis to finish and generate reports
wait_on_runs $::vivado::synth_run_name
set report_dir         [file join $outputs_dir reports]
set report_dir_general [file join $report_dir general]
set report_dir_power   [file join $report_dir power]
if {! $run_implementation} {
    report_gerneral_post_synth $report_dir_general $::vivado::synth_run_name
    report_power_post_synth    $report_dir_power   $::vivado::project_path $tb_name $sim_fileset
}
if {$save_checkpoints} { ::vivado::write_checkpoint $::vivado::synth_run_name }

## Wait for implementation to finish and generate reports
if {$run_implementation} {
    wait_on_runs $::vivado::impl_run_name
    report_gerneral_post_impl $report_dir_general $::vivado::impl_run_name
    report_power_post_impl    $report_dir_power   $::vivado::project_path $tb_name $sim_fileset
    if {$save_checkpoints} { ::vivado::write_checkpoint $::vivado::impl_run_name }
}