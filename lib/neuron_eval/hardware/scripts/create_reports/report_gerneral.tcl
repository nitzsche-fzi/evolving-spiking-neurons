proc report_gerneral_post_synth {output_path synth_run_name} {
    file mkdir $output_path

    open_run $synth_run_name
    write_checkpoint -force "${output_path}/post_synth_checkpoint"
    write_edif ${output_path}/post_synth_netlist.edf -force
    report_utilization -file ${output_path}/post_synth_util.rpt
    report_utilization -file ${output_path}/post_synth_util-h.rpt -hierarchical -hierarchical_depth 10
    report_drc -file ${output_path}/post_synth_drc.rpt  -rpx ${output_path}/post_synth_drc.rpx
    report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_1  -file ${output_path}/post_synth_timing-summary.rpt  -rpx ${output_path}/post_synth_timing-summary.rpx
}

proc report_gerneral_post_impl {output_path impl_run_name} {
    file mkdir $output_path

    open_run $impl_run_name
    write_checkpoint -force "${output_path}/post_impl_checkpoint"
    write_edif ${output_path}/post_impl_netlist.edf -force
    report_utilization -file ${output_path}/post_impl_util.rpt
    report_utilization -file ${output_path}/post_impl_util-h.rpt -hierarchical -hierarchical_depth 10
    report_drc -file ${output_path}/post_impl_drc.rpt  -rpx ${output_path}/post_impl_drc.rpx
    report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -routable_nets -name timing_1  -file ${output_path}/post_impl_timing-summary.rpt -rpx ${output_path}/post_impl_timing-summary.rpx
}


