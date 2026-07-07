# needs following variables:
# set project_path ""
# set sim_fileset ""
# set tb_name ""
# set output_path "./outputs"

# make sure correct simulaiton set exists in vivado gui

proc report_power_post_synth_old {output_path project_path tb_name sim_fileset} {
    file mkdir $output_path

    launch_simulation -simset [get_filesets $sim_fileset] -mode post-synthesis -type functional

    restart
    open_saif ${tb_name}.saif
    log_saif [get_objects -r /${tb_name}/uut/*]
    run all
    close_saif

    # copy saif to output
    file copy -force "[file rootname $project_path].sim/${sim_fileset}/synth/func/xsim/${tb_name}.saif" "${output_path}/post_synth_saif.saif"

    # can also be specified in a constraints file
    set_switching_activity -default_toggle_rate 0.000
    set_switching_activity -default_static_probability 0.000
    set_units -power mW
    read_saif "[file rootname $project_path].sim/${sim_fileset}/synth/func/xsim/${tb_name}.saif"

    # report_power -file "${output_path}/post_synth_power.rpt" -rpx "${output_path}/post_synth_power.rpx" -l 10
    report_power -file "${output_path}/post_synth_power.rpt" -rpx "${output_path}/post_synth_power.rpx" -l 100 -hier all
}

proc report_power_post_synth {output_path project_path tb_name sim_fileset} {
    report_power_base $output_path $project_path $tb_name $sim_fileset synth
}

proc report_power_post_impl {output_path project_path tb_name sim_fileset} {
    report_power_base $output_path $project_path $tb_name $sim_fileset impl
}

# Mode may either be synth or impl
proc report_power_base {output_path project_path tb_name sim_fileset mode} {
    file mkdir $output_path

    switch $mode {
        synth { set sim_mode post-synthesis }
        impl  { set sim_mode post-implementation }
        default {
            set sim_mode post-synthesis
            puts "report_power.tcl - Unknown mode ${mode}, falling back to: $sim_mode"
            set mode synth
        }
    }
    launch_simulation -simset [get_filesets $sim_fileset] -mode $sim_mode -type functional


    set test_cases {idle in-spike out-spike random-nospike random-spike dummy}
    foreach test $test_cases {
        # Advance to test start
        run all
        # Now we are at the test start
        open_saif ${tb_name}_${test}.saif
        #log_saif [get_objects -r /${tb_name}/uut/design_1_i/neuron_0/*]
        log_saif [get_objects -r /${tb_name}/uut/*]
        # Run until stop command at the end of this test
        run all
        close_saif
    }
    
    # can also be specified in a constraints file
    set_switching_activity -default_toggle_rate 0.000
    set_switching_activity -default_static_probability 0.000
    set_units -power mW
    
    foreach test $test_cases {
        # copy saif to output
        set saif_source_path "[file rootname $project_path].sim/${sim_fileset}/$mode/func/xsim/${tb_name}_${test}.saif"
        file copy -force $saif_source_path "${output_path}/post_${mode}_${test}.saif"

        read_saif $saif_source_path

        #report_power -file "${output_path}/post_${mode}_power-${test}.rpt" -rpx "${output_path}/post_${mode}_${test}_power.rpx" -l 10
        report_power -file "${output_path}/post_${mode}_power-${test}.rpt" -rpx "${output_path}/post_${mode}_power-${test}.rpx" -l 100 -hier all
    }
}