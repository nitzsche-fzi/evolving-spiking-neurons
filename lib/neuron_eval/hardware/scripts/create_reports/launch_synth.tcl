proc check_if_run_stale {name} {
    set status [get_property STATUS [get_runs $name]]
    set needs_refresh [get_property NEEDS_REFRESH [get_runs $name]]
    # puts $status
    # puts $needs_refresh

    if {$status == "Not started"} {
        return 1
    }

    return $needs_refresh
}

proc launch_synth {top_name synth_run_name} {
    set_property top $top_name [current_fileset]

    # Check if synth is up to date
    if {[check_if_run_stale $synth_run_name] == 0} {
        puts "Synth up to date"
    } else {
        puts "Synth is out of date!"

        reset_run $synth_run_name
        launch_runs $synth_run_name -jobs 20
    }
}

proc launch_synth_and_impl {top_name synth_run_name impl_run_name} {
    set_property top $top_name [current_fileset]

    # Check if synth is up to date
    if {[check_if_run_stale $synth_run_name] == 0} {
        puts "Synth up to date"
    } else {
        puts "Synth is out of date!"

        reset_run $synth_run_name
    }

    # Check if impl is up to date
    if {[check_if_run_stale $impl_run_name] == 0} {
        puts "Impl up to date"
    } else {
        puts "Impl is out of date!"

        reset_run $impl_run_name
        launch_runs $impl_run_name -jobs 20
    }
}