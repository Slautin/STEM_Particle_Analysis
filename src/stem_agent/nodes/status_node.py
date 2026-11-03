"""Is the microscope ready? Read-only snapshot of the instrument.

Planned:
    make_status_node(check_ready: bool) -> node
        status_start = make_status_node(check_ready=True)    writes start, current; stops run if not ready
        status_end   = make_status_node(check_ready=False)   writes end; always runs, also after failure

MCP:     get_parameters, get_stage, get_fov, get_defocus, <Device>_State, <Corrector>_get_info
Checks:  config.READY_CHECKS (start only); never fixes anything itself
Saves:   raw parameters as JSON in the run folder
Source:  new
"""
