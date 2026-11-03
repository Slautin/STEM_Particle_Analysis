"""Full graph on the fake microscope.

Planned:
    happy path: particles, spectra, summary.json, run_status == "ok"
    no particles: EDS skipped, run_status == "no_particles"
    failure during tuning: undo runs, status_end + save still run, run_status == "failed"
    not ready at status_start: nothing moves
"""
