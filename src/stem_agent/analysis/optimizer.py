"""Quadratic trust-region search, shared by stage z, C1 and A1.

Planned:
    class QuadraticSearch(dims, step, max_offset, max_iterations, min_step=None)
        ask() -> ndarray | None        next offset to measure, None when done
        tell(offset, score)            record one measurement
        best() -> SearchPoint          best measured offset and score
        history -> list[SearchPoint]

Algorithm (same numbers as the notebook):
    start points: 0, +-step (1D) or 0 + 8 points on a circle (2D)
    fit a ridge-regularized quadratic to all points
    propose the best predicted unmeasured point within the trust radius
    radius x1.2 (cap 2*step) if score improves > 0.1 %, else x0.65
    stop when radius < min_step (default step/16) or after max_iterations
    every proposal clipped to +-max_offset

Source: notebook cell 5 (fit_response, propose_next_z, hill_climb_stage_z)
        and cell 7 (fit_response_surface, propose_next_point, hill_climb_aberration), merged
"""
