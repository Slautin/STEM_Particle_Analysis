"""State of one particle-analysis run.

Planned TypedDicts:
    SearchSettings       metric, step, max_iterations, max_offset
    RunSettings          target_xy_m, detector, focus/c1/a1: SearchSettings, tuning_fov_m,
                         segmentation, min_particle_px, max_particles
    RunInfo              run_dir, tiled_uri, instrument_class, tools
    InstrumentSnapshot   stage, fov_m, defocus_m, c1_offset, a1_offset, devices, ready, warnings, raw_path
    TuneResult           best_offset, best_score, metric, n_images, best_image_key, history_path, figure_path
    Overview             tiled_key, fov_m, shape
    Segmentation         method, mask_path, overlay_path, n_particles
    Particle             id, row, col, area_px, beam_xy
    SpectrumRecord       particle_id, tiled_key, npy_path, elements (twin only)

    ParticleRunState     settings, run, start, current, end,
                         tuning (Annotated[dict, operator.or_]),
                         overview, segmentation, particles, spectra, run_status, error

Rules:
    no pixel arrays in state, only Tiled keys and file paths
    nodes return only the keys they change
    start is never changed
"""
