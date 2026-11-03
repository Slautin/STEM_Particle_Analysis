"""Find particles in the overview. Local, no MCP.

Reads:   overview, settings.segmentation, min_particle_px, max_particles
Writes:  segmentation, particles  (run_status="no_particles" if none)
Checks:  drop tiny particles and particles touching the edge; cap the count
Saves:   mask .npy + overlay PNG
Source:  notebook cell 21 (Otsu), cell 28 (SAM)
"""
