"""Acquire one EDS spectrum at the centre of each particle. SIDE EFFECT: parks the beam.

Reads:   particles
Writes:  spectra
MCP:     place_beam, acquire_spectrum
Checks:  beam_xy within 0..1
Saves:   one .npy per particle + stacked-spectra PNG
Source:  notebook cells 22, 24, 29
"""
