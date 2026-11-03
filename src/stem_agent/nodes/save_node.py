"""Finish the run: write the summary, optionally blank the beam.

Reads:   whole state
Writes:  run_status ("ok" unless already "failed" / "no_particles")
MCP:     blank_beam (optional)
Saves:   summary.json (full state, JSON-safe)
Source:  new
"""
