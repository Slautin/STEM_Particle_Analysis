"""Microscope actions used by nodes. Each wraps one or two MCP calls.

Planned (async):
    list_tools() -> list[str]
    get_tiled_uri() -> str                       DATA_get_config
    get_stage() -> list[float]                   [x, y, z, alpha, beta]
    move_stage(position)                         full 5-value vector
    get_fov() / set_fov(m)
    get_defocus()
    acquire_haadf() -> tuple[str, ndarray]       acquire_scanned_image + Tiled read
    correct(name, values)                        correct_aberration "<name> v1 [v2]", relative step
    place_beam(xy)                               fractions 0..1
    acquire_eds() -> tuple[str, ndarray]         acquire_spectrum("eds") + Tiled read
    get_status() -> dict                         get_parameters + device State,
                                                 real and twin keys -> one format
    blank_beam()

Rule: only read-only calls may be retried.
"""
