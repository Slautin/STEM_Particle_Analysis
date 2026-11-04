"""Microscope helpers shared by several nodes. Everything else is called directly in its node.

    get_stage() -> list[float]                 [x, y, z, alpha, beta]; None (NaN over JSON) -> nan
    move_stage(x=None, y=None, z=None) -> list[float]
        only the given axes change, tilt is never touched; refuses steps over LIMITS;
        returns the position read back, fails if any axis is off by more than the tolerance
    acquire_haadf(detector="haadf") -> (tiled_key, image)
"""
import math

import numpy as np

from stem_agent.config import LIMITS
from stem_agent.data import tiled_io
from stem_agent.mcp import session
from stem_agent.mcp.session import InstrumentError


async def get_stage() -> list[float]:
    stage = await session.call("get_stage")
    return [math.nan if v is None else float(v) for v in stage]


async def move_stage(x: float | None = None, y: float | None = None,
                     z: float | None = None) -> list[float]:
    """SIDE EFFECT: moves the stage."""
    current = await get_stage()
    target = list(current)
    for axis, value in enumerate((x, y, z)):
        if value is not None:
            target[axis] = float(value)

    dxy = math.hypot(target[0] - current[0], target[1] - current[1])
    dz = abs(target[2] - current[2])
    if dxy > LIMITS["max_xy_move_m"]:
        raise ValueError(f"x/y move of {dxy * 1e6:.3f} um exceeds "
                         f"{LIMITS['max_xy_move_m'] * 1e6:.0f} um")
    if dz > LIMITS["max_z_step_m"]:
        raise ValueError(f"z step of {dz * 1e6:.3f} um exceeds "
                         f"{LIMITS['max_z_step_m'] * 1e6:.0f} um")

    await session.call("move_stage", target)
    reached = await get_stage()
    error = max(abs(reached[i] - target[i]) for i in range(3))
    if not error <= LIMITS["stage_tolerance_m"]:       # also catches nan
        raise InstrumentError(f"stage landed {error * 1e9:.1f} nm from the target "
                              f"(tolerance {LIMITS['stage_tolerance_m'] * 1e9:.0f} nm)")
    return reached


async def acquire_haadf(detector: str = "haadf") -> tuple[str, np.ndarray]:
    key = await session.call("acquire_image", [detector])
    return key, tiled_io.read_image(key, detector)
