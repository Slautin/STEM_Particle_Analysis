"""OptiSTEM C1/A1 autofocus, run by the microscope itself (AutoScript run_opti_stem, method "C1A1").
SIDE EFFECT: changes focus (C1) and stigmation (A1) through the microscope optics (not the corrector).

Opt-in:  runs only when settings has "auto_focus" ({} = defaults); otherwise skipped, so it can sit
         in any graph without touching runs that do not ask for it
Reads:   settings.auto_focus (config.auto_focus_settings), settings.detector, run.run_dir
Writes:  tuning["auto_focus"] (same fields as a tune step, so save and the summary show it),
         current (fov_m, defocus_m)
MCP:     set_fov / get_fov (if fov_m), get_defocus, acquire_image, auto_focus,
         set_defocus (only to restore)
Steps:   1) FOV (optional)  2) BEFORE: defocus, image, metric  3) auto_focus
         4) wait until the microscope answers again  5) AFTER: defocus, image, metric
         6) clearly worse (after < before x (1 - worse_tolerance)): defocus back to BEFORE
Timeout: MCP gives each Tango command 30 s; OptiSTEM can take longer. The call then returns a
         timeout error while the microscope keeps working, so a timeout is NOT a failure: the node
         polls get_defocus (it fails while the microscope is busy) until it answers, max_wait_s in total.
Restore: defocus back to BEFORE on any error after auto_focus was sent, and when the image got worse.
         A1 cannot be restored: the stigmator is not exposed through MCP.
Twin:    DigitalTwinParticles has no auto_focus (NotImplementedError): the node fails, defocus is
         restored, the graph goes to status_end.
Saves:   auto_focus.json (before / after: defocus, metrics, Tiled key; seconds; verdict),
         auto_focus_before.png, auto_focus_after.png
"""
import asyncio
import json
import time
from pathlib import Path

from stem_agent.analysis.metrics import all_metrics
from stem_agent.config import auto_focus_settings
from stem_agent.mcp import session
from stem_agent.nodes import common
from stem_agent.nodes.set_fov_node import set_fov
from stem_agent.utils import plots
from stem_agent.utils.safety import safe_node

POLL_S = 5.0                     # seconds between "are you free?" calls while OptiSTEM runs


def is_timeout(exc: Exception) -> bool:
    """Tango 'Timeout (30000 mS) exceeded' / API_DeviceTimedOut, or the MCP client's own timeout."""
    text = str(exc).lower()
    return "timeout" in text or "timed out" in text or "timedout" in text


async def run_opti_stem(max_wait_s: float) -> float:
    """auto_focus, waiting past the MCP timeout. -> seconds it took. Any other error is raised."""
    start = time.monotonic()
    try:
        await session.call("auto_focus")
        return time.monotonic() - start
    except Exception as exc:
        if not is_timeout(exc):
            raise                                  # a real error: twin, wrong mode, OptiSTEM failed
        print(f"[auto_focus] still running after the MCP timeout; waiting up to {max_wait_s:.0f} s")
    while time.monotonic() - start < max_wait_s:
        await asyncio.sleep(POLL_S)
        try:
            await session.call("get_defocus")      # answers only once the microscope is free again
            return time.monotonic() - start
        except Exception:
            continue                               # still busy
    raise TimeoutError(f"microscope still busy {max_wait_s:.0f} s after auto_focus")


async def measure(detector: str, metric: str) -> tuple[dict, object]:
    """Defocus + one image + its metrics. -> (record, image)."""
    defocus = float(await session.call("get_defocus"))
    key, image = await common.acquire_haadf(detector)
    metrics = all_metrics(image)
    return {"defocus_m": defocus, "score": metrics[metric], "metrics": metrics, "tiled_key": key}, image


@safe_node()
async def auto_focus_node(state: dict) -> dict:
    settings = state["settings"]
    if "auto_focus" not in settings:
        print("[auto_focus] not in settings, skipped")
        return {}
    cfg = auto_focus_settings(settings)
    metric, detector = cfg["metric"], settings.get("detector", "haadf")
    run_dir = Path(state["run"]["run_dir"])
    current = dict(state.get("current") or {})
    if cfg["fov_m"] is not None:
        current["fov_m"] = await set_fov(cfg["fov_m"])

    before, before_image = await measure(detector, metric)
    try:
        seconds = await run_opti_stem(cfg["max_wait_s"])
        after, after_image = await measure(detector, metric)
    except Exception as exc:
        try:                                       # A1 stays where OptiSTEM left it
            await session.call("set_defocus", before["defocus_m"])
            print(f"[auto_focus] defocus restored after: {exc}")
        except Exception as restore_exc:
            raise RuntimeError(f"{type(exc).__name__}: {exc} | RESTORE FAILED: {restore_exc}") from exc
        raise

    worse = after["score"] < before["score"] * (1 - cfg["worse_tolerance"])
    if worse:                                      # keep the old focus, not a bad OptiSTEM result
        await session.call("set_defocus", before["defocus_m"])
    final, verdict = (before, "worse_restored") if worse else (after, "done")
    change = after["defocus_m"] - before["defocus_m"]

    history_path = run_dir / "auto_focus.json"
    history_path.write_text(json.dumps({"settings": cfg, "seconds": seconds, "before": before,
                                        "after": after, "verdict": verdict}, indent=2), encoding="utf-8")
    before_path, after_path = run_dir / "auto_focus_before.png", run_dir / "auto_focus_after.png"
    plots.save_image(before_image, before_path, f"before auto_focus: {metric} {before['score']:.4g}")
    plots.save_image(after_image, after_path, f"after auto_focus: {metric} {after['score']:.4g}")
    print(f"[auto_focus] {seconds:.0f} s, defocus {change * 1e9:+.1f} nm, {metric} "
          f"{before['score']:.4g} -> {after['score']:.4g}"
          + ("  WORSE: defocus restored (A1 kept)" if worse else ""))

    result = {"name": "auto_focus", "tune": "auto_focus", "fov_m": current.get("fov_m"),
              "run_dir": str(run_dir), "settings": cfg,
              "best_offset_m": [0.0 if worse else change],     # defocus change kept, m
              "best_score": final["score"], "n_images": 2, "stop_reason": verdict,
              "best_image_key": final["tiled_key"], "history_path": str(history_path),
              "figure_path": str(after_path)}
    return {"tuning": {"auto_focus": result}, "current": {**current, "defocus_m": final["defocus_m"]}}
