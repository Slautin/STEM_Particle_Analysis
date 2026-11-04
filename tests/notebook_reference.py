"""The notebook's original search code (focus_stage_navigation_and_particle_spectroscopy.ipynb,
cells 5 and 7), kept as close to verbatim as possible, to check QuadraticSearch against it.

Changes: OPTIMIZATION_METRIC -> "score"; hardware calls (stage.z, corrector, acquisition)
-> a score function; plotting and printing removed.
"""
import numpy as np

OPTIMIZATION_METRIC = "score"


# ---- cell 5: stage z ----

def fit_response(history):
    z_m = np.asarray([item["stage_z_m"] for item in history])
    scores = np.asarray([item[OPTIMIZATION_METRIC] for item in history])
    center = z_m.mean()
    scale = max(z_m.std(), 1e-12)
    x = (z_m - center) / scale
    design = np.column_stack([np.ones(len(x)), x, x * x])
    ridge = 1e-6 * np.eye(design.shape[1])
    coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ scores)

    def predict(candidate_z_m):
        candidate = np.atleast_1d(candidate_z_m)
        x = (candidate - center) / scale
        return np.column_stack([np.ones(len(x)), x, x * x]) @ coefficients

    return predict


def propose_next_z(history, trust_radius_m, starting_z_m, max_offset_m):
    measured_z = np.asarray([item["stage_z_m"] for item in history])
    scores = np.asarray([item[OPTIMIZATION_METRIC] for item in history])
    best_z = measured_z[int(np.argmax(scores))]
    lower = max(best_z - trust_radius_m, starting_z_m - max_offset_m)
    upper = min(best_z + trust_radius_m, starting_z_m + max_offset_m)
    candidates = np.linspace(lower, upper, 401)
    separation = np.min(np.abs(candidates[:, None] - measured_z[None, :]), axis=1)
    candidates = candidates[separation > max(trust_radius_m / 40, 1e-12)]
    if not len(candidates):
        return None
    predict = fit_response(history)
    return float(candidates[int(np.argmax(predict(candidates)))])


def hill_climb_stage_z(score_at_z, starting_z, step_size_m, max_iterations=8,
                       min_step_size_m=None, max_offset_m=2e-6):
    step = float(step_size_m)
    min_step = float(min_step_size_m if min_step_size_m is not None else step / 16)
    max_offset = float(max_offset_m)
    trust_radius = step
    history = []

    def measure_at(target_z, trial):
        lower = starting_z - max_offset
        upper = starting_z + max_offset
        if not lower <= target_z <= upper:
            raise ValueError("Proposed stage Z exceeds MAX_Z_OFFSET")
        history.append({"score": score_at_z(target_z), "trial": trial, "stage_z_m": float(target_z)})
        return history[-1]

    measure_at(starting_z, "baseline")
    measure_at(starting_z - step, "initial -")
    measure_at(starting_z + step, "initial +")
    for iteration in range(max_iterations):
        previous_best = max(item[OPTIMIZATION_METRIC] for item in history)
        proposal = propose_next_z(history, trust_radius, starting_z, max_offset)
        if proposal is None:
            break
        metrics = measure_at(proposal, f"model {iteration}")
        if metrics["score"] > previous_best * 1.001:
            trust_radius = min(trust_radius * 1.20, step * 2, max_offset)
        else:
            trust_radius *= 0.65
        if trust_radius < min_step:
            break
    return history


# ---- cell 7: aberrations ----

def fit_response_surface(history):
    points = np.asarray([item["offsets_m"] for item in history]) * 1e9
    scores = np.asarray([item[OPTIMIZATION_METRIC] for item in history])
    center = points.mean(axis=0)
    scale = np.maximum(points.std(axis=0), 1e-6)
    normalized = (points - center) / scale
    if points.shape[1] == 2:
        x, y = normalized.T
        design = np.column_stack([np.ones(len(points)), x, y, x*x, x*y, y*y])
    else:
        x = normalized[:, 0]
        design = np.column_stack([np.ones(len(points)), x, x*x])
    ridge = 1e-6 * np.eye(design.shape[1])
    coefficients = np.linalg.solve(design.T @ design + ridge, design.T @ scores)

    def predict(candidate_points_nm):
        candidate = np.atleast_2d(candidate_points_nm)
        z = (candidate - center) / scale
        if points.shape[1] == 2:
            x, y = z.T
            matrix = np.column_stack([np.ones(len(z)), x, y, x*x, x*y, y*y])
        else:
            x = z[:, 0]
            matrix = np.column_stack([np.ones(len(z)), x, x*x])
        return matrix @ coefficients

    return predict


def propose_next_point(history, trust_radius_m):
    points_nm = np.asarray([item["offsets_m"] for item in history]) * 1e9
    scores = np.asarray([item[OPTIMIZATION_METRIC] for item in history])
    best = points_nm[int(np.argmax(scores))]
    radius_nm = trust_radius_m * 1e9
    predict = fit_response_surface(history)
    if points_nm.shape[1] == 2:
        axis = np.linspace(-radius_nm, radius_nm, 81)
        dx, dy = np.meshgrid(axis, axis)
        inside = dx*dx + dy*dy <= radius_nm*radius_nm
        candidates = np.column_stack([best[0] + dx[inside], best[1] + dy[inside]])
    else:
        candidates = np.linspace(best[0] - radius_nm, best[0] + radius_nm, 401)[:, None]
    separation = np.linalg.norm(candidates[:, None, :] - points_nm[None, :, :], axis=2).min(axis=1)
    unmeasured = separation > max(radius_nm / 40, 1e-6)
    candidates = candidates[unmeasured]
    if not len(candidates):
        return None
    return candidates[int(np.argmax(predict(candidates)))] * 1e-9


def hill_climb_aberration(score_at, step_size, components, max_iterations=8, min_step_size=None):
    step = float(step_size)
    min_step = float(min_step_size if min_step_size is not None else step / 16)
    trust_radius = step
    current = np.zeros(components, dtype=float)
    history = []

    def measure_at(target, trial):
        nonlocal current
        current = np.asarray(target, dtype=float).copy()
        history.append({"score": score_at(current), "trial": trial, "offsets_m": current.copy()})
        return history[-1]

    measure_at(current, "baseline")
    if components == 2:
        initial_points = [step * np.array([np.cos(angle), np.sin(angle)])
                          for angle in np.linspace(0, 2*np.pi, 8, endpoint=False)]
    else:
        initial_points = [np.array([-step]), np.array([step])]
    for index, point in enumerate(initial_points):
        measure_at(point, f"initial {index}")

    for iteration in range(max_iterations):
        previous_best = max(item[OPTIMIZATION_METRIC] for item in history)
        proposal = propose_next_point(history, trust_radius)
        if proposal is None:
            break
        metrics = measure_at(proposal, f"model {iteration}")
        if metrics["score"] > previous_best * 1.001:
            trust_radius = min(trust_radius * 1.20, step * 2)
        else:
            trust_radius *= 0.65
        if trust_radius < min_step:
            break
    return history
