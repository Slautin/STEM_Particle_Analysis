"""Find particles in a HAADF image. Settings are in nm, so they work at any FOV and image size.

    pixel_size_m(fov_m, shape) -> float                  the FOV spans the image width
    otsu(image, pixel_m, smooth_nm, min_diameter_nm) -> labels
        notebook cell 21: Gaussian smooth -> Otsu threshold (particles are bright) ->
        fill holes -> drop objects smaller than a disk of min_diameter_nm -> label
    find_particles(labels, edge_px=1, max_particles=None) -> list[Particle]
        drops particles touching the frame edge; largest first; ids 1..n
        row, col = the beam aim point: the pixel deepest inside the particle (always inside,
        also for odd shapes and touching particles); centroid = the centre of mass
    segment(image, fov_m, ...) -> (labels, particles)

Notebook cell 21 used sigma 2 px and min_size 50 px; at its 186 nm / 512 px overview that is
about smooth_nm=0.7 and min_diameter_nm=2.9. The defaults here are smaller, to keep the small
particles of the digital twin (about 3 nm).
Planned (optional extra "sam"): sam(image, checkpoint) from notebook cell 28.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu

SMOOTH_NM = 0.3
MIN_DIAMETER_NM = 1.0


def pixel_size_m(fov_m: float, shape) -> float:
    return float(fov_m) / shape[1]


def otsu(image: np.ndarray, pixel_m: float, smooth_nm: float = SMOOTH_NM,
         min_diameter_nm: float = MIN_DIAMETER_NM) -> np.ndarray:
    smooth = gaussian(np.asarray(image, dtype=np.float64), sigma=smooth_nm * 1e-9 / pixel_m)
    mask = ndi.binary_fill_holes(smooth > threshold_otsu(smooth))
    labels, n = ndi.label(mask)
    if n == 0:
        return labels
    min_area_px = np.pi * (min_diameter_nm * 1e-9 / 2) ** 2 / pixel_m ** 2
    areas = ndi.sum(mask, labels, np.arange(1, n + 1))
    keep = np.flatnonzero(areas >= min_area_px) + 1
    relabel = np.zeros(n + 1, dtype=labels.dtype)
    relabel[keep] = np.arange(1, len(keep) + 1)
    return relabel[labels]


def find_particles(labels: np.ndarray, edge_px: int = 1,
                   max_particles: int | None = None) -> list[dict]:
    n = int(labels.max())
    if n == 0:
        return []
    index = np.arange(1, n + 1)
    mask = labels > 0
    depth = ndi.distance_transform_edt(mask)
    areas = ndi.sum(mask, labels, index)
    centroids = ndi.center_of_mass(mask, labels, index)
    aims = ndi.maximum_position(depth, labels, index)
    height, width = labels.shape

    particles = []
    for label, area, centroid, aim, box in zip(index, areas, centroids, aims, ndi.find_objects(labels)):
        rows, cols = box
        if (rows.start < edge_px or cols.start < edge_px
                or rows.stop > height - edge_px or cols.stop > width - edge_px):
            continue
        particles.append({"label": int(label), "row": float(aim[0]), "col": float(aim[1]),
                          "centroid": [float(centroid[0]), float(centroid[1])],
                          "area_px": int(area)})
    particles.sort(key=lambda p: -p["area_px"])
    if max_particles is not None:
        particles = particles[:max_particles]
    for i, particle in enumerate(particles, start=1):
        particle["id"] = i
    return particles


def segment(image: np.ndarray, fov_m: float, smooth_nm: float = SMOOTH_NM,
            min_diameter_nm: float = MIN_DIAMETER_NM, edge_px: int = 1,
            max_particles: int | None = None) -> tuple[np.ndarray, list[dict]]:
    labels = otsu(image, pixel_size_m(fov_m, image.shape), smooth_nm, min_diameter_nm)
    return labels, find_particles(labels, edge_px, max_particles)
