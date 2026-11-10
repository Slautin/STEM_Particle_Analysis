"""Find particles in a HAADF image. Settings are in nm, so they work at any FOV and image size.

    pixel_size_m(fov_m, shape) -> float                  the FOV spans the image width
    otsu(image, pixel_m, smooth_nm, min_diameter_nm) -> labels
        notebook cell 21: Gaussian smooth -> Otsu threshold (particles are bright) ->
        fill holes -> drop objects smaller than a disk of min_diameter_nm -> label
    find_particles(labels, edge_px=1, max_particles=None) -> list[Particle]
        drops particles touching the frame edge; largest first; ids 1..n
        row, col = the beam aim point: the pixel farthest from the particle's own border (always
        inside, also for odd shapes and touching particles); centroid = the centre of mass
    segment(image, fov_m, ...) -> (labels, particles)
    clean_sam_labels(labels, areas, min_visible) -> labels
        SAM2 masks overlap and smaller ones paint over larger ones in its label image:
        1) a mask with less than min_visible of its area still visible is dropped
           (it mostly covered smaller masks: a cluster of particles, or the background)
        2) a region inside exactly one other region (a part of a particle) joins it
    sam_particles(labels, areas, pixel_m, min_visible, min_diameter_nm, max_diameter_nm, edge_px,
                  min_bright, image) -> (labels, particles)
        clean_sam_labels -> drop dark regions (min_bright: share of pixels above the image's Otsu
        level, as otsu() smooths it) -> drop particles outside min..max_diameter_nm -> find_particles

Notebook cell 21 used sigma 2 px and min_size 50 px; at its 186 nm / 512 px overview that is
about smooth_nm=0.7 and min_diameter_nm=2.9. The defaults here are smaller, to keep the small
particles of the digital twin (about 3 nm).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import find_boundaries

SMOOTH_NM = 0.3
MIN_DIAMETER_NM = 1.0
MIN_VISIBLE = 0.5


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
    inside = np.pad(mask & ~find_boundaries(labels, mode="inner"), 1)   # the frame is a border too
    depth = ndi.distance_transform_edt(inside)[1:-1, 1:-1]
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


def clean_sam_labels(labels: np.ndarray, areas: list[dict],
                     min_visible: float = MIN_VISIBLE) -> np.ndarray:
    """One region per particle from SAM's label image (ids may have gaps afterwards)."""
    labels = np.array(labels, dtype=np.int64)
    for area in areas:                                    # 1) mostly hidden by smaller masks
        own = labels == area["id"]
        if own.sum() < min_visible * area["area_px"]:
            labels[own] = 0
    for k in np.unique(labels)[::-1]:                     # 2) inside one other region (small first)
        if k == 0:
            continue
        ring = ndi.binary_dilation(labels == k) & (labels != k)
        around = np.unique(labels[ring])
        if len(around) == 1 and around[0] != 0:
            labels[labels == k] = around[0]
    return labels


def sam_particles(labels: np.ndarray, areas: list[dict], pixel_m: float,
                  min_visible: float = MIN_VISIBLE, min_diameter_nm: float = 0.0,
                  max_diameter_nm: float | None = None, edge_px: int = 1,
                  min_bright: float = 0.0, image: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    labels = clean_sam_labels(labels, areas, min_visible)
    if min_bright > 0:                                    # dark regions: gaps, support film, holes
        if image is None or image.shape != labels.shape:
            raise ValueError("sam min_bright needs the image the labels belong to")
        smooth = gaussian(np.asarray(image, dtype=np.float64), sigma=SMOOTH_NM * 1e-9 / pixel_m)
        lit = (smooth > threshold_otsu(smooth)).ravel()
        n = int(labels.max()) + 1
        share = np.bincount(labels.ravel(), lit, n) / np.maximum(np.bincount(labels.ravel(), minlength=n), 1)
        labels[share[labels] < min_bright] = 0
    size = np.bincount(labels.ravel())[labels]            # every pixel: the area of its region, px
    disk_px = np.pi * (1e-9 / 2) ** 2 / pixel_m ** 2      # area of a 1 nm diameter disk, px
    if min_diameter_nm > 0:
        labels[size < disk_px * min_diameter_nm ** 2] = 0
    if max_diameter_nm is not None:
        labels[size > disk_px * max_diameter_nm ** 2] = 0
    ids = np.unique(labels[labels > 0])                   # renumber 1..n: find_particles needs no gaps
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    lookup[ids] = np.arange(1, len(ids) + 1)
    labels = lookup[labels]
    return labels, find_particles(labels, edge_px)
