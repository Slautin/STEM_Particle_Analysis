"""Synthetic STEM-like images with controlled blur, for testing without a microscope.

The digital twin's image does not change with z, C1 or A1, so tuning is tested here instead.

    particle_image(size=256, n=14, radius_px=(8, 16), lattice_px=3.0, seed=0) -> (image, disks)
        bright round particles with lattice texture on a dark background (values ~0..1);
        disks = [(row, col, radius_px), ...] is the ground truth for segmentation tests
    blur(image, defocus_px=0.0, astigmatism_px=(0, 0), noise=0.0, rng=None) -> image
        round blur (defocus) + stretched blur (two-fold astigmatism) + Gaussian noise
    SyntheticScope(dims, optimum_m, px_per_m, metric="contrast", noise=0.0, size=256, seed=0)
        score(offset_m): metric of the image blurred by the distance to a hidden optimum;
        dims=1 -> defocus (stands in for stage z or C1), dims=2 -> astigmatism (A1)
"""
import numpy as np
from scipy.signal import fftconvolve

from stem_agent.analysis.metrics import METRICS

PROBE_PX = 0.6          # blur that is always there (finite probe size), in pixels


def particle_image(size: int = 256, n: int = 14, radius_px: tuple[float, float] = (8, 16),
                   lattice_px: float = 3.0, seed: int = 0) -> tuple[np.ndarray, list[tuple]]:
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:size, 0:size].astype(float)
    image = np.full((size, size), 0.05)
    disks: list[tuple[float, float, float]] = []
    for _ in range(50 * n):
        if len(disks) == n:
            break
        radius = rng.uniform(*radius_px)
        row, col = rng.uniform(radius + 2, size - radius - 2, size=2)
        if any(np.hypot(row - r, col - c) < radius + rr + 3 for r, c, rr in disks):
            continue
        disks.append((row, col, radius))
        inside = (rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2
        angle = rng.uniform(0, np.pi)
        u = (cols - col) * np.cos(angle) + (rows - row) * np.sin(angle)
        v = -(cols - col) * np.sin(angle) + (rows - row) * np.cos(angle)
        texture = 0.7 + 0.3 * np.cos(2 * np.pi * u / lattice_px) * np.cos(2 * np.pi * v / lattice_px)
        image[inside] = rng.uniform(0.6, 1.0) * texture[inside]
    return image, disks


def blur(image: np.ndarray, defocus_px: float = 0.0, astigmatism_px: tuple[float, float] = (0.0, 0.0),
         noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Astigmatism (ax, ay): length = extra blur along the long axis, angle/2 = its direction."""
    ax, ay = astigmatism_px
    minor = np.sqrt(PROBE_PX ** 2 + defocus_px ** 2)
    major = np.sqrt(minor ** 2 + ax ** 2 + ay ** 2)
    theta = 0.5 * np.arctan2(ay, ax)
    half = int(np.ceil(4 * major))
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-0.5 * ((u / major) ** 2 + (v / minor) ** 2))
    padded = np.pad(np.asarray(image, dtype=np.float64), half, mode="reflect")
    out = fftconvolve(padded, kernel / kernel.sum(), mode="valid")
    if noise:
        out = out + (rng or np.random.default_rng()).normal(0.0, noise, out.shape)
    return out


class SyntheticScope:
    """Hidden optimum + blur that grows with the distance from it."""

    def __init__(self, dims: int, optimum_m, px_per_m: float, metric: str = "contrast",
                 noise: float = 0.0, size: int = 256, seed: int = 0) -> None:
        self.dims = dims
        self.optimum = np.atleast_1d(np.asarray(optimum_m, dtype=float))
        if self.optimum.shape != (dims,):
            raise ValueError(f"optimum must have {dims} value(s)")
        self.px_per_m = float(px_per_m)
        self.metric = metric
        self.noise = noise
        self.sharp, self.disks = particle_image(size=size, seed=seed)
        self.rng = np.random.default_rng(seed + 1)

    def image(self, offset_m) -> np.ndarray:
        delta = (np.atleast_1d(np.asarray(offset_m, dtype=float)) - self.optimum) * self.px_per_m
        if self.dims == 1:
            return blur(self.sharp, defocus_px=abs(delta[0]), noise=self.noise, rng=self.rng)
        return blur(self.sharp, astigmatism_px=(delta[0], delta[1]), noise=self.noise, rng=self.rng)

    def score(self, offset_m) -> float:
        return METRICS[self.metric](self.image(offset_m))
