"""Image metrics for focus and aberration tuning (the search maximizes them).

    contrast(image) -> float                           notebook cell 3
    edge_sharpness(image) -> float                     notebook cell 5
    entropy(image, bins=256) -> float                  notebook cell 7
    high_frequency_power(image, cutoff=0.2) -> float   notebook cell 7
    METRICS: dict[str, Callable]                       name -> function
    all_metrics(image) -> dict[str, float]

Block 2 check on synthetic particle images (notebooks/02_analysis.ipynb):
    contrast, edge_sharpness   fall steadily with blur, also with noise: safe to maximize
    entropy                    RISES with blur (more grey levels): maximizing it defocuses
    high_frequency_power       falls near focus, then rises: noise and the image's non-periodic
                               edges keep a constant high-frequency share while detail fades
"""
from collections.abc import Callable

import numpy as np


def contrast(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    clipped = np.clip(image, low, high)
    return float(np.std(clipped) / (abs(np.mean(clipped)) + 1e-12))


def edge_sharpness(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    clipped = np.clip(image, low, high)
    normalized = clipped / (abs(np.mean(clipped)) + 1e-12)
    gradient_y, gradient_x = np.gradient(normalized)
    return float(np.sqrt(np.mean(gradient_x * gradient_x + gradient_y * gradient_y)))


def entropy(image: np.ndarray, bins: int = 256) -> float:
    image = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        return 0.0
    histogram, _ = np.histogram(np.clip(image, low, high), bins=bins, range=(low, high))
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(bins))


def high_frequency_power(image: np.ndarray, cutoff: float = 0.20) -> float:
    centered = np.asarray(image, dtype=np.float64) - np.mean(image)
    power = np.abs(np.fft.rfft2(centered)) ** 2
    fy = np.fft.fftfreq(centered.shape[0])[:, None]
    fx = np.fft.rfftfreq(centered.shape[1])[None, :]
    high_frequency = np.hypot(fx, fy) >= cutoff
    return float(power[high_frequency].sum() / (power.sum() + 1e-12))


METRICS: dict[str, Callable[[np.ndarray], float]] = {
    "contrast": contrast,
    "edge_sharpness": edge_sharpness,
    "entropy": entropy,
    "high_frequency_power": high_frequency_power,
}


def all_metrics(image: np.ndarray) -> dict[str, float]:
    return {name: fn(image) for name, fn in METRICS.items()}
