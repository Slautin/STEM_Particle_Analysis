"""Image sharpness metrics for focus and aberration tuning.

Planned:
    contrast(image) -> float                       notebook cell 3
    edge_sharpness(image) -> float                 notebook cell 5
    entropy(image, bins=256) -> float              notebook cell 7
    high_frequency_power(image, cutoff=0.2) -> float   notebook cell 7
    METRICS: dict[str, Callable]                   name -> function
    all_metrics(image) -> dict[str, float]
"""
