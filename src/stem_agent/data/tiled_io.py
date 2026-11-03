"""Read full images and spectra from Tiled by key.

Planned:
    connect(uri)                               tiled.client.from_uri, cached
    read_image(key, detector="HAADF") -> ndarray
    read_spectrum(key) -> tuple[ndarray, list[str] | None]
                                               real: 4096 channels; twin: one value per element
                                               (element names from dataset attrs)

Source: notebook cell 3 (read_haadf), cell 18 (spectrum read)
"""
