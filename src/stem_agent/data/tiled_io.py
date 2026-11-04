"""Read full images and spectra from Tiled by key (HTTP, not MCP).

    connect(uri)                               tiled.client.from_uri, reused per URI
    read_image(key, detector="HAADF") -> ndarray
    read_spectrum(key) -> (values, elements | None)
                                               real: channel counts (4096); twin: one value per
                                               element, names in the dataset metadata

Source: notebook cell 3 (read_haadf), cell 18 (spectrum read)
"""
import json

import numpy as np
from tiled.client import from_uri

_client = None
_uri: str | None = None


def connect(uri: str):
    global _client, _uri
    if _client is None or uri != _uri:
        _client, _uri = from_uri(uri), uri
    return _client


def _tiled():
    if _client is None:
        raise RuntimeError("Not connected; call tiled_io.connect(uri) first")
    return _client


def _child(node, name: str):
    """Case-insensitive child lookup: detectors are saved as 'HAADF' but requested as 'haadf'."""
    if not hasattr(node, "keys"):
        return None
    for key in node:
        if key.lower() == name.lower():
            return node[key]
    return None


def read_image(key: str, detector: str = "HAADF") -> np.ndarray:
    """Acquisition layout: <key>/image/<DETECTOR>."""
    node = _tiled()[key]
    for name in ("image", detector):
        child = _child(node, name)
        if child is not None:
            node = child
    return np.asarray(node.read(), dtype=np.float64)


def read_spectrum(key: str) -> tuple[np.ndarray, list[str] | None]:
    """Acquisition layout: <key>/spectrum. Element names (twin) come from its metadata."""
    node = _tiled()[key]
    child = _child(node, "spectrum")
    if child is not None:
        node = child
    values = np.atleast_1d(np.squeeze(np.asarray(node.read(), dtype=np.float64)))   # 1 element stays 1-D
    elements = dict(getattr(node, "metadata", None) or {}).get("elements")
    if elements is None:
        return values, None
    if isinstance(elements, bytes):
        elements = elements.decode()
    if isinstance(elements, str):                    # HDF5 attrs keep lists as JSON text
        elements = json.loads(elements) if elements.startswith("[") else [elements]
    return values, [e.decode() if isinstance(e, bytes) else str(e) for e in elements]
