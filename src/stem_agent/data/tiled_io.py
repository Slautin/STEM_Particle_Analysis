"""Read full images and spectra from Tiled by key (HTTP, not MCP).

Done (block 0):
    connect(uri)                               tiled.client.from_uri, reused per URI
    read_image(key, detector="HAADF") -> ndarray

Planned (block 1):
    read_spectrum(key) -> tuple[ndarray, list[str] | None]
                                               real: 4096 channels; twin: one value per element
                                               (element names from dataset attrs)

Source: notebook cell 3 (read_haadf), cell 18 (spectrum read)
"""
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
