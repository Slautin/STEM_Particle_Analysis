"""Tiled reader tests on fake Tiled nodes: acquisition layouts of the real microscope and the twin."""
import numpy as np
import pytest

from stem_agent.data import tiled_io


class Leaf:
    def __init__(self, data, metadata=None):
        self.data, self.metadata = np.asarray(data), metadata or {}

    def read(self):
        return self.data


@pytest.fixture
def tiled(monkeypatch):
    catalog = {
        "img": {"image": {"HAADF": Leaf(np.ones((4, 4)))}},
        "eds_real": {"spectrum": Leaf(np.arange(4096).reshape(1, 4096))},
        "eds_one": {"spectrum": Leaf([0.97], {"elements": [b"Au"]})},
        "eds_twin": {"spectrum": Leaf([0.6, 0.4], {"elements": np.array(["Au", "Pt"])})},
        "eds_json": {"spectrum": Leaf([0.6, 0.4], {"elements": '["Au", "Pt"]'})},
        "sam": Leaf([[0, 1], [2, 2]], {"area_statistics": [{"id": 1, "area_px": 1}]}),
        "sam_json": Leaf([[0, 1]], {"area_statistics": '[{"id": 1, "area_px": 1}]'}),
        "sam_none": Leaf([[0, 0]]),
    }
    monkeypatch.setattr(tiled_io, "_client", catalog)
    return catalog


def test_read_image_detector_case(tiled):
    assert tiled_io.read_image("img", "haadf").shape == (4, 4)


def test_read_spectrum_real_channels(tiled):
    values, elements = tiled_io.read_spectrum("eds_real")
    assert values.shape == (4096,) and elements is None


def test_read_spectrum_single_element_stays_1d(tiled):
    values, elements = tiled_io.read_spectrum("eds_one")
    assert values.shape == (1,) and elements == ["Au"]


@pytest.mark.parametrize("key", ["eds_twin", "eds_json"])
def test_read_spectrum_twin_elements(tiled, key):
    values, elements = tiled_io.read_spectrum(key)
    assert values.tolist() == [0.6, 0.4] and elements == ["Au", "Pt"]


@pytest.mark.parametrize("key, areas", [("sam", [{"id": 1, "area_px": 1}]),
                                        ("sam_json", [{"id": 1, "area_px": 1}]),
                                        ("sam_none", [])])
def test_read_labels(tiled, key, areas):
    labels, found = tiled_io.read_labels(key)
    assert labels.dtype.kind in "iu" and found == areas
