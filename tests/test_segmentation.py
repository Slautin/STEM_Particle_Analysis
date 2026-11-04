"""Segmentation: counts and centres on synthetic particles, filters, aim point, nm settings."""
import numpy as np
import pytest

from stem_agent.analysis.segmentation import find_particles, otsu, pixel_size_m, segment
from stem_agent.analysis.synthetic import blur, particle_image

FOV = 50e-9          # 256 px -> ~0.2 nm per pixel


def disks_image(disks, size=128):
    rows, cols = np.mgrid[0:size, 0:size]
    image = np.full((size, size), 0.05)
    for row, col, radius in disks:
        image[(rows - row) ** 2 + (cols - col) ** 2 <= radius ** 2] = 0.9
    return image


def test_finds_synthetic_particles():
    image, disks = particle_image(size=256, n=12, seed=4)
    labels, particles = segment(blur(image, defocus_px=1.0), FOV)
    assert len(particles) == len(disks)
    for row, col, _ in disks:
        nearest = min(np.hypot(p["centroid"][0] - row, p["centroid"][1] - col) for p in particles)
        assert nearest < 1.5


def test_drops_small_and_edge_particles():
    image = disks_image([(64, 64, 12), (30, 30, 2), (64, 2, 8)])
    pixel = pixel_size_m(FOV, image.shape)
    labels = otsu(image, pixel, smooth_nm=0.1, min_diameter_nm=8 * pixel * 1e9)   # 8 px
    particles = find_particles(labels)
    assert [(round(p["row"]), round(p["col"])) for p in particles] == [(64, 64)]


def test_largest_first_and_cap():
    image = disks_image([(30, 30, 6), (90, 90, 14), (30, 90, 10)])
    _, particles = segment(image, FOV, smooth_nm=0.1, max_particles=2)
    assert [p["id"] for p in particles] == [1, 2]
    assert particles[0]["area_px"] > particles[1]["area_px"]
    assert (round(particles[0]["row"]), round(particles[0]["col"])) == (90, 90)


def test_aim_point_inside_touching_pair():
    image = disks_image([(64, 50, 12), (64, 73, 12)])       # overlap -> one merged object
    labels, particles = segment(image, FOV, smooth_nm=0.1)
    assert len(particles) == 1
    p = particles[0]
    assert labels[round(p["row"]), round(p["col"])] == p["label"]
    assert np.hypot(p["row"] - 64, p["col"] - 50) < 12 or np.hypot(p["row"] - 64, p["col"] - 73) < 12


def test_settings_in_nm_scale_with_pixels():
    disks = [(64, 64, 10)]
    small, large = disks_image(disks, 128), disks_image([(128, 128, 20)], 256)
    _, a = segment(small, FOV, min_diameter_nm=0.5 * 2 * 10 * FOV / 128 * 1e9)
    _, b = segment(large, FOV, min_diameter_nm=0.5 * 2 * 10 * FOV / 128 * 1e9)
    assert len(a) == len(b) == 1


def test_empty_image():
    labels, particles = segment(np.zeros((64, 64)), FOV)
    assert particles == []


@pytest.mark.parametrize("shape, expected", [((128, 128), FOV / 128), ((64, 256), FOV / 256)])
def test_pixel_size(shape, expected):
    assert pixel_size_m(FOV, shape) == pytest.approx(expected)
