"""Segmentation: counts and centres on synthetic particles, filters, aim point, nm settings;
SAM label cleaning."""
import numpy as np
import pytest

from conftest import sam_cluster

from stem_agent.analysis.segmentation import find_particles, otsu, pixel_size_m, sam_particles, segment
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


# --- SAM labels ---

def test_sam_drops_hidden_mask_and_merges_parts():
    labels, particles = sam_particles(*sam_cluster(), pixel_m=1e-9)
    assert np.unique(labels).tolist() == [0, 1, 2]
    assert [p["area_px"] for p in particles] == [380, 380]          # part 4 back inside particle 3


def test_sam_mostly_visible_mask_is_one_particle():
    _, particles = sam_particles(*sam_cluster(), pixel_m=1e-9, min_visible=0.1)
    assert len(particles) == 1                                       # 2 and 3 joined mask 1


def test_sam_touching_particles_aim_at_their_own_middle():
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[10:30, 10:30], labels[10:30, 30:50] = 1, 2
    out, particles = sam_particles(labels, [], pixel_m=1e-9)
    assert len(particles) == 2
    for p in particles:
        assert out[round(p["row"]), round(p["col"])] == p["label"]
        assert abs(p["col"] - 30) > 8                                # not on the shared border


def test_sam_min_diameter_in_nm():
    assert len(sam_particles(*sam_cluster(), pixel_m=1e-9, min_diameter_nm=20)[1]) == 2   # 380 px: 22 nm
    assert sam_particles(*sam_cluster(), pixel_m=1e-9, min_diameter_nm=30)[1] == []
    assert sam_particles(np.zeros((8, 8), dtype=np.uint32), [], pixel_m=1e-9)[1] == []


def test_sam_max_diameter_drops_big_regions():
    big = np.zeros((40, 60), dtype=np.uint32)
    big[2:38, 2:58] = 1                                              # one region of 2016 px: 51 nm
    assert len(sam_particles(big, [], pixel_m=1e-9, edge_px=0)[1]) == 1
    assert sam_particles(big, [], pixel_m=1e-9, edge_px=0, max_diameter_nm=40)[1] == []
    _, kept = sam_particles(*sam_cluster(), pixel_m=1e-9, min_diameter_nm=20, max_diameter_nm=40)
    assert [p["area_px"] for p in kept] == [380, 380]               # 22 nm: inside the window


def test_aim_point_of_an_edge_particle_is_off_the_frame():
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[10:30, 0:20] = 1                                          # cut by the left edge
    [particle] = find_particles(labels, edge_px=0)
    assert 8 <= particle["col"] <= 11                                # middle of what is visible


def test_sam_min_bright_drops_dark_regions():
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[10:30, 5:25], labels[10:30, 35:55] = 1, 2                 # a particle, a gap
    image = np.ones((40, 60))
    image[10:30, 5:25] = 10.0                                        # only region 1 is bright
    assert len(sam_particles(labels, [], pixel_m=1e-9)[1]) == 2
    out, particles = sam_particles(labels, [], pixel_m=1e-9, min_bright=0.5, image=image)
    assert len(particles) == 1 and out[20, 15] == 1 and out[20, 45] == 0
    with pytest.raises(ValueError, match="needs the image"):
        sam_particles(labels, [], pixel_m=1e-9, min_bright=0.5)
