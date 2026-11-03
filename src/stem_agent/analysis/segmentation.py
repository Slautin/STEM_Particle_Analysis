"""Find particles in a HAADF image.

Planned:
    otsu(image, sigma=2, min_size=50) -> ndarray[int]      label image; notebook cell 21
    sam(image, checkpoint, ...) -> ndarray[int]            optional extra "sam"; notebook cell 28
    particles_from_labels(labels, min_area_px, border_px, max_particles) -> list[Particle]
                                                           centre (row, col), area, beam_xy = [col/w, row/h]
"""
