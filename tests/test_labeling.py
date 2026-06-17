# tests/test_labeling.py
import numpy as np
from src.bead_grid import label_affine, label_projective, label_beads
from tests._synth import synth_grid_centers, make_beads, apply_homography


def test_label_affine_clean_grid():
    centers = synth_grid_centers(5, 5, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 20.0
    labels, frac = label_affine(beads, d_row, d_col, spacing)
    assert frac < 0.05                                   # clean affine grid: tiny residual
    assert len(set(labels)) == 25                        # 5x5 unique
    rs = [r for r, _ in labels]; cs = [c for _, c in labels]
    assert min(rs) == 0 and max(rs) == 4
    assert min(cs) == 0 and max(cs) == 4


def test_label_affine_perspective_has_high_residual():
    H = np.array([[1.2, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0008, 0.0, 1.0]])
    centers = apply_homography(synth_grid_centers(8, 8, spacing=25.0, origin=(60, 60)), H)
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 25.0
    _, frac = label_affine(beads, d_row, d_col, spacing)
    assert frac > 0.3                                     # perspective: affine residual is high


def test_label_projective_recovers_perspective_grid():
    H = np.array([[1.2, 0.05, 0.0], [0.03, 1.0, 0.0], [0.0008, 0.0001, 1.0]])
    true_centers = synth_grid_centers(8, 8, spacing=25.0, origin=(80, 80))
    centers = apply_homography(true_centers, H)
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 25.0
    aff_labels, _ = label_affine(beads, d_row, d_col, spacing)
    labels, res_px = label_projective(beads, aff_labels)
    assert res_px < spacing * 0.3                         # tiny reprojection residual
    assert len(set(labels)) == 64                         # 8x8 unique


def test_label_projective_resolves_collisions_to_unique():
    # Stronger perspective forces affine-label collisions; projective relabeling must still
    # recover a unique (row, col) per bead, matching the true grid.
    H = np.array([[1.3, 0.08, 0.0], [0.05, 1.0, 0.0], [0.0012, 0.0002, 1.0]])
    true_centers = synth_grid_centers(8, 8, spacing=25.0, origin=(80, 80))
    centers = apply_homography(true_centers, H)
    beads = make_beads(centers)
    aff_labels, _ = label_affine(beads, np.array([0.0, 1.0]), np.array([1.0, 0.0]), 25.0)
    # sanity: affine stage actually collides (otherwise this test isn't exercising the path)
    assert len(set(aff_labels)) < len(aff_labels), "affine labels should collide here"
    labels, res_px = label_projective(beads, aff_labels)
    # every bead gets a unique label
    assert len(set(labels)) == len(beads)
    # labels cover a plausible 8x8 extent (allow ±1 shift from perspective origin bias)
    rs = [r for r, _ in labels]; cs = [c for _, c in labels]
    assert (max(rs) - min(rs)) >= 7 and (max(cs) - min(cs)) >= 7
    assert len(labels) == 64
    assert res_px < 25.0 * 0.3


def test_label_beads_affine_for_upright():
    centers = synth_grid_centers(6, 6, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 20.0
    labels, persp = label_beads(beads, d_row, d_col, spacing)
    assert persp is False
    assert len(set(labels)) == 36


def test_label_beads_projective_for_perspective():
    H = np.array([[1.2, 0.05, 0.0], [0.03, 1.0, 0.0], [0.0008, 0.0001, 1.0]])
    centers = apply_homography(synth_grid_centers(8, 8, spacing=25.0, origin=(80, 80)), H)
    beads = make_beads(centers)
    d_row = np.array([0.0, 1.0]); d_col = np.array([1.0, 0.0]); spacing = 25.0
    labels, persp = label_beads(beads, d_row, d_col, spacing)
    assert persp is True
    assert len(set(labels)) == 64
