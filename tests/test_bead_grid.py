import numpy as np
from src.bead_grid import (
    Bead, GridConfidence, TruncationInfo, GridResult, GridFitError,
    AffineMap, ProjectiveMap,
)
from src.grid import CellInfo


def test_bead_dataclass():
    b = Bead(xy=np.array([1.0, 2.0]), color=np.array([10, 20, 30], dtype=np.uint8), radius=5.0)
    assert b.xy.shape == (2,)
    assert b.color.dtype == np.uint8

def test_cellinfo_has_image_xy():
    c = CellInfo(row=0, col=0, color=np.zeros(3, dtype=np.uint8),
                 is_visible=True, is_edge=False, confidence=1.0)
    assert c.image_xy is None  # new field, defaults None

def test_affine_map_to_xy():
    m = AffineMap(origin=np.array([10.0, 20.0]),
                  d_row=np.array([0.0, 1.0]), d_col=np.array([1.0, 0.0]), spacing=5.0)
    # to_xy(r, c) = origin + r*spacing*d_row + c*spacing*d_col
    np.testing.assert_allclose(m.to_xy(2, 3), np.array([10.0 + 3 * 5.0, 20.0 + 2 * 5.0]))

def test_projective_map_to_xy_identity():
    m = ProjectiveMap(H=np.eye(3, dtype=np.float64))
    np.testing.assert_allclose(m.to_xy(2, 3), np.array([2.0, 3.0]))

def test_gridresult_constructs():
    gr = GridResult(rows=2, cols=2, cells=[], outline=None,
                    confidence=GridConfidence(0, 0.0, 0.0, False, "高"),
                    truncation=TruncationInfo(False, []))
    assert gr.rows == 2

def test_gridfiterror_is_exception():
    assert issubclass(GridFitError, Exception)

from tests._synth import synth_grid_centers, apply_homography, render_beads, make_beads

from src.bead_grid import detect_beads


def test_synth_grid_shape_and_spacing():
    pts = synth_grid_centers(rows=3, cols=4, spacing=20.0, origin=(10.0, 10.0))
    assert pts.shape == (12, 2)
    np.testing.assert_allclose(pts[1] - pts[0], [20.0, 0.0], atol=1e-9)   # adjacent column
    np.testing.assert_allclose(pts[4] - pts[0], [0.0, 20.0], atol=1e-9)   # adjacent row

def test_synth_grid_rotation():
    pts = synth_grid_centers(2, 2, spacing=10.0, origin=(0.0, 0.0), angle=np.pi / 2)
    np.testing.assert_allclose(pts[1], [0.0, 10.0], atol=1e-9)

def test_apply_homography_identity():
    pts = synth_grid_centers(2, 2)
    np.testing.assert_allclose(apply_homography(pts, np.eye(3)), pts, atol=1e-9)

def test_render_beads_shape():
    img = render_beads(synth_grid_centers(3, 3), img_size=(200, 200))
    assert img.shape == (200, 200, 3)

def test_make_beads():
    beads = make_beads(synth_grid_centers(2, 2))
    assert len(beads) == 4
    assert beads[0].xy.shape == (2,)


def test_detect_beads_finds_grid():
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(40, 40))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)
    beads = detect_beads(img)
    assert len(beads) >= 20
    detected = np.array([b.xy for b in beads])
    close = sum(1 for t in centers
                if np.min(np.linalg.norm(detected - t, axis=1)) < 6.0)
    assert close >= int(0.8 * len(centers))


def test_detect_beads_empty_image():
    img = np.full((100, 100, 3), 235, dtype=np.uint8)
    assert detect_beads(img) == []

from src.bead_grid import estimate_grid_axes


def test_axes_upright_grid():
    centers = synth_grid_centers(6, 6, spacing=20.0, origin=(30, 30))
    beads = make_beads(centers)
    d_row, d_col, spacing = estimate_grid_axes(beads)
    assert abs(spacing - 20.0) < 2.0
    assert d_row[1] > 0.9        # d_row points down ≈ (0,1)
    assert d_col[0] > 0.9        # d_col points right ≈ (1,0)

def test_axes_rotated_grid():
    centers = synth_grid_centers(6, 6, spacing=20.0, origin=(200, 30), angle=np.radians(30))
    beads = make_beads(centers)
    d_row, d_col, spacing = estimate_grid_axes(beads)
    assert abs(spacing - 20.0) < 2.0
    # row axis tilted ~30° from vertical: d_row ≈ (-sin30, cos30) = (-0.5, 0.866)
    assert abs(d_row[1] - np.cos(np.radians(30))) < 0.15   # downward component ≈ 0.866
    assert abs(d_row[0]) > 0.3                              # tilted (upright grid would give ≈0)
    # d_row still points down, d_col still points right
    assert d_row[1] > 0 and d_col[0] > 0


from src.bead_grid import fit_grid_map, build_cells, AffineMap, TruncationInfo


def test_fit_grid_map_affine_roundtrip():
    centers = synth_grid_centers(4, 4, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    abs_labels = [(r, c) for r in range(4) for c in range(4)]
    gmap = fit_grid_map(beads, abs_labels, False,
                        np.array([0.0, 1.0]), np.array([1.0, 0.0]), 20.0)
    for bead, (r, c) in zip(beads, abs_labels):
        np.testing.assert_allclose(gmap.to_xy(r, c), bead.xy, atol=2.0)


def test_build_cells_filled_and_empty():
    centers = synth_grid_centers(3, 3, spacing=20.0, origin=(40, 40))
    beads = make_beads(centers)
    abs_labels = [(r, c) for r in range(3) for c in range(3)]
    gmap = AffineMap(origin=np.array([40.0, 40.0]),
                     d_row=np.array([0.0, 1.0]), d_col=np.array([1.0, 0.0]), spacing=20.0)
    cells = build_cells(np.full((200, 200, 3), 235, dtype=np.uint8),
                        beads, abs_labels, gmap, 3, 3, TruncationInfo(False, []))
    assert len(cells) == 9
    assert all(c.image_xy is not None for c in cells)
    assert len([c for c in cells if c.is_visible]) == 9
