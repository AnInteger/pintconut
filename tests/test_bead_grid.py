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
