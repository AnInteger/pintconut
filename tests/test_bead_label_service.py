import os
import numpy as np
from tests._synth import synth_grid_centers, render_beads
from src.bead_label_service import prelabel, holes_to_boxes, export_yolo, match_cell_colors
from src.color import ColorMatcher
from src.bead_grid import BeadGridFitter


def test_prelabel_full_grid():
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)
    r = prelabel(img)
    assert r.fit_ok is True
    assert r.median_radius > 0
    assert len(r.boxes) >= 20
    assert all(b["source"] == "detect" for b in r.boxes)


def test_prelabel_finds_interior_hole():
    # 5x5 网格，抠掉正中 (r=2,c=2)→图像坐标 (110,110) 的那颗，制造一个内部洞
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    centers = np.array([c for c in centers
                        if not (abs(c[0] - 110.0) < 1 and abs(c[1] - 110.0) < 1)])
    img = render_beads(centers, img_size=(250, 250), bead_radius=10)
    r = prelabel(img)
    assert r.fit_ok is True
    hole_xy = np.array([h["xy"] for h in r.holes])
    assert len(hole_xy) >= 1
    # 正中那个洞应在 (110,110) 附近
    assert np.min(np.linalg.norm(hole_xy - np.array([110.0, 110.0]), axis=1)) < 8.0


def test_prelabel_degraded_when_too_few_beads():
    centers = synth_grid_centers(3, 3, spacing=30.0, origin=(40, 40))  # 9 < MIN_BEADS(20)
    img = render_beads( centers, img_size=(200, 200), bead_radius=10)
    r = prelabel(img)
    assert r.fit_ok is False
    assert r.holes == []
    assert len(r.boxes) >= 5


def test_holes_to_boxes_marks_autofill():
    holes = [{"row": 2, "col": 2, "xy": (110.0, 110.0), "radius": 10.0}]
    boxes = holes_to_boxes(holes)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["source"] == "autofill"
    assert b["cx"] == 110 and b["cy"] == 110
    assert b["width"] == 20 and b["height"] == 20


def test_export_yolo_writes_valid_labels(tmp_path):
    img = render_beads(synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50)),
                       img_size=(250, 250), bead_radius=10)
    boxes = [{"xyxy": [10, 10, 30, 30], "cx": 20, "cy": 20,
              "width": 20, "height": 20, "source": "detect"}]
    images_dir = tmp_path / "images" / "train"
    labels_dir = tmp_path / "labels" / "train"
    img_path, lbl_path, n = export_yolo(img, boxes, "shot1",
                                        str(images_dir), str(labels_dir))
    assert n == 1
    assert os.path.exists(img_path) and os.path.exists(lbl_path)
    parts = open(lbl_path).read().strip().split()
    assert parts[0] == "0"          # 单类别 bead
    assert len(parts) == 5          # class cx cy w h


def test_match_cell_colors_returns_palette_entries():
    centers = synth_grid_centers(5, 5, spacing=30.0, origin=(50, 50))
    img = render_beads(centers, img_size=(250, 250), bead_radius=10,
                       body_color=(0, 0, 255))   # BGR 红
    res = BeadGridFitter().fit(img)
    matched = match_cell_colors(res, ColorMatcher())
    assert len(matched) > 0
    m = matched[0]
    assert {"row", "col", "xy", "name", "rgb"} <= set(m.keys())
    assert isinstance(m["name"], str)


def test_generate_grid_boxes_recovers_perspective_grid():
    from tests._synth import synth_grid_centers, apply_homography
    from src.bead_label_service import generate_grid_boxes
    rows, cols = 10, 10
    centers = synth_grid_centers(rows, cols, spacing=30.0, origin=(40, 40))
    H = np.array([[1.2, 0.1, 10], [0.05, 1.1, 5], [0.0003, 0.0002, 1.0]], dtype=np.float64)
    warped = apply_homography(centers, H)  # ground-truth bead positions under perspective
    # 4 corner beads: TL(0,0), TR(0,cols-1), BR(rows-1,cols-1), BL(rows-1,0)
    corners = np.array([warped[0], warped[cols - 1], warped[rows * cols - 1],
                        warped[(rows - 1) * cols]], dtype=np.float32)
    boxes = generate_grid_boxes(corners, rows, cols)
    assert len(boxes) == rows * cols
    assert all(b["source"] == "generated" for b in boxes)
    # row-major order matches synth_grid_centers → compare centers directly
    gen_xy = np.array([[b["cx"], b["cy"]] for b in boxes])
    err = np.linalg.norm(gen_xy - warped, axis=1)
    assert err.max() < 1.0   # 4-point homography is exact for planar perspective
    # box width ~ local spacing(≈30) * 0.4 * 2 ≈ 24
    widths = [b["width"] for b in boxes]
    assert 10 < np.median(widths) < 40


def test_preview_box_colors_returns_palette_entries():
    from src.bead_label_service import preview_box_colors
    from src.color import ColorMatcher
    img = render_beads(synth_grid_centers(3, 3, spacing=40.0, origin=(40, 40)),
                       img_size=(200, 200), bead_radius=12, body_color=(0, 0, 255))  # BGR 红
    boxes = [{"cx": 40, "cy": 40, "xyxy": [28, 28, 52, 52],
              "width": 24, "height": 24, "source": "generated", "id": 0}]
    out = preview_box_colors(img, boxes, ColorMatcher())
    assert len(out) == 1
    assert {"xy", "name", "rgb"} <= set(out[0].keys())
    assert isinstance(out[0]["name"], str)
