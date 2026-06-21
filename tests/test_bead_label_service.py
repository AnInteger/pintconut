import numpy as np
from tests._synth import synth_grid_centers, render_beads
from src.bead_label_service import prelabel


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
