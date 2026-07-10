import os
from tests._synth import synth_grid_centers, render_beads
from src.bead_label_service import export_yolo


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
