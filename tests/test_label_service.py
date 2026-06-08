"""Tests for label service module."""
import json
import os
import tempfile
import numpy as np
from src.label_service import (
    is_rectangular,
    segment_image,
    draw_candidates,
    save_label,
    generate_dataset_yaml,
)


def test_is_rectangular_true():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    ok, ratio = is_rectangular(mask)
    assert ok is True
    assert ratio > 0.9


def test_is_rectangular_false():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:80] = 255
    mask[40:80, 20:40] = 255
    ok, ratio = is_rectangular(mask)
    assert ok is False


def test_is_rectangular_empty():
    mask = np.zeros((100, 100), dtype=np.uint8)
    ok, ratio = is_rectangular(mask)
    assert ok is False


def test_segment_image_returns_candidates():
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    img[50:250, 50:250] = [200, 200, 200]
    candidates = segment_image(img)
    assert isinstance(candidates, list)
    if candidates:
        assert "mask" in candidates[0]
        assert "area_ratio" in candidates[0]
        assert "rect_ratio" in candidates[0]


def test_draw_candidates_returns_image():
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    candidates = [{"mask": np.zeros((200, 200), dtype=np.uint8), "area_ratio": 0.5, "rect_ratio": 0.9}]
    candidates[0]["mask"][50:150, 50:150] = 255
    result = draw_candidates(img, candidates)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_save_label_creates_files():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        lbl_dir = os.path.join(tmpdir, "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)
        import cv2
        img_path = os.path.join(img_dir, "test.png")
        cv2.imwrite(img_path, np.ones((100, 100, 3), dtype=np.uint8) * 255)
        save_label(mask, img_path, img_dir, lbl_dir, 100, 100)
        label_path = os.path.join(lbl_dir, "test.txt")
        assert os.path.exists(label_path)
        with open(label_path) as f:
            content = f.read()
        assert content.startswith("0 ")
        parts = content.strip().split()
        assert len(parts) >= 5


def test_generate_dataset_yaml():
    with tempfile.TemporaryDirectory() as tmpdir:
        for d in ["images/train", "images/valid", "labels/train", "labels/valid"]:
            os.makedirs(os.path.join(tmpdir, d))
        yaml_path = generate_dataset_yaml(tmpdir)
        assert os.path.exists(yaml_path)
        with open(yaml_path) as f:
            content = f.read()
        assert "beadboard" in content
        assert "train" in content
