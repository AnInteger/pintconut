"""Bead annotation service — YOLO label export (UI-agnostic, testable without Gradio/models).

注：预标/补洞/网格生成/梯度定半径等"自动检测辅助标注"路线已验证无效并移除
（见 CLAUDE.md 项目宪法）。本模块仅保留被 `bead_annotate_mpl.py` 使用的 `export_yolo`。
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from .bead_prelabel import save_yolo_boxes


def export_yolo(image: np.ndarray, boxes: list[dict], name: str,
                images_dir: str, labels_dir: str) -> tuple[str, str, int]:
    """Write image (jpg) + YOLO detection labels (single class 0). Returns (img_path, label_path, n)."""
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    h, w = image.shape[:2]
    img_path = os.path.join(images_dir, name + ".jpg")
    cv2.imwrite(img_path, image)
    label_path = os.path.join(labels_dir, name + ".txt")
    save_yolo_boxes(boxes, w, h, label_path)
    return img_path, label_path, len(boxes)
