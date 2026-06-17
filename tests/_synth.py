# tests/_synth.py
"""Synthetic fixtures for bead-grid tests."""
import numpy as np
import cv2


def synth_grid_centers(rows, cols, spacing=20.0, origin=(50.0, 50.0), angle=0.0):
    """Affine grid of (x, y) bead centres. local = (c*spacing, r*spacing), rotated by angle."""
    o = np.array(origin, dtype=np.float64)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    pts = []
    for r in range(rows):
        for c in range(cols):
            local = np.array([c * spacing, r * spacing])
            pts.append(o + R @ local)
    return np.array(pts, dtype=np.float64)


def apply_homography(pts, H):
    pts = np.asarray(pts, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    ph = np.hstack([pts, ones])
    out = (H @ ph.T).T
    return out[:, :2] / out[:, 2:3]


def render_beads(centers, img_size=(600, 600), bead_radius=8, ring_thickness=2,
                 body_color=(60, 60, 180), base_color=(235, 235, 235)):
    """Render double-ring beads (dark ring + coloured body) on a light board base."""
    img = np.full((img_size[0], img_size[1], 3), base_color, dtype=np.uint8)
    for (x, y) in centers:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(img, (cx, cy), bead_radius, (40, 40, 40), ring_thickness)              # dark ring
        inner = max(1, bead_radius - ring_thickness)
        cv2.circle(img, (cx, cy), inner, body_color, -1)                                  # body
    return img


def make_beads(centers, color=None, radius=8.0):
    from src.bead_grid import Bead
    col = np.array(color if color is not None else [60, 60, 180], dtype=np.uint8)
    return [Bead(xy=np.array(c, dtype=np.float64), color=col.copy(), radius=radius)
            for c in centers]
