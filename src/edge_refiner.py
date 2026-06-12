"""
Board edge refinement module.

Refines YOLOv8-seg coarse masks into precise straight-line board edges
using Hough line fitting with multi-layer filtering, truncation handling,
and dual-dimension confidence assessment.
"""
from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DetectionError(Exception):
    """Board detection failed (e.g. YOLO found nothing)."""


class RefinementError(Exception):
    """Edge refinement failed (all candidate lines filtered out)."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EdgeQuality:
    """Per-edge quality metrics (all normalised to [0, 1])."""
    q_fit: float              # Line-fit residual (RMS distance of contour pts to line)
    q_density: float          # Edge-point density
    q_coverage: float         # Line-segment coverage ratio
    q_sharpness: float        # Gradient magnitude along edge
    q_color: float | None     # Board colour match (None = insufficient samples)
    q_texture: float | None   # Grid texture strength via FFT (None = patch too small)


@dataclass
class ClipInfo:
    """Truncation info for one board edge."""
    is_clipped: bool
    clip_side: str | None     # "top" / "bottom" / "left" / "right" (image frame side)
    contour_segment: np.ndarray
    boundary_proximity_ratio: float


@dataclass
class EdgeResult:
    """Detection result for one board edge."""
    edge_id: int              # 0-3, contour arclength segments in traversal order
    line: tuple[float, float, float] | None  # (rho, theta, length) Hough params; None = clipped / failed
    quality: EdgeQuality
    is_clipped: bool
    clip_side: str | None


@dataclass
class BoardConfidence:
    """Dual-dimension confidence."""
    q_object: float           # Is this a bead board?
    q_detection: float        # How accurate are the edges?

    @property
    def total(self) -> float:
        return self.q_object * self.q_detection

    @property
    def level(self) -> str:
        t = self.total
        if t >= 0.8:
            return "高"
        if t >= 0.5:
            return "中"
        return "低"


@dataclass
class BoardDetection:
    """Complete board detection result."""
    corners: np.ndarray              # 4×2 float32  (TL, TR, BR, BL)
    edges: list[EdgeResult]
    confidence: BoardConfidence
    visibility_mask: np.ndarray      # H×W bool – visible region in *original* image coords


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RefinerConfig:
    """All tuneable parameters with sensible defaults."""

    # Truncation detection
    clip_boundary_threshold: int = 10
    clip_segment_ratio: float = 0.70

    # ROI
    roi_dilate_px: int = 20
    roi_erode_px: int = 5

    # Colour pre-filter
    color_sample_band: int = 15
    color_L_min: float = 60.0

    # Hough lines
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 80
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10

    # Contour alignment
    alignment_sample_count: int = 100
    alignment_distance_threshold: float = 5.0
    alignment_min_ratio: float = 0.60

    # Boundary exclusion
    boundary_exclusion_px: int = 5

    # Texture verification
    texture_patch_size: int = 80
    texture_patch_offset: int = 20
    texture_period_tolerance: float = 0.4
    texture_peak_prominence_min: float = 2.0
    texture_peak_prominence_max: float = 10.0

    # Geometry consistency
    geo_angle_min: float = 60.0
    geo_angle_max: float = 120.0
    geo_parallel_tolerance: float = 15.0
    geo_aspect_tolerance: float = 0.4

    # Confidence weights – object
    weight_object_color: float = 0.40
    weight_object_texture: float = 0.60

    # Confidence weights – detection
    weight_det_fit: float = 0.35
    weight_det_density: float = 0.20
    weight_det_coverage: float = 0.15
    weight_det_sharpness: float = 0.15
    weight_det_consistency: float = 0.15


# ---------------------------------------------------------------------------
# EdgeRefiner
# ---------------------------------------------------------------------------

class EdgeRefiner:
    """Refines a YOLOv8-seg coarse mask into precise straight-line edges."""

    def __init__(self, config: RefinerConfig | None = None):
        self.config = config or RefinerConfig()

    # ---- public entry point ------------------------------------------------

    def refine(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        board_size: tuple[int, int],
    ) -> BoardDetection:
        """Run the full refinement pipeline.

        Parameters
        ----------
        mask : H×W uint8 (0/1)
            Binary mask from YOLOv8-seg.
        image : H×W×3 BGR
            Original photograph.
        board_size : (rows, cols)
            Bead grid dimensions.

        Returns
        -------
        BoardDetection
        """
        cfg = self.config
        h, w = mask.shape[:2]
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # Step 1 – truncation detection
        edge_clips = self._detect_truncation(mask, h, w)

        # Step 2 – pre-compute shared data
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = self._compute_roi(mask, h, w)
        all_hough = self._detect_hough_lines(roi, image_gray)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        main_contour = max(contours, key=cv2.contourArea) if contours else None

        # Step 3 – per-edge line fitting (uses shared data)
        edge_results: list[EdgeResult] = []
        for edge_id in range(4):
            clip = edge_clips[edge_id]
            if clip.is_clipped:
                edge_results.append(EdgeResult(
                    edge_id=edge_id,
                    line=None,
                    quality=EdgeQuality(0, 0, 0, 0, None, None),
                    is_clipped=True,
                    clip_side=clip.clip_side,
                ))
            else:
                edge_results.append(self._fit_visible_edge(
                    edge_id, mask, image, image_lab, clip,
                    board_size, h, w,
                    roi, image_gray, all_hough, main_contour,
                ))

        # Step 4 – reconstruct clipped edges
        vis_lines = {e.edge_id: e.line for e in edge_results if e.line is not None and not e.is_clipped}
        if len(vis_lines) >= 2:
            rebuilt = self._reconstruct_clipped_edges(vis_lines, edge_clips, board_size, h, w)
            for eid, line in rebuilt.items():
                edge_results[eid].line = line

        # Step 5 – corner computation
        corners = self._compute_corners(edge_results)

        # Step 6 – confidence assessment
        confidence = self._evaluate_confidence(edge_results, corners, board_size)

        # Step 7 – visibility mask
        visibility = self._build_visibility_mask(corners, mask, h, w)

        return BoardDetection(
            corners=corners,
            edges=edge_results,
            confidence=confidence,
            visibility_mask=visibility,
        )

    # ---- truncation detection ----------------------------------------------

    def _detect_truncation(self, mask: np.ndarray, h: int, w: int) -> list[ClipInfo]:
        cfg = self.config
        # Use CHAIN_APPROX_NONE to keep ALL contour points (needed for
        # boundary-proximity analysis).  CHAIN_APPROX_SIMPLE would compress
        # a rectangle to just 4 corner points, making proximity useless.
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            return [ClipInfo(False, None, np.array([]), 0.0) for _ in range(4)]

        contour = max(contours, key=cv2.contourArea)
        segments = self._split_contour_by_arclength(contour)

        clip_infos: list[ClipInfo] = []
        for seg_pts in segments:
            if len(seg_pts) < 3:
                clip_infos.append(ClipInfo(False, None, seg_pts, 0.0))
                continue

            min_dists = np.minimum(
                np.minimum(seg_pts[:, 0], w - 1 - seg_pts[:, 0]),
                np.minimum(seg_pts[:, 1], h - 1 - seg_pts[:, 1]),
            )
            near = min_dists < cfg.clip_boundary_threshold
            ratio = float(np.mean(near))
            is_clipped = ratio >= cfg.clip_segment_ratio

            clip_side: str | None = None
            if is_clipped:
                bpts = seg_pts[near]
                mx, my = bpts.mean(axis=0)
                dists = {"left": mx, "right": w - 1 - mx, "top": my, "bottom": h - 1 - my}
                clip_side = min(dists, key=dists.get)  # type: ignore[arg-type]

            clip_infos.append(ClipInfo(is_clipped, clip_side, seg_pts, ratio))

        return clip_infos

    @staticmethod
    def _split_contour_by_arclength(contour: np.ndarray) -> list[np.ndarray]:
        pts = contour.reshape(-1, 2).astype(np.float32)
        n = len(pts)

        diffs = np.empty_like(pts)
        diffs[:-1] = np.diff(pts, axis=0)
        diffs[-1] = pts[0] - pts[-1]
        arc = np.sqrt((diffs ** 2).sum(axis=1))
        cum = np.cumsum(arc)
        total = cum[-1]

        split_idx = [np.searchsorted(cum, r * total) % n for r in (0.0, 0.25, 0.5, 0.75, 1.0)]

        segments: list[np.ndarray] = []
        for i in range(4):
            s, e = split_idx[i], split_idx[i + 1]
            segments.append(np.vstack([pts[s:], pts[:e]]) if e <= s else pts[s:e])
        return segments

    # ---- ROI ---------------------------------------------------------------

    def _compute_roi(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        cfg = self.config
        k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.roi_dilate_px * 2 + 1,) * 2)
        k_e = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.roi_erode_px * 2 + 1,) * 2)
        dilated = cv2.dilate(mask, k_d)
        eroded = cv2.erode(mask, k_e)
        roi = cv2.subtract(dilated, eroded)

        bx = cfg.boundary_exclusion_px
        roi[:bx, :] = 0
        roi[-bx:, :] = 0
        roi[:, :bx] = 0
        roi[:, -bx:] = 0
        return roi

    # ---- L2: colour pre-filter ---------------------------------------------

    def _color_prefilter(self, mask: np.ndarray, image_lab: np.ndarray) -> tuple[float, bool]:
        cfg = self.config
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.color_sample_band * 2 + 1,) * 2)
        inner = cv2.erode(mask, k)
        pixels = image_lab[inner > 0]
        if len(pixels) < 50:
            return 0.0, True

        mean_L, mean_a, mean_b = pixels.mean(axis=0)[:3]
        l_s = np.clip((mean_L - cfg.color_L_min) / 30.0, 0, 1)
        a_s = 1.0 - np.clip(abs(mean_a) / 15.0, 0, 1)
        b_s = 1.0 - np.clip(abs(mean_b - 3) / 20.0, 0, 1)
        score = 0.50 * l_s + 0.25 * a_s + 0.25 * b_s
        return float(score), score >= 0.3

    # ---- L3: Hough lines ---------------------------------------------------

    def _detect_hough_lines(self, roi: np.ndarray, image_gray: np.ndarray) -> list[tuple[float, float, float]]:
        cfg = self.config
        masked = image_gray.copy()
        masked[roi == 0] = 0
        edges = cv2.Canny(masked, 50, 150)
        lines = cv2.HoughLinesP(
            edges, cfg.hough_rho, cfg.hough_theta, cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )
        if lines is None:
            return []

        candidates: list[tuple[float, float, float]] = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = np.hypot(x2 - x1, y2 - y1)
            theta = np.arctan2(y2 - y1, x2 - x1)
            rho = x1 * np.cos(theta) + y1 * np.sin(theta)
            candidates.append((rho, theta, length))
        return candidates

    # ---- L3 helper: filter candidates for one edge -------------------------

    @staticmethod
    def _filter_candidates_for_edge(
        all_candidates: list[tuple[float, float, float]],
        contour_segment: np.ndarray,
        angle_tolerance: float = np.pi / 6,
    ) -> list[tuple[float, float, float]]:
        if len(contour_segment) < 2 or not all_candidates:
            return list(all_candidates)
        d = contour_segment[-1].astype(np.float64) - contour_segment[0].astype(np.float64)
        seg_angle = np.arctan2(d[1], d[0])

        filtered: list[tuple[float, float, float]] = []
        for rho, theta, length in all_candidates:
            diff = abs(theta - seg_angle)
            diff = min(diff, np.pi - diff)
            if diff < angle_tolerance:
                filtered.append((rho, theta, length))
        return filtered if filtered else list(all_candidates)

    # ---- L4: contour alignment filter --------------------------------------

    def _filter_by_alignment(
        self,
        candidates: list[tuple[float, float, float]],
        mask_contour: np.ndarray,
    ) -> list[tuple[float, float, float]]:
        from scipy.spatial import cKDTree

        cfg = self.config
        contour_pts = mask_contour.reshape(-1, 2).astype(np.float32)
        tree = cKDTree(contour_pts)

        filtered: list[tuple[float, float, float]] = []
        for rho, theta, length in candidates:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            x0, y0 = rho * cos_t, rho * sin_t
            dx, dy = -sin_t, cos_t
            ts = np.linspace(-length / 2, length / 2, cfg.alignment_sample_count)
            sample_pts = np.column_stack([x0 + ts * dx, y0 + ts * dy])
            distances, _ = tree.query(sample_pts)
            if np.mean(distances < cfg.alignment_distance_threshold) >= cfg.alignment_min_ratio:
                filtered.append((rho, theta, length))
        return filtered

    # ---- L5: geometry validation -------------------------------------------

    def _validate_geometry(self, edge_results: list[EdgeResult], board_size: tuple[int, int]) -> bool:
        cfg = self.config
        visible = [e for e in edge_results if e.line is not None and not e.is_clipped]
        if len(visible) < 2:
            return True

        # Adjacent-edge angle check: only pairs whose segment indices differ by 1 (mod 4)
        for i in range(len(visible)):
            for j in range(i + 1, len(visible)):
                id_diff = abs(visible[i].edge_id - visible[j].edge_id)
                is_adjacent = id_diff == 1 or id_diff == 3  # wrap-around
                if not is_adjacent:
                    continue  # skip opposite-edge pairs
                angle_diff = abs(visible[i].line[1] - visible[j].line[1])
                deg = np.degrees(min(angle_diff, np.pi - angle_diff))
                if deg < cfg.geo_angle_min or deg > cfg.geo_angle_max:
                    return False

        # Convexity check
        if len(visible) >= 3:
            pts_list: list[np.ndarray] = []
            for i in range(len(visible)):
                for j in range(i + 1, len(visible)):
                    pt = self._line_intersection(visible[i].line, visible[j].line)
                    if pt is not None:
                        pts_list.append(pt)
            if len(pts_list) >= 3:
                pts = np.array(pts_list, dtype=np.float32)
                # Order by convex hull
                hull = cv2.convexHull(pts)
                if len(hull) != len(pts_list):
                    return False
        return True

    # ---- L6: texture verification ------------------------------------------

    def _texture_verify(
        self,
        line: tuple[float, float, float],
        image_bgr: np.ndarray,
        board_size: tuple[int, int],
        estimated_edge_length: float,
    ) -> float:
        cfg = self.config
        rho, theta, length = line
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0, y0 = rho * cos_t, rho * sin_t
        normal = np.array([-sin_t, cos_t])

        center = np.array([x0, y0]) + normal * cfg.texture_patch_offset
        h, w = image_bgr.shape[:2]
        half = cfg.texture_patch_size // 2
        cx, cy = int(center[0]), int(center[1])
        y1, y2 = max(0, cy - half), min(h, cy + half)
        x1, x2 = max(0, cx - half), min(w, cx + half)
        patch = image_bgr[y1:y2, x1:x2]
        if patch.shape[0] < 40 or patch.shape[1] < 40:
            return 0.0

        rows, cols = board_size
        est_period = estimated_edge_length / max(rows, cols)
        period_min = est_period * (1 - cfg.texture_period_tolerance)
        period_max = est_period * (1 + cfg.texture_period_tolerance)
        return self._compute_texture_score(patch, period_min, period_max)

    def _compute_texture_score(self, patch: np.ndarray, period_min: float, period_max: float) -> float:
        cfg = self.config
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

        ph, pw = mag.shape
        cy, cx = ph // 2, pw // 2

        r_in = int((1.0 / period_max) * min(ph, pw))
        r_out = int((1.0 / period_min) * min(ph, pw))

        Y, X = np.ogrid[:ph, :pw]
        dist = np.sqrt((X - cx).astype(np.float32) ** 2 + (Y - cy).astype(np.float32) ** 2)
        ring = (dist >= r_in) & (dist <= r_out)
        dc_excl = int(min(ph, pw) * 0.05)
        ring[dist < dc_excl] = False

        vals = mag[ring]
        if len(vals) == 0:
            return 0.0

        prominence = float(mag[ring].max()) / (float(np.median(vals)) + 1e-6)
        return float(np.clip(
            (prominence - cfg.texture_peak_prominence_min)
            / (cfg.texture_peak_prominence_max - cfg.texture_peak_prominence_min),
            0, 1,
        ))

    # ---- single-edge fitting orchestration ----------------------------------

    def _fit_visible_edge(
        self,
        edge_id: int,
        mask: np.ndarray,
        image: np.ndarray,
        image_lab: np.ndarray,
        clip_info: ClipInfo,
        board_size: tuple[int, int],
        h: int, w: int,
        roi: np.ndarray,
        image_gray: np.ndarray,
        all_hough: list[tuple[float, float, float]],
        main_contour: np.ndarray | None,
    ) -> EdgeResult:
        _fail = EdgeResult(edge_id, None, EdgeQuality(0, 0, 0, 0, None, None), False, None)

        # L2 colour
        q_color, passed = self._color_prefilter(mask, image_lab)
        if not passed:
            _fail.quality.q_color = q_color
            return _fail

        # L3 – select candidates matching this edge direction
        candidates = self._filter_candidates_for_edge(all_hough, clip_info.contour_segment)
        if not candidates:
            _fail.quality.q_color = q_color
            return _fail

        # L4 alignment
        if main_contour is None:
            return _fail
        filtered = self._filter_by_alignment(candidates, main_contour)
        if not filtered:
            _fail.quality.q_color = q_color
            return _fail

        # Select best line for this segment
        best = self._select_best_line(filtered, clip_info.contour_segment)
        if best is None:
            return _fail

        # L6 texture
        seg = clip_info.contour_segment
        if len(seg) >= 2:
            d = np.diff(seg, axis=0)
            est_len = float(np.sqrt((d ** 2).sum(axis=1)).sum())
        else:
            est_len = min(h, w) * 0.5
        q_texture = self._texture_verify(best, image, board_size, est_len)

        # Quality
        quality = self._compute_edge_quality(best, seg, image_gray, q_color, q_texture)
        return EdgeResult(edge_id, best, quality, False, None)

    @staticmethod
    def _select_best_line(
        candidates: list[tuple[float, float, float]],
        contour_segment: np.ndarray,
    ) -> tuple[float, float, float] | None:
        if not candidates:
            return None
        if len(contour_segment) < 2:
            return max(candidates, key=lambda c: c[2])

        d = contour_segment[-1].astype(np.float64) - contour_segment[0].astype(np.float64)
        seg_angle = np.arctan2(d[1], d[0])
        max_len = max(c[2] for c in candidates)

        best: tuple[float, float, float] | None = None
        best_score = -1.0
        for rho, theta, length in candidates:
            diff = abs(theta - seg_angle)
            diff = min(diff, np.pi - diff)
            a_s = max(0.0, 1.0 - diff / (np.pi / 4))
            l_s = length / max_len
            s = 0.6 * a_s + 0.4 * l_s
            if s > best_score:
                best_score = s
                best = (rho, theta, length)
        return best

    # ---- edge quality metrics ----------------------------------------------

    def _compute_edge_quality(
        self,
        line: tuple[float, float, float],
        contour_segment: np.ndarray,
        image_gray: np.ndarray,
        q_color: float | None,
        q_texture: float | None,
    ) -> EdgeQuality:
        rho, theta, length = line
        pts = contour_segment.reshape(-1, 2).astype(np.float64)
        if len(pts) == 0:
            return EdgeQuality(0, 0, 0, 0, q_color, q_texture)

        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dists = np.abs(pts[:, 0] * cos_t + pts[:, 1] * sin_t - rho)
        rms = float(np.sqrt(np.mean(dists ** 2)))
        q_fit = float(np.clip(1.0 - rms / 8.0, 0, 1))

        expected = max(length / 2, 10)
        q_density = float(np.clip(len(pts) / expected, 0, 1))

        if len(pts) >= 2:
            d = np.diff(pts, axis=0)
            seg_len = float(np.sqrt((d ** 2).sum(axis=1)).sum())
            q_coverage = float(np.clip(length / max(seg_len, 1), 0, 1))
        else:
            q_coverage = 0.0

        q_sharpness = self._compute_edge_sharpness(line, image_gray)
        return EdgeQuality(q_fit, q_density, q_coverage, q_sharpness, q_color, q_texture)

    @staticmethod
    def _compute_edge_sharpness(line: tuple[float, float, float], image_gray: np.ndarray) -> float:
        rho, theta, length = line
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x0, y0 = rho * cos_t, rho * sin_t
        dx, dy = -sin_t, cos_t

        gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
        gm = np.sqrt(gx ** 2 + gy ** 2)

        n_samples = min(int(length), 200)
        ts = np.linspace(-length / 2, length / 2, n_samples)
        sx = (x0 + ts * dx).astype(int)
        sy = (y0 + ts * dy).astype(int)
        h, w = image_gray.shape
        valid = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        if valid.sum() < 5:
            return 0.0
        return float(np.clip(np.median(gm[sy[valid], sx[valid]]) / 100.0, 0, 1))

    # ---- clipped-edge reconstruction ---------------------------------------

    def _reconstruct_clipped_edges(
        self,
        visible_lines: dict[int, tuple[float, float, float]],
        edge_clips: list[ClipInfo],
        board_size: tuple[int, int],
        h: int, w: int,
    ) -> dict[int, tuple[float, float, float]]:
        aspect = board_size[1] / board_size[0]
        n = len(visible_lines)
        if n == 3:
            return self._reconstruct_1_clipped(visible_lines, edge_clips, aspect, h, w)
        if n == 2:
            ids = list(visible_lines)
            if abs(ids[0] - ids[1]) == 2:
                return self._reconstruct_opposite(visible_lines, edge_clips, aspect, h, w)
            return self._reconstruct_adjacent(visible_lines, edge_clips, aspect, h, w)
        if n == 1:
            return self._reconstruct_3_clipped(visible_lines, edge_clips, aspect, h, w)
        return {}

    def _reconstruct_1_clipped(
        self,
        vis: dict[int, tuple[float, float, float]],
        clips: list[ClipInfo],
        aspect: float, h: int, w: int,
    ) -> dict[int, tuple[float, float, float]]:
        cid = next(i for i in range(4) if i not in vis)
        opp = (cid + 2) % 4
        if opp not in vis:
            return self._reconstruct_from_corners(vis, cid, aspect, h, w)
        opp_rho, opp_theta, opp_len = vis[opp]
        seg = clips[cid].contour_segment
        if len(seg) >= 2:
            mid = seg.mean(axis=0)
            offset = mid[0] * np.cos(opp_theta) + mid[1] * np.sin(opp_theta) - opp_rho
        else:
            offset = min(h, w) * 0.3
        return {cid: (opp_rho + offset, opp_theta, opp_len)}

    def _reconstruct_adjacent(
        self,
        vis: dict[int, tuple[float, float, float]],
        clips: list[ClipInfo],
        aspect: float, h: int, w: int,
    ) -> dict[int, tuple[float, float, float]]:
        result: dict[int, tuple[float, float, float]] = {}
        for cid in range(4):
            if cid in vis:
                continue
            parallel = (cid + 2) % 4
            if parallel in vis:
                ref_rho, ref_theta, ref_len = vis[parallel]
                seg = clips[cid].contour_segment
                if len(seg) >= 2:
                    mid = seg.mean(axis=0)
                    offset = mid[0] * np.cos(ref_theta) + mid[1] * np.sin(ref_theta) - ref_rho
                else:
                    other_ids = [i for i in vis if i != parallel]
                    other_len = vis[other_ids[0]][2] if other_ids else ref_len
                    offset = other_len * (1.0 / aspect if cid in (0, 2) else aspect)
                result[cid] = (ref_rho + offset, ref_theta, ref_len)
        return result

    def _reconstruct_opposite(
        self,
        vis: dict[int, tuple[float, float, float]],
        clips: list[ClipInfo],
        aspect: float, h: int, w: int,
    ) -> dict[int, tuple[float, float, float]]:
        ids = list(vis)
        theta_a = vis[ids[0]][1]
        perp = theta_a + np.pi / 2
        avg_len = (vis[ids[0]][2] + vis[ids[1]][2]) / 2

        result: dict[int, tuple[float, float, float]] = {}
        for cid in range(4):
            if cid in vis:
                continue
            seg = clips[cid].contour_segment
            if len(seg) >= 2:
                mid = seg.mean(axis=0)
                rho = mid[0] * np.cos(perp) + mid[1] * np.sin(perp)
            else:
                rho = min(h, w) * 0.5
            result[cid] = (rho, perp, avg_len)
        return result

    def _reconstruct_3_clipped(
        self,
        vis: dict[int, tuple[float, float, float]],
        clips: list[ClipInfo],
        aspect: float, h: int, w: int,
    ) -> dict[int, tuple[float, float, float]]:
        vid = next(iter(vis))
        rho, theta, length = vis[vid]
        perp = theta + np.pi / 2

        result: dict[int, tuple[float, float, float]] = {}
        for cid in range(4):
            if cid in vis:
                continue
            parallel = (cid + 2) % 4
            seg = clips[cid].contour_segment
            if parallel == vid:
                # opposite edge – same direction
                if len(seg) >= 2:
                    mid = seg.mean(axis=0)
                    new_rho = mid[0] * np.cos(theta) + mid[1] * np.sin(theta)
                else:
                    new_rho = rho + length * (1.0 / aspect if cid in (0, 2) else aspect)
                result[cid] = (new_rho, theta, length)
            else:
                if len(seg) >= 2:
                    mid = seg.mean(axis=0)
                    new_rho = mid[0] * np.cos(perp) + mid[1] * np.sin(perp)
                else:
                    new_rho = length * 0.5
                result[cid] = (new_rho, perp, length)
        return result

    def _reconstruct_from_corners(
        self,
        vis: dict[int, tuple[float, float, float]],
        cid: int,
        aspect: float, h: int, w: int,
    ) -> dict[int, tuple[float, float, float]]:
        ids = list(vis)
        corners: list[np.ndarray] = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pt = self._line_intersection(vis[ids[i]], vis[ids[j]])
                if pt is not None:
                    corners.append(pt)
        if len(corners) < 2:
            return {}
        arr = np.array(corners)
        center = arr.mean(axis=0)
        diffs = arr - center
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        far = arr[np.argmax(dists)]
        d = far - center
        th = np.arctan2(d[1], d[0])
        r = far[0] * np.cos(th) + far[1] * np.sin(th)
        return {cid: (r, th, float(np.max(dists)))}

    # ---- corner computation ------------------------------------------------

    def _compute_corners(self, edge_results: list[EdgeResult]) -> np.ndarray:
        lines = [e.line for e in edge_results]
        corners = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            a, b = lines[i], lines[(i + 1) % 4]
            if a is not None and b is not None:
                pt = self._line_intersection(a, b)
                if pt is not None:
                    corners[i] = pt
        return self._order_corners(corners)

    @staticmethod
    def _order_corners(corners: np.ndarray) -> np.ndarray:
        sums = corners[:, 0] + corners[:, 1]
        diffs = corners[:, 1] - corners[:, 0]
        return np.array([
            corners[np.argmin(sums)],
            corners[np.argmin(diffs)],
            corners[np.argmax(sums)],
            corners[np.argmax(diffs)],
        ], dtype=np.float32)

    @staticmethod
    def _line_intersection(
        line1: tuple[float, float, float],
        line2: tuple[float, float, float],
    ) -> np.ndarray | None:
        r1, t1, _ = line1
        r2, t2, _ = line2
        c1, s1 = np.cos(t1), np.sin(t1)
        c2, s2 = np.cos(t2), np.sin(t2)
        det = c1 * s2 - c2 * s1
        if abs(det) < 1e-6:
            return None
        x = (r1 * s2 - r2 * s1) / det
        y = (r2 * c1 - r1 * c2) / det
        return np.array([x, y], dtype=np.float32)

    # ---- confidence --------------------------------------------------------

    def _evaluate_confidence(
        self,
        edge_results: list[EdgeResult],
        corners: np.ndarray,
        board_size: tuple[int, int],
    ) -> BoardConfidence:
        cfg = self.config
        vis = [e for e in edge_results if not e.is_clipped and e.line is not None]

        # Q_object
        c_scores = [e.quality.q_color for e in vis if e.quality.q_color is not None]
        t_scores = [e.quality.q_texture for e in vis if e.quality.q_texture is not None]
        avg_c = float(np.mean(c_scores)) if c_scores else 0.5
        avg_t = float(np.mean(t_scores)) if t_scores else 0.5
        q_obj = cfg.weight_object_color * avg_c + cfg.weight_object_texture * avg_t

        # Q_detection
        if vis:
            avg_fit = float(np.mean([e.quality.q_fit for e in vis]))
            avg_den = float(np.mean([e.quality.q_density for e in vis]))
            avg_cov = float(np.mean([e.quality.q_coverage for e in vis]))
            avg_shp = float(np.mean([e.quality.q_sharpness for e in vis]))
            q_cons = self._geometry_consistency_score(corners, board_size)
            fits = [e.quality.q_fit for e in vis]
            var_pen = 1.0 - float(np.clip(np.std(fits) * 2, 0, 0.5))
            q_det = (
                cfg.weight_det_fit * avg_fit
                + cfg.weight_det_density * avg_den
                + cfg.weight_det_coverage * avg_cov
                + cfg.weight_det_sharpness * avg_shp
                + cfg.weight_det_consistency * q_cons
            ) * var_pen
        else:
            q_det = 0.1

        return BoardConfidence(
            q_object=float(np.clip(q_obj, 0, 1)),
            q_detection=float(np.clip(q_det, 0, 1)),
        )

    def _geometry_consistency_score(self, corners: np.ndarray, board_size: tuple[int, int]) -> float:
        if np.any(corners == 0):
            return 0.3

        expected = board_size[1] / board_size[0]
        sides = []
        for i in range(4):
            p1, p2 = corners[i], corners[(i + 1) % 4]
            l = float(np.sqrt(((p2 - p1) ** 2).sum()))
            d = (p2 - p1) / max(l, 1e-6)
            sides.append((l, d))

        # Angle
        a_scores = []
        for i in range(4):
            d1, d2 = sides[i][1], sides[(i + 1) % 4][1]
            ang = np.degrees(np.arccos(np.clip(abs(np.dot(d1, d2)), 0, 1)))
            a_scores.append(1.0 - abs(ang - 90) / 60)
        a_s = float(np.clip(np.mean(a_scores), 0, 1))

        # Parallel
        p_scores = [float(abs(np.dot(sides[i][1], sides[i + 2][1]))) for i in range(2)]
        p_s = float(np.clip(np.mean(p_scores), 0, 1))

        # Aspect
        width = (sides[0][0] + sides[2][0]) / 2
        height = (sides[1][0] + sides[3][0]) / 2
        if height > 0:
            err = abs(width / height - expected) / expected
            asp_s = float(np.clip(1.0 - err / 0.4, 0, 1))
        else:
            asp_s = 0.0

        return 0.40 * a_s + 0.35 * p_s + 0.25 * asp_s

    # ---- visibility mask ---------------------------------------------------

    @staticmethod
    def _build_visibility_mask(corners: np.ndarray, mask: np.ndarray, h: int, w: int) -> np.ndarray:
        refined = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(refined, [corners.astype(np.int32)], 1)
        return (refined > 0) & (mask > 0)
