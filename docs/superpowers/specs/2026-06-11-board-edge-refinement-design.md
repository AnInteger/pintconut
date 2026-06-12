# 拼豆板边缘精化设计文档

> 日期: 2026-06-11
> 状态: Draft
> 涉及模块: `src/detect.py`, `src/grid.py`, `src/compare.py`, `data/colors.json`

---

## 1. 问题陈述

当前板子检测管线使用 YOLOv8-seg 生成像素级分割掩码，再用 `cv2.approxPolyDP` 从轮廓提取四角点。存在两个核心问题：

1. **边缘锯齿**：板子是规则矩形，但分割掩码的边缘天然呈锯齿状，`approxPolyDP` 拟合的轮廓继承了这些锯齿。
2. **区域不完整**：模型有时无法检出完整的板子区域，导致边缘向内偏移。

此外还有三个实际场景需求：
- 板子可在画面中任意角度旋转
- 较重的透视形变（矩形变成梯形/不规则四边形）
- 板子可能被画面边界截断（部分可见）

**本质问题**：用像素级分割拟合一个本质上是直边矩形（或透视四边形）的物体，模型表达能力和物体几何先验不匹配。

## 2. 解决方案概览

保留 YOLOv8-seg 做**粗检测**，新增**直线拟合后处理**阶段：

```
YOLOv8-seg 粗掩码
      │
      ▼
截断检测（每条边：可见 V / 截断 C）
      │
      ▼
可见边 → 多层过滤直线拟合 → 精确直线
截断边 → 几何重建（可见边约束 + 宽高比）
      │
      ▼
4 角点 → 透视校正 → 网格提取（格子级可见性标记）
```

核心思路：**用"直线"作为板子边缘的几何先验**。即使掩码锯齿或部分缺失，只要能提取出直线段，就能重建精确四边形。

## 3. 详细设计

### 3.1 新增模块 `src/edge_refiner.py`

`edge_refiner.py` 是本次改动的核心新增模块，负责从 YOLOv8-seg 粗掩码中提取精确的板子边缘直线。

#### 3.1.1 顶层接口

```python
@dataclass
class EdgeResult:
    """单条边的检测结果"""
    edge_id: int                    # 0-3, 轮廓弧长均分的 4 段，按顺序相邻（段 i 与段 i±1 相邻）
    line: tuple[float, float, float] | None  # (rho, theta, length) Hough 参数化, None=截断或拟合失败
    quality: EdgeQuality            # 拟合质量详情
    is_clipped: bool                # 是否被画面截断
    clip_side: str | None           # 截断在画面哪侧："top"/"bottom"/"left"/"right"

@dataclass
class EdgeQuality:
    """边的质量评分"""
    q_fit: float          # 直线拟合残差 [0,1]
    q_density: float      # 边缘点密度 [0,1]
    q_coverage: float     # 线段覆盖率 [0,1]
    q_sharpness: float    # 边缘梯度锐度 [0,1]
    q_color: float | None # 板子颜色匹配 [0,1], None=采样不足
    q_texture: float | None  # 网格纹理强度 [0,1], None=采样不足

@dataclass
class BoardDetection:
    """板子检测完整结果"""
    corners: np.ndarray             # 4x2 float32, 四角点坐标
    edges: list[EdgeResult]         # 4 条边的结果
    confidence: BoardConfidence     # 置信度评估
    visibility_mask: np.ndarray     # HxW bool, 可见区域掩码

@dataclass
class BoardConfidence:
    """双维度置信度"""
    q_object: float                 # 对象置信度：这是不是拼豆板？[0,1]
    q_detection: float              # 检测置信度：边缘找得准不准？[0,1]

    @property
    def total(self) -> float:
        return self.q_object * self.q_detection

    @property
    def level(self) -> str:
        t = self.total
        if t >= 0.8: return "高"
        if t >= 0.5: return "中"
        return "低"

@dataclass
class RefinerConfig:
    """可配置参数"""
    # 截断检测
    clip_boundary_threshold: int = 10     # 轮廓点距画面边界多少像素内视为截断
    clip_segment_ratio: float = 0.70      # 连续段中多少比例贴近边界才标记为截断

    # ROI
    roi_dilate_px: int = 20               # 掩码膨胀像素数
    roi_erode_px: int = 5                 # 掩码腐蚀像素数（内侧边界）

    # 颜色初筛
    color_sample_band: int = 15           # 掩码内侧采样带宽度
    color_L_min: float = 60.0             # LAB L 通道最低值
    color_a_range: tuple[float, float] = (-15.0, 15.0)
    color_b_range: tuple[float, float] = (-10.0, 20.0)

    # 霍夫直线
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 80
    hough_min_line_length: int = 50
    hough_max_line_gap: int = 10

    # 轮廓贴合度
    alignment_sample_count: int = 100
    alignment_distance_threshold: float = 5.0
    alignment_min_ratio: float = 0.60

    # 画面边界排除
    boundary_exclusion_px: int = 5

    # 纹理验证
    texture_patch_size: int = 80
    texture_patch_offset: int = 20        # 采样块距候选直线的偏移
    texture_period_tolerance: float = 0.4 # 周期估算容差比例
    texture_peak_prominence_min: float = 2.0
    texture_peak_prominence_max: float = 10.0

    # 几何一致性
    geo_angle_min: float = 60.0           # 相邻边最小夹角
    geo_angle_max: float = 120.0          # 相邻边最大夹角
    geo_parallel_tolerance: float = 15.0  # 对边平行度容差（度）
    geo_aspect_tolerance: float = 0.4     # 宽高比偏移容差

    # 置信度权重
    weight_object_color: float = 0.40
    weight_object_texture: float = 0.60
    weight_det_fit: float = 0.35
    weight_det_density: float = 0.20
    weight_det_coverage: float = 0.15
    weight_det_sharpness: float = 0.15
    weight_det_consistency: float = 0.15
```

#### 3.1.2 主流程

```python
class EdgeRefiner:
    """从 YOLOv8-seg 粗掩码中提取精确的板子边缘。"""

    def __init__(self, config: RefinerConfig | None = None):
        self.config = config or RefinerConfig()

    def refine(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        board_size: tuple[int, int],
    ) -> BoardDetection:
        """
        Args:
            mask: YOLOv8-seg 输出的二值掩码 (HxW, uint8, 0/1)
            image: 原始图像 (HxWx3, BGR)
            board_size: (rows, cols) 板子格数
        Returns:
            BoardDetection 完整检测结果
        """
        cfg = self.config
        h, w = mask.shape[:2]
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # ── Step 1: 截断检测 ──
        edge_clips = self._detect_truncation(mask, h, w)

        # ── Step 2: 预计算共享数据（ROI、灰度图、霍夫候选线） ──
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = self._compute_roi(mask, h, w)
        all_hough_candidates = self._detect_hough_lines(roi, image_gray)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        main_contour = max(contours, key=cv2.contourArea) if contours else None

        # ── Step 3: 可见边直线拟合（使用共享数据） ──
        edge_results: list[EdgeResult] = []
        for edge_id in range(4):
            if edge_clips[edge_id].is_clipped:
                edge_results.append(EdgeResult(
                    edge_id=edge_id,
                    line=None,
                    quality=EdgeQuality(0, 0, 0, 0, None, None),
                    is_clipped=True,
                    clip_side=edge_clips[edge_id].clip_side,
                ))
            else:
                result = self._fit_visible_edge(
                    edge_id, mask, image, image_lab,
                    edge_clips[edge_id], board_size, h, w,
                    roi, image_gray, all_hough_candidates, main_contour,
                )
                edge_results.append(result)

        # ── Step 3: 截断边几何重建 ──
        visible_lines = {e.edge_id: e.line for e in edge_results if not e.is_clipped}
        if len(visible_lines) >= 2:
            reconstructed = self._reconstruct_clipped_edges(
                visible_lines, edge_clips, board_size, h, w,
            )
            for edge_id, line in reconstructed.items():
                edge_results[edge_id].line = line

        # ── Step 4: 角点计算 ──
        corners = self._compute_corners(edge_results)

        # ── Step 5: 置信度评估 ──
        confidence = self._evaluate_confidence(edge_results, corners, board_size)

        # ── Step 6: 可见区域掩码 ──
        visibility_mask = self._build_visibility_mask(corners, mask, h, w)

        return BoardDetection(
            corners=corners,
            edges=edge_results,
            confidence=confidence,
            visibility_mask=visibility_mask,
        )
```

### 3.2 截断检测

判断掩码轮廓的哪些段紧贴画面边界，从而确定板子的哪条边被截断。

#### 3.2.1 数据结构

```python
@dataclass
class ClipInfo:
    """单条边的截断信息"""
    is_clipped: bool
    clip_side: str | None           # "top"/"bottom"/"left"/"right"
    contour_segment: np.ndarray     # 该边对应的轮廓点序列 (Nx1x2)
    boundary_proximity_ratio: float # 贴近边界点的比例
```

#### 3.2.2 算法

```python
def _detect_truncation(
    self, mask: np.ndarray, h: int, w: int,
) -> list[ClipInfo]:
    """
    将掩码外轮廓按弧长均分为 4 段。
    段 0~3 按轮廓遍历顺序相邻，edge_id 仅作为索引标签，
    不对应板子的"上/右/下/左"语义。
    后续角点计算依赖相邻关系（段 i 与段 (i+1)%4 的交点），
    最终通过 _order_corners 统一映射到 TL/TR/BR/BL。
    """
    cfg = self.config
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return [ClipInfo(False, None, np.array([]), 0.0) for _ in range(4)]

    contour = max(contours, key=cv2.contourArea)
    contour_pts = contour.reshape(-1, 2)

    # 将轮廓点均匀分为 4 段（按弧长）
    perimeter = cv2.arcLength(contour, True)
    segment_length = perimeter / 4
    segments = self._split_contour_by_arclength(contour, segment_length)

    # 对每段判断截断
    clip_infos = []
    for seg_pts in segments:
        if len(seg_pts) < 3:
            clip_infos.append(ClipInfo(False, None, seg_pts, 0.0))
            continue

        # 计算每个点距画面四边的最短距离
        min_dists = np.minimum(
            np.minimum(seg_pts[:, 0], w - 1 - seg_pts[:, 0]),
            np.minimum(seg_pts[:, 1], h - 1 - seg_pts[:, 1]),
        )
        near_boundary = min_dists < cfg.clip_boundary_threshold
        proximity_ratio = np.mean(near_boundary)

        is_clipped = proximity_ratio >= cfg.clip_segment_ratio

        # 确定截断在画面哪侧
        clip_side = None
        if is_clipped:
            boundary_pts = seg_pts[near_boundary]
            mean_x, mean_y = boundary_pts.mean(axis=0)
            distances_to_sides = {
                "left": mean_x,
                "right": w - 1 - mean_x,
                "top": mean_y,
                "bottom": h - 1 - mean_y,
            }
            clip_side = min(distances_to_sides, key=distances_to_sides.get)

        clip_infos.append(ClipInfo(
            is_clipped=is_clipped,
            clip_side=clip_side,
            contour_segment=seg_pts,
            boundary_proximity_ratio=proximity_ratio,
        ))

    return clip_infos

def _split_contour_by_arclength(
    self, contour: np.ndarray, segment_length: float,
) -> list[np.ndarray]:
    """将闭合轮廓按弧长均分为 4 段。"""
    pts = contour.reshape(-1, 2).astype(np.float32)
    n = len(pts)

    # 计算累计弧长
    diffs = np.diff(pts, axis=0, append=pts[:1])
    arc_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_arc = np.cumsum(arc_lengths)
    total_arc = cum_arc[-1]

    # 4 等分点
    split_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    split_indices = [
        np.searchsorted(cum_arc, r * total_arc) % n
        for r in split_ratios
    ]

    segments = []
    for i in range(4):
        start = split_indices[i]
        end = split_indices[i + 1]
        if end > start:
            seg = pts[start:end]
        else:
            seg = np.vstack([pts[start:], pts[:end]])
        segments.append(seg)

    return segments
```

### 3.3 可见边直线拟合（6 层过滤）

对每条可见边，依次通过 6 层过滤提取精确直线。

#### 3.3.1 L1: 掩码约束 ROI

```python
def _compute_roi(self, mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """生成环形 ROI：膨胀掩码 - 腐蚀掩码，只保留边缘附近区域。"""
    cfg = self.config
    dilated = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.roi_dilate_px * 2 + 1, cfg.roi_dilate_px * 2 + 1)),
    )
    eroded = cv2.erode(
        mask, cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.roi_erode_px * 2 + 1, cfg.roi_erode_px * 2 + 1)),
    )
    roi = cv2.subtract(dilated, eroded)

    # 叠加画面边界排除区：距画面边缘 ≤ boundary_exclusion_px 的区域置零
    exclusion = np.zeros_like(roi)
    exclusion[:cfg.boundary_exclusion_px, :] = 1
    exclusion[-cfg.boundary_exclusion_px:, :] = 1
    exclusion[:, :cfg.boundary_exclusion_px] = 1
    exclusion[:, -cfg.boundary_exclusion_px:] = 1
    roi[exclusion > 0] = 0

    return roi
```

#### 3.3.2 L2: 颜色初筛

```python
def _color_prefilter(
    self, mask: np.ndarray, image_lab: np.ndarray,
) -> tuple[float, bool]:
    """
    在掩码内侧采样带状区域的颜色，与板子颜色参考比较。

    Returns:
        (score, passed): 颜色匹配分数 [0,1]，是否通过初筛
    """
    cfg = self.config
    inner_mask = cv2.erode(
        mask,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (cfg.color_sample_band * 2 + 1, cfg.color_sample_band * 2 + 1),
        ),
    )
    pixels = image_lab[inner_mask > 0]

    if len(pixels) < 50:
        return 0.0, True  # 像素不足，跳过颜色筛，不阻塞

    mean_L, mean_a, mean_b = pixels.mean(axis=0)[:3]

    l_score = np.clip((mean_L - cfg.color_L_min) / 30.0, 0, 1)
    a_score = 1.0 - np.clip(abs(mean_a) / 15.0, 0, 1)
    b_score = 1.0 - np.clip(abs(mean_b - 3) / 20.0, 0, 1)

    score = 0.50 * l_score + 0.25 * a_score + 0.25 * b_score

    # 排除距画面边缘 <10px 的像素（截断场景）
    # （上面 erode 已自然排除了边缘像素）

    return score, score >= 0.3
```

#### 3.3.3 L3: 霍夫直线检测

```python
def _detect_hough_lines(
    self, roi: np.ndarray, image_gray: np.ndarray,
) -> list[tuple[float, float, float]]:
    """
    在 ROI 约束下执行 Canny + 霍夫直线检测。

    Returns:
        list of (rho, theta, length) — 检测到的候选直线
    """
    cfg = self.config

    # 仅在 ROI 内做边缘检测
    masked_gray = image_gray.copy()
    masked_gray[roi == 0] = 0

    edges = cv2.Canny(masked_gray, 50, 150)

    # 霍夫直线检测
    lines = cv2.HoughLinesP(
        edges,
        rho=cfg.hough_rho,
        theta=cfg.hough_theta,
        threshold=cfg.hough_threshold,
        minLineLength=cfg.hough_min_line_length,
        maxLineGap=cfg.hough_max_line_gap,
    )

    if lines is None:
        return []

    # 转换为 (rho, theta) 参数化直线 + 线段长度
    candidates = []
    for seg in lines.reshape(-1, 4):
        x1, y1, x2, y2 = seg
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        theta = np.arctan2(y2 - y1, x2 - x1)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        candidates.append((rho, theta, length))

    return candidates
```

#### 3.3.4 L4: 轮廓贴合度验证

```python
def _filter_by_alignment(
    self,
    candidates: list[tuple[float, float, float]],
    mask_contour: np.ndarray,
) -> list[tuple[float, float, float]]:
    """
    过滤候选直线：只保留紧贴掩码轮廓的直线。
    """
    cfg = self.config
    contour_pts = mask_contour.reshape(-1, 2).astype(np.float32)

    filtered = []
    for rho, theta, length in candidates:
        # 沿直线均匀采样
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # 直线上一点
        x0, y0 = rho * cos_t, rho * sin_t
        # 直线方向向量
        dx, dy = -sin_t, cos_t

        t_vals = np.linspace(-length / 2, length / 2, cfg.alignment_sample_count)
        sample_pts = np.column_stack([
            x0 + t_vals * dx,
            y0 + t_vals * dy,
        ])

        # 计算每个采样点到轮廓的最短距离
        # 用 KDTree 加速
        from scipy.spatial import cKDTree
        tree = cKDTree(contour_pts)
        distances, _ = tree.query(sample_pts)

        alignment = np.mean(distances < cfg.alignment_distance_threshold)
        if alignment >= cfg.alignment_min_ratio:
            filtered.append((rho, theta, length))

    return filtered
```

#### 3.3.5 L5: 几何一致性校验

在所有可见边都完成 L1-L4 后，统一校验几何一致性。此步骤在 `_fit_visible_edge` 的外层调用中执行（见 3.3.7）。

#### 3.3.6 L6: 纹理验证

```python
def _texture_verify(
    self,
    line: tuple[float, float, float],
    image_bgr: np.ndarray,
    board_size: tuple[int, int],
    estimated_edge_length: float,
) -> float:
    """
    在候选直线内侧采样图像块，用 FFT 检测网格纹理。

    Returns:
        texture_score [0, 1]
    """
    cfg = self.config
    rho, theta, length = line
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 直线上一点 + 法向量（指向内侧）
    x0, y0 = rho * cos_t, rho * sin_t
    normal = np.array([-sin_t, cos_t])  # 法向量（需要确保指向掩码内侧）

    # 采样块中心：沿法向偏移 texture_patch_offset
    patch_center = np.array([x0, y0]) + normal * cfg.texture_patch_offset

    h, w = image_bgr.shape[:2]
    half = cfg.texture_patch_size // 2
    cx, cy = int(patch_center[0]), int(patch_center[1])

    # 提取 patch（边界 clamp）
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    patch = image_bgr[y1:y2, x1:x2]

    if patch.shape[0] < 40 or patch.shape[1] < 40:
        return 0.0  # patch 太小

    # 估算网格周期
    rows, cols = board_size
    est_period = estimated_edge_length / max(rows, cols)
    period_min = est_period * (1 - cfg.texture_period_tolerance)
    period_max = est_period * (1 + cfg.texture_period_tolerance)

    return self._compute_texture_score(patch, period_min, period_max)


def _compute_texture_score(
    self, patch: np.ndarray, period_min: float, period_max: float,
) -> float:
    """
    对图像块做 FFT，在预期频率范围内查找峰值。
    """
    cfg = self.config
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2D FFT
    dft = cv2.dft(np.float32(blurred), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

    ph, pw = magnitude.shape
    center_y, center_x = ph // 2, pw // 2

    # 频率范围
    freq_low = 1.0 / period_max
    freq_high = 1.0 / period_min

    r_inner = int(freq_low * min(ph, pw))
    r_outer = int(freq_high * min(ph, pw))

    # 构建环形掩码
    Y, X = np.ogrid[:ph, :pw]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    ring_mask = ((dist_from_center >= r_inner) & (dist_from_center <= r_outer)).astype(np.float32)

    # 排除直流分量附近
    dc_exclusion = int(min(ph, pw) * 0.05)
    ring_mask[dist_from_center < dc_exclusion] = 0

    ring_mag = magnitude * ring_mask
    ring_values = ring_mag[ring_mask > 0]

    if len(ring_values) == 0:
        return 0.0

    peak_value = ring_mag.max()
    median_value = np.median(ring_values)
    prominence = peak_value / (median_value + 1e-6)

    # 归一化
    score = np.clip(
        (prominence - cfg.texture_peak_prominence_min)
        / (cfg.texture_peak_prominence_max - cfg.texture_peak_prominence_min),
        0, 1,
    )
    return score
```

#### 3.3.7 可见边拟合主函数

```python
def _fit_visible_edge(
    self,
    edge_id: int,
    mask: np.ndarray,
    image: np.ndarray,
    image_lab: np.ndarray,
    clip_info: ClipInfo,
    board_size: tuple[int, int],
    h: int, w: int,
    roi: np.ndarray,                      # 共享: 预计算的 ROI
    image_gray: np.ndarray,               # 共享: 预转换的灰度图
    all_hough_candidates: list,           # 共享: 预检测的所有霍夫候选线
    main_contour: np.ndarray | None,      # 共享: 预提取的主轮廓
) -> EdgeResult:
    """对一条可见边执行 6 层过滤直线拟合。使用预计算的共享数据。"""
    cfg = self.config

    # L2: 颜色初筛
    q_color, color_passed = self._color_prefilter(mask, image_lab)
    if not color_passed:
        return EdgeResult(
            edge_id=edge_id, line=None,
            quality=EdgeQuality(0, 0, 0, 0, q_color, None),
            is_clipped=False, clip_side=None,
        )

    # L3: 从共享的霍夫候选线中筛选与当前边方向匹配的
    candidates = self._filter_candidates_for_edge(
        all_hough_candidates, clip_info.contour_segment,
    )
    if not candidates:
        return EdgeResult(
            edge_id=edge_id, line=None,
            quality=EdgeQuality(0, 0, 0, 0, q_color, None),
            is_clipped=False, clip_side=None,
        )

    # L4: 轮廓贴合度
    if main_contour is None:
        return EdgeResult(
            edge_id=edge_id, line=None,
            quality=EdgeQuality(0, 0, 0, 0, q_color, None),
            is_clipped=False, clip_side=None,
        )
    filtered = self._filter_by_alignment(candidates, main_contour)
    if not filtered:
        return EdgeResult(
            edge_id=edge_id, line=None,
            quality=EdgeQuality(0, 0, 0, 0, q_color, None),
            is_clipped=False, clip_side=None,
        )

    # 按该边对应的轮廓段方向聚类，选出最佳直线
    best_line = self._select_best_line(filtered, clip_info.contour_segment)


def _filter_candidates_for_edge(
    self,
    all_candidates: list[tuple[float, float, float]],
    contour_segment: np.ndarray,
    angle_tolerance: float = np.pi / 6,
) -> list[tuple[float, float, float]]:
    """
    从共享的霍夫候选线中，筛选方向与当前边轮廓段匹配的候选。
    避免每条边都重新运行 Canny + Hough。
    """
    if not contour_segment or len(contour_segment) < 2 or not all_candidates:
        return list(all_candidates)

    seg_start = contour_segment[0].astype(np.float64)
    seg_end = contour_segment[-1].astype(np.float64)
    seg_dir = seg_end - seg_start
    seg_angle = np.arctan2(seg_dir[1], seg_dir[0])

    filtered = []
    for rho, theta, length in all_candidates:
        angle_diff = abs(theta - seg_angle)
        angle_diff = min(angle_diff, np.pi - angle_diff)
        if angle_diff < angle_tolerance:
            filtered.append((rho, theta, length))

    # 如果过滤后为空，放宽条件返回所有候选
    return filtered if filtered else list(all_candidates)
    if best_line is None:
        return EdgeResult(
            edge_id=edge_id, line=None,
            quality=EdgeQuality(0, 0, 0, 0, q_color, None),
            is_clipped=False, clip_side=None,
        )

    # L5: 几何一致性（在所有边都拟合后统一校验，此处先跳过）
    # L6: 纹理验证
    # 估算该边长度（从轮廓段估算）
    seg_pts = clip_info.contour_segment
    if len(seg_pts) >= 2:
        diffs = np.diff(seg_pts, axis=0)
        est_length = np.sqrt((diffs ** 2).sum(axis=1)).sum()
    else:
        est_length = min(h, w) * 0.5

    q_texture = self._texture_verify(best_line, image, board_size, est_length)

    # 计算质量评分
    quality = self._compute_edge_quality(
        best_line, clip_info.contour_segment, image_gray, q_color, q_texture,
    )

    return EdgeResult(
        edge_id=edge_id,
        line=best_line,
        quality=quality,
        is_clipped=False,
        clip_side=None,
    )


def _select_best_line(
    self,
    candidates: list[tuple[float, float, float]],
    contour_segment: np.ndarray,
) -> tuple[float, float, float] | None:
    """
    从候选直线中选出最匹配当前轮廓段的直线。
    策略：选择方向与轮廓段主方向最接近、且最长的直线。
    """
    if not contour_segment or len(contour_segment) < 2:
        # 无轮廓段参考，取最长的
        return max(candidates, key=lambda c: c[2]) if candidates else None

    # 轮廓段主方向
    seg_start = contour_segment[0].astype(np.float64)
    seg_end = contour_segment[-1].astype(np.float64)
    seg_dir = seg_end - seg_start
    seg_angle = np.arctan2(seg_dir[1], seg_dir[0])

    best = None
    best_score = -1
    for rho, theta, length in candidates:
        # 角度差（取最短角距离）
        angle_diff = abs(theta - seg_angle)
        angle_diff = min(angle_diff, np.pi - angle_diff)

        # 角度越接近、长度越长，分数越高
        angle_score = max(0, 1.0 - angle_diff / (np.pi / 4))
        length_score = length / max(c[2] for c in candidates)
        score = 0.6 * angle_score + 0.4 * length_score

        if score > best_score:
            best_score = score
            best = (rho, theta, length)

    return best
```

#### 3.3.8 单边质量评分

```python
def _compute_edge_quality(
    self,
    line: tuple[float, float, float],
    contour_segment: np.ndarray,
    image_gray: np.ndarray,
    q_color: float | None,
    q_texture: float | None,
) -> EdgeQuality:
    """计算单条边的各项质量评分。"""
    rho, theta, length = line
    seg_pts = contour_segment.reshape(-1, 2).astype(np.float64)

    # q_fit: 轮廓点到直线的 RMS 距离
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    distances = np.abs(seg_pts[:, 0] * cos_t + seg_pts[:, 1] * sin_t - rho)
    rms = np.sqrt(np.mean(distances ** 2))
    q_fit = np.clip(1.0 - rms / 8.0, 0, 1)  # ≤2px→1.0, >8px→0.0

    # q_density: 轮廓点数 / 期望点数
    expected_points = max(length / 2, 10)
    q_density = np.clip(len(seg_pts) / expected_points, 0, 1)

    # q_coverage: 检测线段长度 / 轮廓段长度
    if len(seg_pts) >= 2:
        seg_diffs = np.diff(seg_pts, axis=0)
        seg_length = np.sqrt((seg_diffs ** 2).sum(axis=1)).sum()
        q_coverage = np.clip(length / max(seg_length, 1), 0, 1)
    else:
        q_coverage = 0.0

    # q_sharpness: 沿边缘的梯度幅值中位数
    q_sharpness = self._compute_edge_sharpness(line, image_gray)

    return EdgeQuality(
        q_fit=q_fit,
        q_density=q_density,
        q_coverage=q_coverage,
        q_sharpness=q_sharpness,
        q_color=q_color,
        q_texture=q_texture,
    )


def _compute_edge_sharpness(
    self, line: tuple[float, float, float], image_gray: np.ndarray,
) -> float:
    """沿边缘采样图像梯度的中位数。"""
    rho, theta, length = line
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x0, y0 = rho * cos_t, rho * sin_t
    dx, dy = -sin_t, cos_t

    # 计算整幅图的 Sobel 梯度幅值
    grad_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # 沿直线采样梯度值
    t_vals = np.linspace(-length / 2, length / 2, min(int(length), 200))
    sample_pts_x = (x0 + t_vals * dx).astype(int)
    sample_pts_y = (y0 + t_vals * dy).astype(int)

    h, w = image_gray.shape
    valid = (
        (sample_pts_x >= 0) & (sample_pts_x < w) &
        (sample_pts_y >= 0) & (sample_pts_y < h)
    )
    if valid.sum() < 5:
        return 0.0

    grad_samples = grad_mag[sample_pts_y[valid], sample_pts_x[valid]]
    median_grad = np.median(grad_samples)

    # 归一化: 经验值清晰边缘约 50-200
    return np.clip(median_grad / 100.0, 0, 1)
```

### 3.4 几何一致性校验 (L5)

在所有可见边完成 L1-L4 后统一校验：

```python
def _validate_geometry(
    self,
    edge_results: list[EdgeResult],
    board_size: tuple[int, int],
) -> bool:
    """
    校验已拟合的直线是否几何一致。

    Returns:
        True = 一致，False = 存在矛盾
    """
    cfg = self.config
    visible = [e for e in edge_results if e.line is not None and not e.is_clipped]

    if len(visible) < 2:
        return True  # 信息不足，不做校验

    lines = [e.line for e in visible]

    # 检查 1: 相邻边夹角
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            angle_diff = abs(lines[i][1] - lines[j][1])
            angle_deg = np.degrees(min(angle_diff, np.pi - angle_diff))
            # 相邻边夹角应在 60°~120°
            if 10 < angle_deg < 170:  # 排除接近平行的
                if angle_deg < cfg.geo_angle_min or angle_deg > cfg.geo_angle_max:
                    return False

    # 检查 2: 对边平行度（如果有 3+ 条可见边）
    if len(visible) >= 3:
        angles = [e.line[1] for e in visible]
        # 找最近平行对
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                diff = abs(angles[i] - angles[j])
                diff_deg = np.degrees(min(diff, np.pi - diff))
                if diff_deg < cfg.geo_parallel_tolerance:
                    # 找到一对近似平行的线 → 对边
                    pass  # 正常

    # 检查 3: 拓扑一致性（交点是否构成凸四边形）
    if len(visible) >= 3:
        corners = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                pt = self._line_intersection(lines[i], lines[j])
                if pt is not None:
                    corners.append(pt)
        if len(corners) >= 3:
            # 检查凸性
            pts = np.array(corners, dtype=np.float32)
            if not self._is_convex(pts):
                return False

    return True


@staticmethod
def _line_intersection(
    line1: tuple[float, float, float],
    line2: tuple[float, float, float],
) -> np.ndarray | None:
    """计算两条直线的交点。直线参数: (rho, theta, length)。"""
    rho1, theta1, _ = line1
    rho2, theta2, _ = line2

    cos1, sin1 = np.cos(theta1), np.sin(theta1)
    cos2, sin2 = np.cos(theta2), np.sin(theta2)

    det = cos1 * sin2 - cos2 * sin1
    if abs(det) < 1e-6:
        return None  # 平行

    x = (rho1 * sin2 - rho2 * sin1) / det
    y = (rho2 * cos1 - rho1 * cos2) / det
    return np.array([x, y], dtype=np.float32)


@staticmethod
def _is_convex(points: np.ndarray) -> bool:
    """检查点集是否构成凸多边形。"""
    n = len(points)
    if n < 3:
        return False

    signs = []
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        signs.append(cross > 0)

    return all(signs) or not any(signs)
```

### 3.5 截断边几何重建

根据可见边的数量和位置，估算截断边的位置。

```python
def _reconstruct_clipped_edges(
    self,
    visible_lines: dict[int, tuple[float, float, float]],
    edge_clips: list[ClipInfo],
    board_size: tuple[int, int],
    h: int, w: int,
) -> dict[int, tuple[float, float, float]]:
    """
    根据可见边重建截断边的直线方程。

    Args:
        visible_lines: {edge_id: (rho, theta, length)} 可见边的直线
        edge_clips: 4 条边的截断信息
        board_size: (rows, cols)
        h, w: 图像尺寸

    Returns:
        {edge_id: (rho, theta, estimated_length)} 重建的截断边
    """
    rows, cols = board_size
    aspect = cols / rows  # 板子宽高比

    reconstructed = {}
    n_visible = len(visible_lines)
    ids = list(visible_lines.keys())

    if n_visible == 3:
        # T1: 3V+1C → 用平行约束重建第 4 边
        reconstructed = self._reconstruct_1_clipped(visible_lines, edge_clips, aspect, h, w)

    elif n_visible == 2:
        # 判断两条可见边是相邻还是对边
        id0, id1 = ids[0], ids[1]
        if abs(id0 - id1) == 2:
            # T2b: 对边
            reconstructed = self._reconstruct_opposite_clipped(
                visible_lines, edge_clips, aspect, h, w,
            )
        else:
            # T2a: 相邻边
            reconstructed = self._reconstruct_adjacent_clipped(
                visible_lines, edge_clips, aspect, h, w,
            )

    elif n_visible == 1:
        # T3: 1V+3C → 信息严重不足，做粗略估算
        reconstructed = self._reconstruct_3_clipped(
            visible_lines, edge_clips, aspect, h, w,
        )

    return reconstructed


def _reconstruct_1_clipped(
    self,
    visible_lines: dict[int, tuple[float, float, float]],
    edge_clips: list[ClipInfo],
    aspect: float,
    h: int, w: int,
) -> dict[int, tuple[float, float, float]]:
    """
    T1 (3V+1C): 已知 A, B, C 三角，用平行约束重建 D。

    设可见边为 E0, E1, E2 (相邻), 截断边为 E3。
    E3 应平行于 E1 (对边), 且通过 A 和 C 延伸的交点方向。

    简化：找到与截断边平行的可见边（对边），做平行偏移。
    """
    clipped_id = None
    for i in range(4):
        if i not in visible_lines:
            clipped_id = i
            break

    if clipped_id is None:
        return {}

    # 截断边的对边（+2 mod 4）
    opposite_id = (clipped_id + 2) % 4
    if opposite_id not in visible_lines:
        # 无法平行参考，用相邻边交点推算
        return self._reconstruct_from_corners(visible_lines, clipped_id, aspect, h, w)

    # 取对边的方向，做偏移
    opp_rho, opp_theta, opp_length = visible_lines[opposite_id]

    # 从交点估算偏移量
    # 截断边经过的角点 = 两条相邻可见边与截断方向的交点
    # 简化：用掩码的截断边轮廓段估算偏移
    clip_contour = edge_clips[clipped_id].contour_segment
    if len(clip_contour) >= 2:
        contour_mid = clip_contour.mean(axis=0)
        # 截断边到对边的距离
        offset = contour_mid[0] * np.cos(opp_theta) + contour_mid[1] * np.sin(opp_theta) - opp_rho
        new_rho = opp_rho + offset
    else:
        # 从其他交点推算
        new_rho = opp_rho + min(h, w) * 0.3  # 粗略估算

    return {clipped_id: (new_rho, opp_theta, opp_length)}


def _reconstruct_adjacent_clipped(
    self,
    visible_lines: dict[int, tuple[float, float, float]],
    edge_clips: list[ClipInfo],
    aspect: float,
    h: int, w: int,
) -> dict[int, tuple[float, float, float]]:
    """
    T2a (2V 相邻): 已知两条相邻边 → 1 个精确角点。
    利用角点 + 方向 + 宽高比重建两条截断边。
    """
    ids = sorted(visible_lines.keys())
    line_a = visible_lines[ids[0]]
    line_b = visible_lines[ids[1]]

    # 两线交点 = 可见角点
    corner = self._line_intersection(line_a, line_b)
    if corner is None:
        return {}

    # 两条截断边分别平行于 line_a 和 line_b
    # 从掩码范围估算偏移距离
    result = {}
    for clipped_id in range(4):
        if clipped_id in visible_lines:
            continue
        # 该截断边平行于哪条可见边？
        # 边的拓扑: 0-上, 1-右, 2-下, 3-左
        # 对边关系: 0↔2, 1↔3
        parallel_to = (clipped_id + 2) % 4
        if parallel_to in visible_lines:
            ref_rho, ref_theta, ref_len = visible_lines[parallel_to]

            # 从掩码估算偏移
            clip_contour = edge_clips[clipped_id].contour_segment
            if len(clip_contour) >= 2:
                contour_mid = clip_contour.mean(axis=0)
                offset = contour_mid[0] * np.cos(ref_theta) + contour_mid[1] * np.sin(ref_theta) - ref_rho
                new_rho = ref_rho + offset
            else:
                # 从角点和宽高比估算
                # 沿另一条边的方向延伸
                other_id = [i for i in ids if i != parallel_to][0] if len(ids) > 1 else ids[0]
                other_line = visible_lines[other_id]
                other_len = other_line[2]
                est_length = other_len * (1.0 / aspect if clipped_id in [0, 2] else aspect)
                new_rho = ref_rho + est_length

            result[clipped_id] = (new_rho, ref_theta, ref_len)

    return result


def _reconstruct_opposite_clipped(
    self,
    visible_lines: dict[int, tuple[float, float, float]],
    edge_clips: list[ClipInfo],
    aspect: float,
    h: int, w: int,
) -> dict[int, tuple[float, float, float]]:
    """
    T2b (2V 对边): 两条对边可见。
    从掩码范围估算两条截断边的位置。
    精度有限，截断方向上依赖估算。
    """
    ids = sorted(visible_lines.keys())
    line_a = visible_lines[ids[0]]
    line_b = visible_lines[ids[1]]

    # 两条对边的方向
    theta_a = line_a[1]
    # 截断边垂直于此方向
    perp_theta = theta_a + np.pi / 2

    # 从掩码范围估算截断边的 rho
    result = {}
    for clipped_id in range(4):
        if clipped_id in visible_lines:
            continue

        clip_contour = edge_clips[clipped_id].contour_segment
        if len(clip_contour) >= 2:
            contour_mid = clip_contour.mean(axis=0)
            rho = contour_mid[0] * np.cos(perp_theta) + contour_mid[1] * np.sin(perp_theta)
        else:
            rho = min(h, w) * 0.5

        # 估算长度 = 可见边的长度（因为对边可见，长度可参考）
        avg_length = (line_a[2] + line_b[2]) / 2
        result[clipped_id] = (rho, perp_theta, avg_length)

    return result


def _reconstruct_3_clipped(
    self,
    visible_lines: dict[int, tuple[float, float, float]],
    edge_clips: list[ClipInfo],
    aspect: float,
    h: int, w: int,
) -> dict[int, tuple[float, float, float]]:
    """
    T3 (1V+3C): 仅 1 条可见边。
    粗略估算，置信度会很低。
    """
    vis_id = list(visible_lines.keys())[0]
    vis_line = visible_lines[vis_id]
    rho, theta, length = vis_line

    # 垂直方向
    perp_theta = theta + np.pi / 2

    # 估算截断边的位置（非常粗略）
    result = {}
    for clipped_id in range(4):
        if clipped_id in visible_lines:
            continue

        parallel_to = (clipped_id + 2) % 4
        if parallel_to == vis_id:
            # 对边：平行偏移
            offset = length * (1.0 / aspect if clipped_id in [0, 2] else aspect)
            # 方向从掩码范围推算
            clip_contour = edge_clips[clipped_id].contour_segment
            if len(clip_contour) >= 2:
                contour_mid = clip_contour.mean(axis=0)
                new_rho = contour_mid[0] * np.cos(theta) + contour_mid[1] * np.sin(theta)
            else:
                new_rho = rho + offset
            result[clipped_id] = (new_rho, theta, length)
        else:
            # 相邻边：垂直方向
            clip_contour = edge_clips[clipped_id].contour_segment
            if len(clip_contour) >= 2:
                contour_mid = clip_contour.mean(axis=0)
                new_rho = contour_mid[0] * np.cos(perp_theta) + contour_mid[1] * np.sin(perp_theta)
            else:
                new_rho = length * 0.5
            result[clipped_id] = (new_rho, perp_theta, length)

    return result


def _reconstruct_from_corners(
    self,
    visible_lines: dict[int, tuple[float, float, float]],
    clipped_id: int,
    aspect: float,
    h: int, w: int,
) -> dict[int, tuple[float, float, float]]:
    """辅助：从可见边的交点推算截断边。"""
    # 计算所有可见边的交点
    ids = list(visible_lines.keys())
    corners = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pt = self._line_intersection(visible_lines[ids[i]], visible_lines[ids[j]])
            if pt is not None:
                corners.append(pt)

    if len(corners) < 2:
        return {}

    # 从交点分布推算截断边方向
    corners_arr = np.array(corners)
    center = corners_arr.mean(axis=0)

    # 取距离中心最远的角点方向作为估算
    diffs = corners_arr - center
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    farthest = corners_arr[np.argmax(dists)]
    direction = farthest - center

    theta = np.arctan2(direction[1], direction[0])
    rho = farthest[0] * np.cos(theta) + farthest[1] * np.sin(theta)

    return {clipped_id: (rho, theta, np.max(dists))}
```

### 3.6 角点计算

从 4 条边的直线方程（包含可见边和重建边）计算 4 个角点。

```python
def _compute_corners(self, edge_results: list[EdgeResult]) -> np.ndarray:
    """
    从 4 条边的直线方程计算 4 个角点。

    相邻边 (i, i+1) 的交点 = 角点 i。
    即 corner_i = intersection(edge_i, edge_{(i+1)%4})

    Returns:
        4x2 float32: [上左, 上右, 下右, 下左]（按板子坐标系）
    """
    lines = []
    for e in edge_results:
        if e.line is not None:
            lines.append(e.line)
        else:
            lines.append(None)

    corners = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        line_a = lines[i]
        line_b = lines[(i + 1) % 4]

        if line_a is not None and line_b is not None:
            pt = self._line_intersection(line_a, line_b)
            if pt is not None:
                corners[i] = pt
            else:
                corners[i] = [0, 0]  # 平行线无交点
        else:
            corners[i] = [0, 0]  # 缺失线

    # 排序为标准顺序: TL, TR, BR, BL
    corners = self._order_corners(corners)
    return corners


@staticmethod
def _order_corners(corners: np.ndarray) -> np.ndarray:
    """按 TL, TR, BR, BL 排序角点。"""
    sums = corners[:, 0] + corners[:, 1]
    diffs = corners[:, 1] - corners[:, 0]
    return np.array([
        corners[np.argmin(sums)],    # TL
        corners[np.argmin(diffs)],   # TR
        corners[np.argmax(sums)],    # BR
        corners[np.argmax(diffs)],   # BL
    ], dtype=np.float32)
```

### 3.7 置信度评估

```python
def _evaluate_confidence(
    self,
    edge_results: list[EdgeResult],
    corners: np.ndarray,
    board_size: tuple[int, int],
) -> BoardConfidence:
    """计算双维度置信度。"""
    cfg = self.config
    visible_edges = [e for e in edge_results if not e.is_clipped and e.line is not None]

    # ── 对象置信度 Q_obj ──
    color_scores = [e.quality.q_color for e in visible_edges if e.quality.q_color is not None]
    texture_scores = [e.quality.q_texture for e in visible_edges if e.quality.q_texture is not None]

    avg_color = np.mean(color_scores) if color_scores else 0.5  # 无数据给中性分
    avg_texture = np.mean(texture_scores) if texture_scores else 0.5

    q_object = cfg.weight_object_color * avg_color + cfg.weight_object_texture * avg_texture

    # ── 检测置信度 Q_det ──
    if visible_edges:
        avg_fit = np.mean([e.quality.q_fit for e in visible_edges])
        avg_density = np.mean([e.quality.q_density for e in visible_edges])
        avg_coverage = np.mean([e.quality.q_coverage for e in visible_edges])
        avg_sharpness = np.mean([e.quality.q_sharpness for e in visible_edges])

        # 几何一致性评分
        q_consistency = self._geometry_consistency_score(corners, board_size)

        # 质量方差惩罚（防止一条极好带三条极差）
        qualities = [e.quality.q_fit for e in visible_edges]
        variance_penalty = 1.0 - np.clip(np.std(qualities) * 2, 0, 0.5)

        q_detection = (
            cfg.weight_det_fit * avg_fit
            + cfg.weight_det_density * avg_density
            + cfg.weight_det_coverage * avg_coverage
            + cfg.weight_det_sharpness * avg_sharpness
            + cfg.weight_det_consistency * q_consistency
        ) * variance_penalty
    else:
        q_detection = 0.1  # 无可见边，极低置信度

    return BoardConfidence(
        q_object=float(np.clip(q_object, 0, 1)),
        q_detection=float(np.clip(q_detection, 0, 1)),
    )


def _geometry_consistency_score(
    self, corners: np.ndarray, board_size: tuple[int, int],
) -> float:
    """
    评估四边形的几何一致性。
    考虑角度、对边平行度、宽高比。
    """
    if np.any(corners == 0):
        return 0.3  # 有缺失角点

    rows, cols = board_size
    expected_aspect = cols / rows

    # 计算四边
    sides = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        length = np.sqrt(((p2 - p1) ** 2).sum())
        direction = (p2 - p1) / max(length, 1e-6)
        sides.append((length, direction))

    # 角度评分: 相邻边夹角
    angle_scores = []
    for i in range(4):
        d1 = sides[i][1]
        d2 = sides[(i + 1) % 4][1]
        cos_angle = np.clip(np.dot(d1, d2), -1, 1)
        angle = np.degrees(np.arccos(abs(cos_angle)))
        angle_scores.append(1.0 - abs(angle - 90) / 60)  # 偏离90°越多扣分
    angle_score = np.clip(np.mean(angle_scores), 0, 1)

    # 对边平行度评分
    parallel_scores = []
    for i in range(2):
        d1 = sides[i][1]
        d2 = sides[i + 2][1]
        cos_sim = abs(np.dot(d1, d2))
        parallel_scores.append(cos_sim)
    parallel_score = np.clip(np.mean(parallel_scores), 0, 1)

    # 宽高比评分
    width = (sides[0][0] + sides[2][0]) / 2
    height = (sides[1][0] + sides[3][0]) / 2
    if height > 0:
        actual_aspect = width / height
        aspect_error = abs(actual_aspect - expected_aspect) / expected_aspect
        aspect_score = np.clip(1.0 - aspect_error / 0.4, 0, 1)
    else:
        aspect_score = 0.0

    return 0.40 * angle_score + 0.35 * parallel_score + 0.25 * aspect_score
```

### 3.8 可见区域掩码

```python
def _build_visibility_mask(
    self,
    corners: np.ndarray,
    mask: np.ndarray,
    h: int, w: int,
) -> np.ndarray:
    """
    构建格子级可见性掩码。
    基于原始 YOLO 掩码，标记每个像素是否在可见板子区域内。
    """
    # 原始 YOLO 掩码即为可见区域的良好近似
    # 用精化后的四边形裁剪，去除锯齿边缘的噪声
    refined_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(refined_mask, [corners.astype(np.int32)], 1)

    # 可见区域 = 四边形 AND 原始掩码
    # 这去除了四边形外但掩码内的锯齿噪声
    # 也去除了掩码外但四边形内的估算区域
    visibility = (refined_mask > 0) & (mask > 0)

    return visibility
```

### 3.9 对现有模块的改动

#### 3.9.1 `src/detect.py` — BoardDetector

新增 `refine()` 方法，保留原有 `detect()` 和 `extract_corners()` 不变（向后兼容）：

```python
# detect.py 新增内容

class BoardDetector:
    # ... 保留现有代码 ...

    def refine(
        self,
        image: np.ndarray,
        board_size: tuple[int, int],
        config: RefinerConfig | None = None,
    ) -> BoardDetection:
        """
        完整的板子检测 + 边缘精化。

        替代 detect() + extract_corners() 的新接口，
        返回包含精化角点、置信度和可见性的完整结果。
        """
        mask = self.detect(image)
        if mask is None:
            raise DetectionError("YOLOv8-seg 未检测到板子")

        refiner = EdgeRefiner(config)
        return refiner.refine(mask, image, board_size)
```

#### 3.9.2 `src/grid.py` — PerspectiveCorrector + GridExtractor

**PerspectiveCorrector 新增 `correct_with_matrix()` 方法**，返回校正后图像和变换矩阵（供可见性掩码使用）：

```python
class PerspectiveCorrector:
    # ... 保留现有 correct() 方法 ...

    def correct_with_matrix(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        output_size: tuple[int, int] = (400, 400),
    ) -> tuple[np.ndarray, np.ndarray]:
        """透视校正，同时返回变换矩阵。

        Returns:
            (corrected_image, transform_matrix)
        """
        dst = np.array([
            [0, 0],
            [output_size[0], 0],
            [output_size[0], output_size[1]],
            [0, output_size[1]],
        ], dtype=np.float32)

        src = corners.astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(
            image, matrix, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return corrected, matrix
```

**GridExtractor 新增 `extract_with_visibility()` 方法**，支持格子级可见性标记：

```python
# grid.py 新增内容

@dataclass
class CellInfo:
    """单个格子的元信息"""
    row: int
    col: int
    color: np.ndarray          # RGB uint8
    is_visible: bool           # 格子中心在可见区域内
    is_edge: bool              # 格子靠近截断边
    confidence: float          # 该格子的比较置信度 [0,1]


class GridExtractor:
    # ... 保留现有 extract() 方法 ...

    def extract_with_visibility(
        self,
        board_image: np.ndarray,
        rows: int,
        cols: int,
        visibility_mask: np.ndarray,
        edge_margin: int = 2,
    ) -> list[CellInfo]:
        """
        带可见性标记的网格提取。

        Args:
            board_image: 透视校正后的板子图像
            rows, cols: 格数
            visibility_mask: HxW bool 可见区域掩码
                **必须是校正后坐标系的掩码**（通过 warpPerspective 变换得到），
                而非原始图像坐标系的掩码。
            edge_margin: 截断边附近多少个格子标记为 "边缘"
        """
        h, w = board_image.shape[:2]
        # visibility_mask 应与 board_image 同尺寸（校正后）
        assert visibility_mask.shape[:2] == (h, w), (
            f"visibility_mask 尺寸 {visibility_mask.shape[:2]} "
            f"与 board_image {h, w} 不匹配，请确保已透视变换"
        )

        cell_h = h / rows
        cell_w = w / cols
        cells = []

        for r in range(rows):
            for c in range(cols):
                # 格子中心
                cy = int((r + 0.5) * cell_h)
                cx = int((c + 0.5) * cell_w)

                # 可见性
                is_visible = (
                    0 <= cy < h and 0 <= cx < w and visibility_mask[cy, cx]
                )

                # 是否在边缘（靠近图像边界或不可见区域）
                is_edge = self._is_near_boundary(
                    r, c, rows, cols, visibility_mask, cell_h, cell_w, edge_margin,
                )

                # 颜色采样
                if is_visible:
                    margin_y = cell_h * 0.3
                    margin_x = cell_w * 0.3
                    y1 = max(0, int(r * cell_h + margin_y))
                    y2 = min(h, int((r + 1) * cell_h - margin_y))
                    x1 = max(0, int(c * cell_w + margin_x))
                    x2 = min(w, int((c + 1) * cell_w - margin_x))
                    if y2 > y1 and x2 > x1:
                        cell_region = board_image[y1:y2, x1:x2]
                        color = np.median(cell_region.reshape(-1, 3), axis=0).astype(np.uint8)
                    else:
                        color = np.array([0, 0, 0], dtype=np.uint8)
                else:
                    color = np.array([0, 0, 0], dtype=np.uint8)

                cells.append(CellInfo(
                    row=r, col=c,
                    color=color,
                    is_visible=is_visible,
                    is_edge=is_edge,
                    confidence=0.5 if is_edge else (1.0 if is_visible else 0.0),
                ))

        return cells

    @staticmethod
    def _is_near_boundary(
        r, c, rows, cols, vis_mask, cell_h, cell_w, margin,
    ) -> bool:
        """判断格子是否靠近不可见区域边界。"""
        h, w = vis_mask.shape[:2]
        for dr in range(-margin, margin + 1):
            for dc in range(-margin, margin + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    cy = int((nr + 0.5) * cell_h)
                    cx = int((nc + 0.5) * cell_w)
                    if 0 <= cy < h and 0 <= cx < w and not vis_mask[cy, cx]:
                        return True
        return False
```

#### 3.9.3 `src/compare.py` — DiffComparator

新增 `compare_with_confidence()` 方法，过滤不可见和低置信度格子：

```python
# compare.py 新增内容

@dataclass
class DiffResult:
    """带置信度的差异结果"""
    row: int
    col: int
    type: str                     # "color_mismatch"
    photo_color: list[int]
    blueprint_color: list[int]
    cell_confidence: float        # 格子置信度
    is_reliable: bool             # 是否可靠（可见且非边缘）


class DiffComparator:
    # ... 保留现有 compare() 和 annotate() ...

    def compare_with_confidence(
        self,
        photo_cells: list[CellInfo],
        blueprint_grid: np.ndarray,
    ) -> list[DiffResult]:
        """
        带置信度的网格比较。自动过滤不可见格子。
        """
        diffs = []
        for cell in photo_cells:
            if not cell.is_visible:
                continue  # 跳过不可见格子

            r, c = cell.row, cell.col
            if r >= blueprint_grid.shape[0] or c >= blueprint_grid.shape[1]:
                continue

            photo_rgb = cell.color.astype(np.float32)
            bp_rgb = blueprint_grid[r, c].astype(np.float32)

            photo_lab = self._rgb_to_lab(photo_rgb)
            bp_lab = self._rgb_to_lab(bp_rgb)
            distance = np.sqrt(np.sum((photo_lab - bp_lab) ** 2))

            if distance > self.color_tolerance:
                diffs.append(DiffResult(
                    row=r, col=c,
                    type="color_mismatch",
                    photo_color=cell.color.tolist(),
                    blueprint_color=blueprint_grid[r, c].tolist(),
                    cell_confidence=cell.confidence,
                    is_reliable=cell.confidence >= 0.8,
                ))

        return diffs

    def annotate_with_confidence(
        self,
        photo: np.ndarray,
        diffs: list[DiffResult],
        rows: int,
        cols: int,
    ) -> np.ndarray:
        """
        带置信度的差异标注。
        可靠差异标红色，不可靠（边缘）标橙色。
        """
        result = photo.copy()
        h, w = result.shape[:2]
        cell_h = h / rows
        cell_w = w / cols

        for diff in diffs:
            r, c = diff.row, diff.col
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)

            color = (0, 0, 255) if diff.is_reliable else (0, 165, 255)  # 红色=可靠, 橙色=不可靠
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            size = min(int(cell_h), int(cell_w)) // 6
            cv2.line(result, (cx - size, cy - size), (cx + size, cy + size), color, 2)
            cv2.line(result, (cx - size, cy + size), (cx + size, cy - size), color, 2)

        return result
```

#### 3.9.4 `src/cli.py` — 管线集成

更新 CLI 入口，使用新的精化管线：

```python
# cli.py 改动要点（非完整代码，仅展示关键变化）

def main():
    # ... 参数解析 ...

    # 原有:
    # mask = detector.detect(photo)
    # corners = detector.extract_corners(mask)

    # 新接口:
    detection = detector.refine(photo, board_size=(rows, cols))

    print(f"检测置信度: 对象={detection.confidence.q_object:.2f} "
          f"检测={detection.confidence.q_detection:.2f} "
          f"综合={detection.confidence.total:.2f} ({detection.confidence.level})")

    if detection.confidence.total < 0.5:
        print("⚠ 置信度较低，结果可能不准确，建议重新拍摄")

    # 透视校正
    corrected, transform_matrix = corrector.correct_with_matrix(
        photo, detection.corners, output_size,
    )

    # 将可见区域掩码通过同一透视变换映射到校正后坐标系
    visibility_corrected = cv2.warpPerspective(
        detection.visibility_mask.astype(np.uint8),
        transform_matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
    ).astype(bool)

    # 网格提取（带可见性）
    cells = extractor.extract_with_visibility(
        corrected, rows, cols, visibility_corrected,
    )

    # 差异比较（带置信度过滤）
    diffs = comparator.compare_with_confidence(cells, blueprint_grid)

    reliable_diffs = [d for d in diffs if d.is_reliable]
    unreliable_diffs = [d for d in diffs if not d.is_reliable]

    print(f"发现 {len(reliable_diffs)} 处可靠差异")
    if unreliable_diffs:
        print(f"发现 {len(unreliable_diffs)} 处边缘区域差异（低置信度）")

    # 标注
    annotated = comparator.annotate_with_confidence(corrected, diffs, rows, cols)
    cv2.imwrite(output_path, annotated)
```

#### 3.9.5 `data/colors.json` — 新增板子颜色参考

在现有颜色数组后新增（或独立为新字段）：

```json
{
  "board_color_ref": {
    "L_range": [75, 100],
    "a_range": [-8, 8],
    "b_range": [-10, 15],
    "description": "拼豆板底色参考范围（LAB空间）- 白色/米白色塑料"
  }
}
```

> 注：具体集成方式待实现阶段确定——可以在 `RefinerConfig` 中硬编码默认值，同时支持从 `colors.json` 加载覆盖。

### 3.10 错误处理

```python
class DetectionError(Exception):
    """板子检测失败"""
    pass

class RefinementError(Exception):
    """边缘精化失败（所有候选直线均被过滤）"""
    pass
```

`EdgeRefiner.refine()` 的错误处理策略：

| 场景 | 处理 |
|------|------|
| YOLO 未检测到板子 | 抛出 `DetectionError` |
| 颜色初筛未通过 | 降级到原有 `extract_corners()` 方法 |
| 霍夫检测无候选直线 | 降级到原有 `extract_corners()` 方法 |
| 所有候选被 L4/L5/L6 过滤 | 降级到原有 `extract_corners()` 方法 |
| 截断重建后角点不合理 | 降级到原有 `extract_corners()` 方法 |

降级策略确保：**新方法失败时，系统不会崩溃，而是回退到原有行为。**

### 3.11 依赖变更

```
# requirements.txt 新增
scipy>=1.10  # cKDTree 用于轮廓贴合度计算
```

现有依赖（numpy, opencv-python, ultralytics）无需变更。

## 4. 测试策略

### 4.1 单元测试

| 测试文件 | 覆盖内容 |
|----------|---------|
| `tests/test_edge_refiner.py` | EdgeRefiner 核心逻辑 |
| `tests/test_truncation.py` | 截断检测各类型 |
| `tests/test_texture.py` | FFT 纹理验证 |
| `tests/test_confidence.py` | 置信度评估 |

重点测试用例：

- **截断检测**：构造贴近画面边界 vs 不贴近的掩码轮廓段，验证分类正确
- **直线拟合**：在已知直线上生成带噪声的轮廓点，验证拟合精度
- **颜色初筛**：白色/米白色区域通过，深色区域拒绝
- **纹理验证**：规则点阵图像 → 高分；均匀色块 → 低分；随机噪声 → 低分
- **截断重建**：
  - T0: 4 条可见边 → 无重建
  - T1: 3 条可见边 → 第 4 边平行约束重建
  - T2a: 2 条相邻可见边 → 宽高比重建
  - T2b: 2 条对边可见 → 垂直方向估算
  - T3: 1 条可见边 → 粗略估算
- **置信度**：高拟合物 → 高分；低拟合 + 高方差 → 低分
- **降级回退**：各种失败场景 → 回退到 `extract_corners()`

### 4.2 Mock 策略

所有单元测试不需要真实模型文件：
- YOLOv8-seg 输出用预构造的 numpy 掩码模拟
- 图像用 `np.random` 或简单的渐变图/棋盘格图
- 不依赖真实拼豆板照片

### 4.3 回归测试

使用现有的测试 fixture（`tests/fixtures/`）验证：
- 改动后管线对完整板子照片的检测结果与改动前一致（或更优）
- `BoardDetector.detect()` 和 `extract_corners()` 原有接口不变

## 5. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/edge_refiner.py` | 新增 | 核心精化模块（~500 行） |
| `src/detect.py` | 修改 | 新增 `refine()` 方法 |
| `src/grid.py` | 修改 | 新增 `CellInfo`, `extract_with_visibility()` |
| `src/compare.py` | 修改 | 新增 `DiffResult`, `compare_with_confidence()`, `annotate_with_confidence()` |
| `src/cli.py` | 修改 | 使用新管线 |
| `data/colors.json` | 修改 | 新增 `board_color_ref` 字段 |
| `requirements.txt` | 修改 | 新增 `scipy` |
| `tests/test_edge_refiner.py` | 新增 | 精化模块测试 |
| `tests/test_truncation.py` | 新增 | 截断检测测试 |
| `tests/test_texture.py` | 新增 | 纹理验证测试 |
| `tests/test_confidence.py` | 新增 | 置信度测试 |
