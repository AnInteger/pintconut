# 点豆标注（Click-Bead Labeling）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `bead_annotate_ui.py` 新增「点豆」标注模式——点一颗豆中心 → 自动找边缘半径 → 画框；取代固定半径的「加框」模式。

**Architecture:** 把验证过的 `grad_outer + 半径钳制` 算法作为纯函数落进 `src/bead_label_service.py`（`gradient_magnitude` + `find_bead_radius`，单一真源）；UI 加载照片时预算一次梯度，点豆模式下每次点击调 `find_bead_radius`，第二次点同颗边缘可手动覆盖半径。角点网格模式保留不动。

**Tech Stack:** Python 3.10+ / OpenCV (Sobel 梯度、cv2.circle) / NumPy / Gradio / pytest

**Spec:** `docs/superpowers/specs/2026-06-23-click-bead-labeling-design.md`

---

## 文件结构

| 文件 | 责任 | 改动 |
|---|---|---|
| `src/bead_label_service.py` | 纯逻辑（算法唯一真源） | 新增 `gradient_magnitude`、`_ring_profile`、`find_bead_radius` |
| `tests/test_bead_label_service.py` | service 单测 | 新增 `find_bead_radius` 的 5 个用例 |
| `tests/validation/eval_edge_finder.py` | 用户真值基准评估 | 改为 import service 的两个函数；PRIMARY=`find_bead_radius`；删重复实现 |
| `src/bead_annotate_ui.py` | Gradio 标注 UI | 新「点豆」模式取代「加框」；`h_click` 点豆分支；`_draw` 红框警示；加载时预算 gmag |
| `tests/test_bead_annotate_ui.py` | UI handler 单测 | 更新 `h_click` 调用（去 radius 参数）；加 3 个点豆用例 |

---

## Task 1: service — `gradient_magnitude` + `find_bead_radius`（TDD）

**Files:**
- Modify: `src/bead_label_service.py`（文件末尾追加）
- Test: `tests/test_bead_label_service.py`（追加）

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_label_service.py` 末尾）

```python
def _gmag_ring(size, cx, cy, ring_r, val=50.0):
    """gmag with a single high-gradient ring at radius ring_r (rest zero)."""
    g = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    g[np.abs(r - ring_r) < 1.5] = val
    return g


def test_find_bead_radius_clean_bead():
    from src.bead_label_service import gradient_magnitude, find_bead_radius
    from tests._synth import render_beads
    img = render_beads(np.array([[60.0, 60.0]]), img_size=(120, 120),
                       bead_radius=30, body_color=(0, 0, 200))  # BGR 红
    gmag = gradient_magnitude(img)
    r, warn = find_bead_radius(gmag, 60, 60)
    assert abs(r - 30) <= 3
    assert warn is False


def test_find_bead_radius_skips_highlight():
    """Moderate white highlight inside red bead: grad_outer picks the outer edge."""
    import cv2
    from src.bead_label_service import gradient_magnitude, find_bead_radius
    from tests._synth import render_beads
    img = render_beads(np.array([[60.0, 60.0]]), img_size=(120, 120),
                       bead_radius=30, body_color=(0, 0, 200))
    cv2.circle(img, (60, 60), 10, (255, 255, 255), -1)  # white highlight
    gmag = gradient_magnitude(img)
    r, warn = find_bead_radius(gmag, 60, 60)
    assert abs(r - 30) <= 3   # outer edge, not the highlight ring at r=10
    assert warn is False


def test_find_bead_radius_ballooning_warns_without_priors():
    """Flat-high gradient (no clean edge) → candidate hits ceiling → warn=True."""
    from src.bead_label_service import find_bead_radius
    gmag_flat = np.ones((140, 140), dtype=np.float32)  # flat → outermost = r_max-1
    r, warn = find_bead_radius(gmag_flat, 70, 70)
    assert warn is True


def test_find_bead_radius_clamps_high_with_priors():
    """Ballooning candidate (119) clamped to 1.2*median(55)=66 once >=3 priors."""
    from src.bead_label_service import find_bead_radius
    gmag_flat = np.ones((140, 140), dtype=np.float32)
    r, warn = find_bead_radius(gmag_flat, 70, 70, prior_radii=[55, 56, 54])
    assert r == 66
    assert warn is False


def test_find_bead_radius_clamps_low_with_priors():
    """Tiny candidate (5, floor hit) clamped to 0.8*median(55)=44 once >=3 priors."""
    from src.bead_label_service import find_bead_radius
    gmag = _gmag_ring(140, 70, 70, ring_r=4)  # ring at r=4 → candidate≈5 (floor hit)
    r, warn = find_bead_radius(gmag, 70, 70, prior_radii=[55, 56, 54])
    assert r == 44
    assert warn is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_label_service.py -k find_bead_radius -v`
Expected: FAIL — `ImportError: cannot import name 'gradient_magnitude'`（函数还没实现）

- [ ] **Step 3: 实现**（追加到 `src/bead_label_service.py` 末尾）

```python
def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a BGR image (single-channel)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _ring_profile(sample: np.ndarray, cx, cy, r_min=3, r_max=120, n_ang=120):
    """Mean of `sample` along each circle of radius r (radial profile)."""
    H, W = sample.shape
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    prof = []
    for r in range(r_min, r_max + 1):
        xs = np.round(cx + r * cos).astype(int)
        ys = np.round(cy + r * sin).astype(int)
        ok = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        prof.append(float(sample[ys[ok], xs[ok]].mean()) if ok.sum() > n_ang * 0.6 else 0.0)
    return np.array(prof), list(range(r_min, r_max + 1))


def find_bead_radius(gmag: np.ndarray, cx, cy, prior_radii=None,
                     r_min=3, r_max=120) -> tuple[int, bool]:
    """Edge radius from a center click. Returns (radius, warn).

    Algorithm (validated on user ground truth gt_NYQC4978.txt, 11/12 within 6px):
      1. grad_outer: outermost radial-gradient ring above 0.6*peak (skips highlights).
      2. clamp to [0.8, 1.2]*median(prior_radii) once >=3 priors exist (kills ballooning).
      3. warn=True when the pre-clamp candidate hit floor/ceiling AND no clamp ran.
    """
    prof, rs = _ring_profile(gmag, cx, cy, r_min, r_max)
    peak = float(prof.max()) if prof.size else 0.0
    if peak <= 0:
        return r_min, True
    thr = 0.6 * peak
    outer = r_min
    for k in range(1, len(prof) - 1):
        if prof[k] > thr and prof[k] >= prof[k - 1] and prof[k] >= prof[k + 1]:
            outer = rs[k]          # keep updating -> outermost local max above thr
    candidate = outer if outer > r_min else rs[int(np.argmax(prof))]

    hit_bound = candidate <= r_min + 2 or candidate >= 0.85 * r_max

    radii = list(prior_radii or [])
    if len(radii) >= 3:
        med = float(np.median(radii))
        candidate = float(np.clip(candidate, 0.8 * med, 1.2 * med))
        return int(round(candidate)), False

    return int(round(candidate)), bool(hit_bound)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_label_service.py -k find_bead_radius -v`
Expected: 5 passed

- [ ] **Step 5: 跑全量 service 测试确认无回归**

Run: `python -m pytest tests/test_bead_label_service.py -v`
Expected: all passed（原有用例不受影响）

- [ ] **Step 6: 提交**

```bash
git add src/bead_label_service.py tests/test_bead_label_service.py
git commit -m "feat(bead_label_service): find_bead_radius (grad_outer + radius clamp)"
```

---

## Task 2: eval 脚本改为 import service（单一真源）

**Files:**
- Modify: `tests/validation/eval_edge_finder.py`

把 `gradient_magnitude` 和 grad_outer 的重复实现删掉，从 service import；PRIMARY 改为生产函数 `find_bead_radius`（带 prior_radii）。`gradient`/`coverage` 基线保留本地作对比。

- [ ] **Step 1: 改 import + 删重复函数**

把文件顶部 import 区（`import cv2` / `import numpy as np` 之后）替换为：

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.bead_label_service import gradient_magnitude, find_bead_radius
```

删除文件里的 `def gradient_magnitude(...)`（原 47-51 行）和 `def find_radius_gradient_outer(...)`（原 74-85 行）——它们现在是 service 里的单一真源。`_ring_profile`、`find_radius_gradient`、`find_radius_coverage`、`find_radius_saturation` 保留（基线/辅助）。

- [ ] **Step 2: PRIMARY 改为 find_bead_radius，带 prior_radii**

把模块常量区改为：

```python
IMG_DEFAULT = "training/test_images/training/1/NYQC4978.png"
GT_DEFAULT = "tests/validation/gt_NYQC4978.txt"
OVERLAY_OUT = "tests/validation/overlay_algo_vs_gt_NYQC4978.png"
PRIMARY = "find_bead_radius"
```

在 `main()` 里 `gmag = gradient_magnitude(img)` 这行不变（现在调用的是 import 来的函数）。

新增一个对 PRIMARY 的评估函数（用其它豆的真值半径作 prior，模拟「板子其余已标」）：

```python
def run_primary(gmag, gt):
    truths = [r for _, _, r in gt]
    radii, r_errs, ious = [], [], []
    for i, (cx, cy, r_gt) in enumerate(gt):
        prior = truths[:i] + truths[i + 1:]   # other beads' true radii as prior
        r_algo, _warn = find_bead_radius(gmag, cx, cy, prior_radii=prior)
        radii.append(r_algo)
        r_errs.append(abs(r_algo - r_gt))
        ious.append(iou(box(cx, cy, r_algo), box(cx, cy, r_gt)))
    return radii, r_errs, ious
```

- [ ] **Step 3: main() 输出 PRIMARY 结果**

在 `main()` 末尾（原 `RESULT(grad_outer)` 那段之后）追加 PRIMARY 段，删除/替换原基于 `grad_outer` 的 verdict：

```python
    print(f"\n  -- PRIMARY: {PRIMARY} (production: grad_outer + clamp) --")
    pr, pe, pi = run_primary(gmag, gt)
    pnf = sum(1 for e in pe if e > 10)
    print(f"  {PRIMARY}: |dR| median={med(pe):.2f} max={max(pe):.0f} | "
          f"IoU median={med(pi):.2f} min={min(pi):.2f} | off-by>10px={pnf}/{len(gt)}")
    for i, (cx, cy, r_gt) in enumerate(gt):
        flag = "  <-- off" if pe[i] > 10 else ""
        print(f"    bead {i+1}: r_gt={r_gt:.0f} r_algo={pr[i]} |dR|={pe[i]:.0f} IoU={pi[i]:.2f}{flag}")

    # overlay: green=truth, magenta=PRIMARY
    disp = img.copy()
    for (cx, cy, r_gt), r_algo in zip(gt, pr):
        cv2.circle(disp, (int(cx), int(cy)), int(r_gt), (0, 255, 0), 2)
        cv2.circle(disp, (int(cx), int(cy)), int(r_algo), (255, 0, 255), 1)
    cv2.imwrite(OVERLAY_OUT, disp)
    print(f"\noverlay -> {OVERLAY_OUT}  (green=truth, magenta={PRIMARY})")
```

（保留上面的 `gradient/grad_outer/coverage` 对比表段不动，作基线参照。）

- [ ] **Step 4: 跑 eval，确认 PRIMARY 数**

Run: `python tests/validation/eval_edge_finder.py`
Expected: 末尾输出类似
```
  find_bead_radius: |dR| median=2.00 max=13 | IoU median=0.93 min=0.65 | off-by>10px=1/12
    bead 11: r_gt=53 r_algo=66 |dR|=13 IoU=0.65  <-- off
```
即：11 颗 |dR|≤6px，#11 被 clamp 到 66（IoU 0.65，相比 raw grad_outer 的 0.20 已大幅改善；残余 13px 由 UI 红框+手覆盖收尾）。

- [ ] **Step 5: 提交**

```bash
git add tests/validation/eval_edge_finder.py
git commit -m "refactor(eval): import find_bead_radius from service (single source of truth)"
```

---

## Task 3: UI 加载时预算 gmag + state 字段

**Files:**
- Modify: `src/bead_annotate_ui.py`（`_new_state`、`h_load`、import 行）
- Test: `tests/test_bead_annotate_ui.py`（追加）

- [ ] **Step 1: 写失败测试**（追加到 `tests/test_bead_annotate_ui.py`）

```python
def test_h_load_sets_gmag(tmp_path, monkeypatch):
    import src.bead_annotate_ui as ui
    monkeypatch.setattr(ui, "PHOTOS_DIR", str(tmp_path))
    cv2.imwrite(str(tmp_path / "t.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    _img, state, _status, _bl, _st = ui.h_load("t.png", ui._new_state())
    assert state["gmag"] is not None
    assert state["gmag"].shape == (20, 20)
    assert state["pending"] is None
```

文件顶部已 `import numpy as np`；补 `import cv2`（若未有）。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_annotate_ui.py::test_h_load_sets_gmag -v`
Expected: FAIL — `KeyError: 'gmag'`（state 还没这字段）

- [ ] **Step 3: 改 import + `_new_state` + `h_load`**

import 行（原 16 行）改为：

```python
from src.bead_label_service import (generate_grid_boxes, preview_box_colors, export_yolo,
                                    gradient_magnitude, find_bead_radius)
```

`_new_state` 改为（加 `gmag`、`pending`）：

```python
def _new_state(img_bgr=None, name="", rows=29, cols=29):
    return {"img_bgr": img_bgr, "boxes": [], "corners": [], "rows": rows, "cols": cols,
            "name": name, "next_id": 0, "gmag": None, "pending": None}
```

`h_load` 成功分支加 gmag 预算（替换原 `state = _new_state(img, ...)` 那行及其 return）：

```python
def h_load(name, state):
    path = os.path.join(PHOTOS_DIR, name or "")
    img = cv2.imread(path) if name else None
    if img is None:
        return None, _new_state(), f"❌ 读不到 {name}", gr.update(), _stats()
    state = _new_state(img, os.path.splitext(name)[0])
    state["gmag"] = gradient_magnitude(img)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), state, f"✅ 已加载 {name}", gr.update(choices=_choices(state)), _stats()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_bead_annotate_ui.py::test_h_load_sets_gmag -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/bead_annotate_ui.py tests/test_bead_annotate_ui.py
git commit -m "feat(bead_annotate_ui): precompute gradient magnitude on photo load"
```

---

## Task 4: UI「点豆」模式（取代「加框」）+ 红框警示

**Files:**
- Modify: `src/bead_annotate_ui.py`（`h_click`、`_draw`、`build_ui` 的 mode/radio/wiring）
- Test: `tests/test_bead_annotate_ui.py`（更新现有调用 + 加点豆用例）

- [ ] **Step 1: 更新现有测试（去掉 radius 参数）+ 写点豆失败测试**

把现有 3 个 `h_click` 角点/加框用例的调用去掉 `radius` 实参（签名不再有 radius），并把 `test_h_click_box_mode_adds_manual_box` 替换为点豆用例。

`test_h_click_corner_mode_appends_click` 改为：
```python
    _canvas, state2, _boxlist, status = h_click(
        _FakeSelectData(123.0, 456.0), state, "角点", False)
```

`test_h_click_caps_at_four_corners` 里循环改为：
```python
        _img, state, _b, _s = h_click(_FakeSelectData(x, y), state, "角点", False)
```

删掉 `test_h_click_box_mode_adds_manual_box`，替换为以下 3 个点豆用例（追加）：

```python
def _gmag_ring(size, cx, cy, ring_r, val=50.0):
    g = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    g[np.abs(r - ring_r) < 1.5] = val
    return g


def test_h_click_dianzhu_center_adds_auto_box():
    from src.bead_annotate_ui import h_click, _new_state
    state = _new_state(img_bgr=np.zeros((120, 120, 3), dtype=np.uint8))
    state["gmag"] = _gmag_ring(120, 60, 60, ring_r=15)   # clean ring → r≈15, warn=False
    _img, state2, _b, _s = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)
    assert len(state2["boxes"]) == 1
    b = state2["boxes"][0]
    assert b["source"] == "auto"
    assert b["cx"] == 60 and b["cy"] == 60
    assert b["warn"] is False
    assert state2["pending"] is not None      # awaiting possible edge-override


def test_h_click_dianzhu_edge_override():
    from src.bead_annotate_ui import h_click, _new_state
    state = _new_state(img_bgr=np.zeros((120, 120, 3), dtype=np.uint8))
    state["gmag"] = _gmag_ring(120, 60, 60, ring_r=15)
    _, state, _, _ = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)   # center
    # 2nd click 10px from center, inside pending box (r=15) → edge override
    _, state2, _, _ = h_click(_FakeSelectData(70.0, 60.0), state, "点豆", False)
    b = state2["boxes"][0]
    assert b["source"] == "manual"
    assert b["width"] == 20                    # 2 * 10
    assert state2["pending"] is None


def test_h_click_dianzhu_outside_is_new_bead():
    from src.bead_annotate_ui import h_click, _new_state
    state = _new_state(img_bgr=np.zeros((160, 160, 3), dtype=np.uint8))
    # two rings so both centers find a clean edge
    g = _gmag_ring(160, 60, 60, ring_r=15) + _gmag_ring(160, 110, 60, ring_r=15)
    state["gmag"] = g
    _, state, _, _ = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)   # bead 1
    # 2nd click 50px away → outside pending box (r=15) → new bead
    _, state2, _, _ = h_click(_FakeSelectData(110.0, 60.0), state, "点豆", False)
    assert len(state2["boxes"]) == 2
    assert state2["pending"] is not None       # new pending on bead 2
```

`test_h_click_annotates_evt_as_selectdata` 不变。

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_bead_annotate_ui.py -v`
Expected: FAIL — 角点用例因多传了 `radius` 实参报错 / 点豆用例因 `h_click` 还没点豆分支而失败

- [ ] **Step 3: 改 `_draw`（红框警示）**

把 `_draw` 里画框循环（原 114-116 行）改为按 `warn` 选色：

```python
    for b in state["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        color = (0, 0, 255) if b.get("warn") else SRC_BGR.get(b["source"], (255, 255, 255))
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
```

（`(0,0,255)` 是 BGR 红。）

- [ ] **Step 4: 重写 `h_click`（去 radius、加 点豆 分支）**

整段替换 `h_click`（原 141-162 行）为：

```python
def h_click(evt: gr.SelectData, state, mode, show_color):
    img = state.get("img_bgr")
    if img is None or evt is None:
        return _draw(state, show_color), state, gr.update(), ""
    x, y = evt.index
    if mode == "角点":
        corners = list(state.get("corners", []))
        if len(corners) < 4:
            corners.append((float(x), float(y)))
            state = {**state, "corners": corners}
        n = len(state["corners"])
        hint = {0: "①点左上角豆", 1: "②点右上角豆", 2: "③点右下角豆",
                3: "④点左下角豆"}.get(n, "")
        msg = f"✅ 已点 {n}/4 角豆，可生成网格" if n == 4 else f"已点 {n}/4 角豆，下一颗：{hint}"
        return _draw(state, show_color), state, gr.update(), msg

    # 点豆模式
    gmag = state.get("gmag")
    if gmag is None:
        return _draw(state, show_color), state, gr.update(), "❌ 先加载照片"
    cx, cy = int(round(x)), int(round(y))
    pending = state.get("pending")
    if pending:
        pcx, pcy, _pr = pending
        if ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 < _pr:
            # 2nd click inside pending box → manual edge override
            r = int(((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5)
            boxes = list(state["boxes"])
            last = dict(boxes[-1])
            last.update({"xyxy": [pcx - r, pcy - r, pcx + r, pcy + r],
                         "cx": pcx, "cy": pcy, "width": 2 * r, "height": 2 * r,
                         "source": "manual", "warn": False})
            boxes[-1] = last
            state = {**state, "boxes": boxes, "pending": None}
            return _draw(state, show_color), state, gr.update(choices=_choices(state)), "已手动覆盖半径"
    # new bead center
    prior = [b["width"] / 2 for b in state["boxes"]]
    r, warn = find_bead_radius(gmag, cx, cy, prior_radii=prior or None)
    nb, nid = _stamp_ids(state.get("next_id", 0), [{
        "xyxy": [cx - r, cy - r, cx + r, cy + r], "cx": cx, "cy": cy,
        "width": 2 * r, "height": 2 * r, "source": "auto", "warn": warn}])
    state = {**state, "boxes": state["boxes"] + nb, "next_id": nid,
             "pending": (cx, cy, r)}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), ""
```

- [ ] **Step 5: 改 `build_ui`（mode 选项 + 删 radius + 改 wiring）**

mode Radio（原 224 行）改为：
```python
        mode = gr.Radio(["角点", "点豆"], value="角点", label="点击模式")
```

删除 radius Number 组件（原 230 行 `radius = gr.Number(...)`）。

canvas.select 接线（原 245 行）改为：
```python
        canvas.select(h_click, [state, mode, show_color], [canvas, state, box_list, status])
```

顶部说明 Markdown（原 203-212 行）末尾「框色」一句改为：
```
框色：🟢生成/自动 / 🔵手动覆盖 / 🔴算法警示(点边缘覆盖)。点豆模式：点中心→自动框；再点同颗边缘→手动覆盖半径。
```

- [ ] **Step 6: 跑 UI 测试确认通过**

Run: `python -m pytest tests/test_bead_annotate_ui.py -v`
Expected: all passed（含 3 个新点豆用例 + 更新后的角点用例）

- [ ] **Step 7: 跑全量测试确认无回归**

Run: `python -m pytest tests/ -v --ignore=tests/test_label_ui_integration.py`
Expected: all passed（集成测试需 Gradio server，跳过）

- [ ] **Step 8: 提交**

```bash
git add src/bead_annotate_ui.py tests/test_bead_annotate_ui.py
git commit -m "feat(bead_annotate_ui): 点豆 mode (auto edge-find + radius override), replaces 加框"
```

---

## Task 5: 手动冒烟（用户执行）

**Files:** 无代码改动（验证）

- [ ] **Step 1: 启动 UI**

Run: `python src/bead_annotate_ui.py`
打开 `http://localhost:7860`（**改了 UI 必须硬刷新浏览器**，见 memory）。

- [ ] **Step 2: 验证点豆流程**

1. 加载 `NYQC4978.png`（或任一照片）
2. 模式选「点豆」
3. 依次点几颗豆中心 → 每颗出绿框（auto）；点边缘可手动覆盖（转蓝）
4. 故意点一颗高光/病理豆 → 应见红框警示 → 点其边缘覆盖
5. 导出 YOLO → 检查 `training/bead_dataset/labels/train/` 下 `.txt` 每行 5 个数

- [ ] **Step 3: 角点网格模式回归确认**

切「角点」模式，点 4 角豆 + 生成网格，确认未被点豆改动破坏。

---

## Self-Review（写完自查）

**Spec 覆盖：**
- §2/§3 点豆模式 + 取代加框 → Task 4 ✓
- §4.1 grad_outer / §4.2 clamp → Task 1 ✓
- §5.1 service 两函数 + eval import → Task 1 + Task 2 ✓
- §5.2 UI 模式/h_click/warn/删加框 → Task 4 ✓
- §5.3 角点网格保留 → Task 4 仅加分支、不动角点路径 + Task 5 回归 ✓
- §6 错误处理（warn/override/撤销）→ Task 4 warn+override；撤销复用 h_delete ✓
- §7 测试 → Task 1（5 例）+ Task 3（1 例）+ Task 4（3 例）✓

**类型一致：** `find_bead_radius(gmag, cx, cy, prior_radii=None, r_min=3, r_max=120) -> (int, bool)` 在 Task 1/2/4 签名一致；box 的 `warn: bool` 字段在 Task 4 写入与 `_draw` 读取一致；`h_click(evt, state, mode, show_color)` 去掉 radius 后测试调用一致。

**占位符扫描：** 无 TBD/TODO；每步含完整代码或确切命令。
