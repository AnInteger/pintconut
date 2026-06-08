# Pintconut 标注工具 UI 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Gradio 搭建一个向导式 Web UI，引导用户完成从上传照片到导出 YOLO 数据集的完整标注流程。

**Architecture:** 从现有 `training/semi_auto_label.py` 提取核心分割/标注逻辑到 `src/label_service.py`（纯函数，无 UI 依赖），然后用 Gradio 在 `src/label_ui.py` 中构建四步向导 UI。

**Tech Stack:** Gradio >=4.0, FastSAM (ultralytics), OpenCV, NumPy

---

## 文件结构

```
pintconut/
├── src/
│   ├── label_service.py      # 标注核心逻辑（从 semi_auto_label.py 提取）
│   └── label_ui.py           # Gradio Web UI 入口
└── tests/
    └── test_label_service.py # label_service 单元测试
```

---

### Task 1: 安装 Gradio 依赖

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 添加 gradio 到 requirements.txt**

在 `requirements.txt` 末尾追加一行：

```
gradio>=4.0.0
```

- [ ] **Step 2: 安装**

Run: `source .venv/bin/activate && pip install "gradio>=4.0.0"`
Expected: 安装成功

- [ ] **Step 3: 验证**

Run: `source .venv/bin/activate && python -c "import gradio; print(gradio.__version__)"`
Expected: 输出 4.x 版本号

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add gradio dependency"
```

---

### Task 2: 提取标注核心逻辑到 label_service.py

**Files:**
- Create: `src/label_service.py`
- Create: `tests/test_label_service.py`

从 `training/semi_auto_label.py` 提取纯逻辑（无 UI、无 print、无 input），封装为可被 Gradio 和终端工具共用的函数。

- [ ] **Step 1: 写 label_service 的测试**

`tests/test_label_service.py`:

```python
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
    """矩形 mask 应返回 (True, ratio接近1.0)。"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255
    ok, ratio = is_rectangular(mask)
    assert ok is True
    assert ratio > 0.9


def test_is_rectangular_false():
    """不规则 mask 应返回 False。"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    # L-shaped mask
    mask[20:50, 20:50] = 255
    mask[50:80, 20:35] = 255
    ok, ratio = is_rectangular(mask)
    assert ok is False


def test_is_rectangular_empty():
    """空 mask 应返回 False。"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    ok, ratio = is_rectangular(mask)
    assert ok is False


def test_segment_image_returns_candidates():
    """segment_image 应返回候选区域列表。"""
    # 创建一个有矩形区域的测试图片
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    img[50:250, 50:250] = [200, 200, 200]
    candidates = segment_image(img)
    assert isinstance(candidates, list)
    if candidates:
        assert "mask" in candidates[0]
        assert "area_ratio" in candidates[0]
        assert "rect_ratio" in candidates[0]


def test_draw_candidates_returns_image():
    """draw_candidates 应返回与输入同尺寸的图片。"""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 128
    candidates = [{
        "mask": np.zeros((200, 200), dtype=np.uint8),
        "area_ratio": 0.5,
        "rect_ratio": 0.9,
    }]
    candidates[0]["mask"][50:150, 50:150] = 255
    result = draw_candidates(img, candidates)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_save_label_creates_files():
    """save_label 应创建图片和标签文件。"""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "images")
        lbl_dir = os.path.join(tmpdir, "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)

        # 创建一个假图片
        img_path = os.path.join(img_dir, "test.png")
        import cv2
        cv2.imwrite(img_path, np.ones((100, 100, 3), dtype=np.uint8) * 255)

        save_label(mask, img_path, img_dir, lbl_dir, 100, 100)

        label_path = os.path.join(lbl_dir, "test.txt")
        assert os.path.exists(label_path)
        with open(label_path) as f:
            content = f.read()
        assert content.startswith("0 ")  # class 0
        # Should have coordinate pairs
        parts = content.strip().split()
        assert len(parts) >= 5  # "0" + at least 2 x,y pairs


def test_generate_dataset_yaml():
    """generate_dataset_yaml 应创建 data.yaml。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create expected dirs
        os.makedirs(os.path.join(tmpdir, "images", "train"))
        os.makedirs(os.path.join(tmpdir, "images", "valid"))
        os.makedirs(os.path.join(tmpdir, "labels", "train"))
        os.makedirs(os.path.join(tmpdir, "labels", "valid"))

        yaml_path = generate_dataset_yaml(tmpdir)
        assert os.path.exists(yaml_path)
        with open(yaml_path) as f:
            content = f.read()
        assert "beadboard" in content
        assert "train" in content
```

- [ ] **Step 2: 运行测试确认失败**

Run: `source .venv/bin/activate && python -m pytest tests/test_label_service.py -v`
Expected: FAIL

- [ ] **Step 3: 实现 label_service.py**

`src/label_service.py`:

```python
"""
Label service - core logic for bead board labeling.

Provides pure functions (no UI, no I/O side effects beyond file operations)
for segmenting images, filtering candidates, and saving YOLO labels.
Shared by both the Gradio UI and the CLI labeling tool.
"""
import os
import shutil

import cv2
import numpy as np

# Colors for drawing candidate regions (BGR)
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 0),
    (0, 128, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def is_rectangular(mask: np.ndarray, threshold: float = 0.80) -> tuple[bool, float]:
    """Check if a binary mask region is approximately rectangular.

    Compares mask area to its convex hull area. Ratio close to 1.0 = rectangular.

    Args:
        mask: Binary mask (uint8).
        threshold: Minimum rectangularity ratio.

    Returns:
        (is_rectangular, ratio) tuple.
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False, 0.0
    contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(contour)
    if mask_area < 100:
        return False, 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False, 0.0
    ratio = mask_area / hull_area
    return ratio > threshold, ratio


def segment_image(
    image: np.ndarray,
    model=None,
) -> list[dict]:
    """Run FastSAM on an image and return filtered candidate regions.

    Args:
        image: Input image (HxWx3 BGR array).
        model: FastSAM model instance. If None, loads default.

    Returns:
        List of candidate dicts with keys: mask, area, area_ratio, rect_ratio.
    """
    if model is None:
        from ultralytics import FastSAM
        model = FastSAM("FastSAM-s.pt")

    h, w = image.shape[:2]
    img_area = h * w

    results = model(image, device="cpu", retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

    if results[0].masks is None:
        return []

    masks = results[0].masks.data.cpu().numpy()

    candidates = []
    for m in masks:
        mask_resized = cv2.resize(m, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        area = int(np.sum(mask_binary))
        area_ratio = area / img_area

        if area_ratio < 0.05 or area_ratio > 0.95:
            continue

        rect_ok, rect_ratio = is_rectangular(mask_binary, threshold=0.5)
        if not rect_ok:
            continue

        candidates.append({
            "mask": mask_binary,
            "area": area,
            "area_ratio": area_ratio,
            "rect_ratio": rect_ratio,
        })

    candidates.sort(key=lambda c: c["area"], reverse=True)
    return candidates[:10]


def draw_candidates(image: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Draw colored candidate regions with number labels on the image.

    Args:
        image: Original image (HxWx3 BGR).
        candidates: List of candidate dicts with 'mask' key.

    Returns:
        Annotated image (same shape as input).
    """
    display = image.copy()
    for i, cand in enumerate(candidates):
        color = COLORS[i % len(COLORS)]
        mask = cand["mask"]

        overlay = display.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, color, 2)

        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 30, 30 + i * 40
        else:
            cx, cy = 30, 30 + i * 40

        label = str(i)
        cv2.putText(display, label, (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(display, label, (cx - 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)

    return display


def save_label(
    mask: np.ndarray,
    image_path: str,
    output_images_dir: str,
    output_labels_dir: str,
    img_w: int,
    img_h: int,
) -> str:
    """Save a mask as YOLO segmentation label and copy source image.

    Args:
        mask: Binary mask (uint8, same size as image).
        image_path: Path to source image.
        output_images_dir: Directory to copy the image to.
        output_labels_dir: Directory to write the .txt label to.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Path to the created label file, or empty string if failed.
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    dest_image = os.path.join(output_images_dir, basename + ext)
    if not os.path.exists(dest_image):
        shutil.copy2(image_path, dest_image)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    label_path = os.path.join(output_labels_dir, basename + ".txt")
    with open(label_path, "w") as f:
        coords = []
        for point in contour:
            x = point[0][0] / img_w
            y = point[0][1] / img_h
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")
        f.write("0 " + " ".join(coords) + "\n")

    return label_path


def generate_dataset_yaml(dataset_dir: str) -> str:
    """Generate data.yaml for the dataset if it doesn't exist.

    Args:
        dataset_dir: Root directory of the dataset.

    Returns:
        Path to the data.yaml file.
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        return yaml_path

    yaml_content = f"""# Auto-generated dataset config
path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/valid

names:
  0: beadboard
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path
```

- [ ] **Step 4: 运行测试确认通过**

Run: `source .venv/bin/activate && python -m pytest tests/test_label_service.py -v`
Expected: 全部 PASS（`test_segment_image` 需要 FastSAM 模型，如果没加载到可能会慢）

- [ ] **Step 5: 运行全部测试确认无回归**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 6: Commit**

```bash
git add src/label_service.py tests/test_label_service.py
git commit -m "feat: add label service module (extracted from semi_auto_label)"
```

---

### Task 3: Gradio UI 实现

**Files:**
- Create: `src/label_ui.py`

实现四步向导 UI：上传 → 分割 → 逐张确认 → 导出。

- [ ] **Step 1: 实现 Gradio UI**

`src/label_ui.py`:

```python
"""
Pintconut 标注工具 - Gradio Web UI

四步向导流程：
  ① 上传照片 → ② 自动分割 → ③ 逐张确认 → ④ 导出数据集

启动方式:
    python src/label_ui.py
"""
import os
import shutil
import tempfile

import cv2
import gradio as gr
import numpy as np

from src.label_service import segment_image, draw_candidates, save_label, generate_dataset_yaml

# Global state
_model = None
_photos_dir = "training/photos"
_dataset_dir = "training/dataset"


def _get_model():
    """Lazy-load FastSAM model."""
    global _model
    if _model is None:
        from ultralytics import FastSAM
        _model = FastSAM("FastSAM-s.pt")
    return _model


def upload_photos(files):
    """Step 1: Save uploaded files to training/photos/."""
    if not files:
        return "❌ 请上传照片", gr.update(visible=False)

    os.makedirs(_photos_dir, exist_ok=True)
    count = 0
    for f in files:
        basename = os.path.basename(f.name)
        dest = os.path.join(_photos_dir, basename)
        shutil.copy2(f.name, dest)
        count += 1

    return (
        f"✅ 已上传 {count} 张照片到 {_photos_dir}/",
        gr.update(visible=True),
    )


def run_segmentation(progress=gr.Progress()):
    """Step 2: Run FastSAM on all photos, return annotated images."""
    model = _get_model()

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    images = sorted(
        f for f in os.listdir(_photos_dir)
        if f.lower().endswith(exts)
    )

    if not images:
        return [], "❌ 没有找到照片", gr.update(visible=False)

    results = []
    for i, img_file in enumerate(progress.tqdm(images, desc="分割中")):
        img_path = os.path.join(_photos_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue

        candidates = segment_image(image, model=model)

        if candidates:
            annotated = draw_candidates(image, candidates)
            # Convert BGR to RGB for Gradio
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            info = f"候选 {len(candidates)} 个区域"
        else:
            annotated_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            info = "⚠️ 无候选区域"

        results.append((annotated_rgb, f"{img_file}\n{info}"))

    status = f"✅ 已处理 {len(results)}/{len(images)} 张照片"
    return results, status, gr.update(visible=True)


def confirm_selection(img_index, candidate_index, gallery_images):
    """Step 3: Confirm selection for current image."""
    model = _get_model()

    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    images = sorted(
        f for f in os.listdir(_photos_dir)
        if f.lower().endswith(exts)
    )

    if img_index >= len(images):
        return "❌ 没有更多图片", img_index, gallery_images

    img_file = images[img_index]
    img_path = os.path.join(_photos_dir, img_file)

    image = cv2.imread(img_path)
    if image is None:
        return f"❌ 无法读取: {img_file}", img_index, gallery_images

    h, w = image.shape[:2]
    output_images_dir = os.path.join(_dataset_dir, "images", "train")
    output_labels_dir = os.path.join(_dataset_dir, "labels", "train")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    candidates = segment_image(image, model=model)

    if candidate_index < 0 or candidate_index >= len(candidates):
        return f"⚠️ 无效选择 (0-{len(candidates)-1})", img_index, gallery_images

    selected = candidates[candidate_index]
    label_path = save_label(
        selected["mask"], img_path,
        output_images_dir, output_labels_dir, w, h,
    )

    next_index = img_index + 1
    status = f"✅ {img_file} 已标注 (候选 {candidate_index})"

    if next_index >= len(images):
        status += " — 所有图片已处理！"
        return status, next_index, gallery_images

    return status, next_index, gallery_images


def skip_image(img_index, gallery_images):
    """Step 3: Skip current image."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    images = sorted(
        f for f in os.listdir(_photos_dir)
        if f.lower().endswith(exts)
    )

    next_index = img_index + 1
    img_file = images[img_index] if img_index < len(images) else "?"

    if next_index >= len(images):
        return f"⏭ 跳过 {img_file} — 所有图片已处理！", next_index, gallery_images

    return f"⏭ 跳过 {img_file}", next_index, gallery_images


def export_dataset():
    """Step 4: Generate dataset with data.yaml."""
    output_images_dir = os.path.join(_dataset_dir, "images", "train")
    output_labels_dir = os.path.join(_dataset_dir, "labels", "train")

    n_images = len([f for f in os.listdir(output_images_dir) if f.endswith(('.png', '.jpg'))]) if os.path.exists(output_images_dir) else 0
    n_labels = len([f for f in os.listdir(output_labels_dir) if f.endswith('.txt')]) if os.path.exists(output_labels_dir) else 0

    if n_labels == 0:
        return "❌ 没有标注数据，请先完成步骤③"

    # Create valid split (copy 20% from train)
    valid_images_dir = os.path.join(_dataset_dir, "images", "valid")
    valid_labels_dir = os.path.join(_dataset_dir, "labels", "valid")
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)

    yaml_path = generate_dataset_yaml(_dataset_dir)

    return (
        f"✅ 数据集已导出！\n"
        f"   图片: {n_images} 张\n"
        f"   标签: {n_labels} 个\n"
        f"   配置: {yaml_path}\n\n"
        f"🚀 下一步运行训练:\n"
        f"   python training/train.py --data {_dataset_dir}/data.yaml"
    )


def build_ui():
    """Build the Gradio UI with wizard steps."""
    with gr.Blocks(title="Pintconut 标注工具") as app:
        gr.Markdown("# 🫘 Pintconut 拼豆标注工具")
        gr.Markdown("四步完成训练数据标注：上传照片 → 自动分割 → 逐张确认 → 导出数据集")

        # State
        current_index = gr.State(value=0)

        with gr.Tabs():
            # Step 1: Upload
            with gr.Tab("① 上传照片"):
                gr.Markdown("上传拼板照片（支持多选）")
                file_input = gr.File(file_count="multiple", file_types=["image"], label="选择照片")
                upload_btn = gr.Button("📤 上传", variant="primary")
                upload_status = gr.Textbox(label="状态", interactive=False)
                upload_btn.click(
                    fn=upload_photos,
                    inputs=[file_input],
                    outputs=[upload_status],
                )

            # Step 2: Segment
            with gr.Tab("② 自动分割"):
                gr.Markdown("FastSAM 自动识别图片中的候选区域")
                seg_btn = gr.Button("🔍 开始分割", variant="primary")
                seg_status = gr.Textbox(label="状态", interactive=False)
                gallery = gr.Gallery(label="分割结果", columns=3, height=500)
                seg_btn.click(
                    fn=run_segmentation,
                    outputs=[gallery, seg_status],
                )

            # Step 3: Confirm
            with gr.Tab("③ 逐张确认"):
                gr.Markdown("查看分割结果，输入正确的候选区域编号进行确认")
                with gr.Row():
                    img_idx_input = gr.Number(value=0, label="图片序号（从0开始）", precision=0)
                    cand_idx_input = gr.Number(value=0, label="选择的候选区域编号", precision=0)
                with gr.Row():
                    confirm_btn = gr.Button("✅ 确认选择", variant="primary")
                    skip_btn = gr.Button("⏭ 跳过")
                confirm_status = gr.Textbox(label="状态", interactive=False)

                confirm_btn.click(
                    fn=confirm_selection,
                    inputs=[img_idx_input, cand_idx_input, gallery],
                    outputs=[confirm_status, img_idx_input, gallery],
                )
                skip_btn.click(
                    fn=skip_image,
                    inputs=[img_idx_input, gallery],
                    outputs=[confirm_status, img_idx_input, gallery],
                )

            # Step 4: Export
            with gr.Tab("④ 导出数据集"):
                gr.Markdown("将标注结果导出为 YOLO 格式数据集")
                export_btn = gr.Button("📦 导出数据集", variant="primary")
                export_status = gr.Textbox(label="状态", interactive=False, lines=8)
                export_btn.click(
                    fn=export_dataset,
                    outputs=[export_status],
                )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=False)
```

- [ ] **Step 2: 验证 UI 能启动**

Run: `source .venv/bin/activate && timeout 10 python src/label_ui.py 2>&1 || true`
Expected: 输出包含 `Running on local URL: http://127.0.0.1:7860`

- [ ] **Step 3: 运行全部测试确认无回归**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add src/label_ui.py
git commit -m "feat: add Gradio labeling UI with wizard flow"
```

---

### Task 4: 更新 .gitignore + requirements.txt 最终检查

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 添加 Gradio 相关忽略项**

在 `.gitignore` 末尾追加：

```
# Gradio
flagged/
.gradio/
```

- [ ] **Step 2: 运行全部测试**

Run: `source .venv/bin/activate && python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add Gradio entries to .gitignore"
```

---

## 实现顺序

```
Task 1  安装 Gradio
  ↓
Task 2  提取 label_service.py（纯逻辑 + 测试）
  ↓
Task 3  Gradio UI（label_ui.py）
  ↓
Task 4  .gitignore 更新
```

Task 2 是核心——提取纯逻辑并测试。Task 3 在此基础上构建 UI，只需关注展示和交互。
