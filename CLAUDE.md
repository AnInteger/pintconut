# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pintconut** (拼豆差异检测器) — compares photos of completed bead board artwork against blueprint designs, automatically detecting and annotating misplaced beads using computer vision.

Pipeline: Photo + Blueprint → Bead Detection (YOLO, single-class "bead") → Grid Fitting → Color Matching (LAB space, 221-color palette) → Diff Annotation

> **Current state**: production `cli.py` still uses HoughCircles (`BeadGridFitter` without a detector); wiring the YOLO `BeadDetector` into production is planned (see spec 2026-07-10).

---

## 项目宪法（关键决策与已验证结论）

> brainstorming 讨论确认，2026-07-10。避免重复踩坑。

### 检测路线
- **生产目标 = YOLO 目标检测**（单类 bead）。`BeadDetector`（`src/bead_detect.py`）是计划中的生产检测器——**有测试，勿当死代码删**。
- YOLO 模型目前只用在工具：`bead_tune.py`（调 conf/iou）、`bead_annotate_mpl.py` 的 `f` 键 SAHI 预填。

### 已验证无效，不要再试
以下"自动检测辅助标注/检测"路线在本项目照片（光照/反光/透视复杂）上实测不灵：
- 几何网格拟合当检测器（HoughCircles 找豆 + lattice 编号）
- 四角网格批量生成标注框
- 梯度（Sobel）自动定半径
- HoughCircles 预标 + 空洞补洞

### 精度天花板 = 数据
- `board30` 训练日志：val mAP 全程 = 0、`cls_loss` 多次飙到 35-87 → 模型没学进去 + 训练不稳定。
- 根因在数据质与量；量被标注速度锁死；速度被工具卡顿锁死。
- 杠杆链：修工具性能 → 标得快 → 产可信数据 → 诊断/修 mAP=0 → 再谈 YOLO11/调参/接线。

### 评测准则
- 在**可信 GT** 上用标准 **mAP50**（mAP50-95 参考）。
- **不引入自定义计数指标**（曾提议用 |预测数−真实数| 绕 GT 噪声，判定 hack，否决）。
- 正路是产可信 GT（完整/一致/贴边标注），让标准 mAP 本身可信。

### 训练准则（有 GPU）
- `imgsz 1280`、`epochs 300` + `patience`、`single_cls=True`、cos-lr、`close_mosaic` 开。
- 盯 `cls_loss`：飙到几十 = 发散（查 lr / 0 实例 batch / 标签格式）。

### 数据路径准则
- `data.yaml` 用**相对路径**，勿写死绝对路径（曾因仓库迁移失效）。

## Common Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests (pytest)
python -m pytest tests/ -v
python -m pytest tests/test_color.py -v          # single test file
python -m pytest tests/test_grid.py::test_extract_2x2 -v  # single test

# Run the labeling web UI (Gradio)
python src/label_ui.py                            # http://localhost:7860

# Run the bead annotation UI (for bead-level training data)
python src/bead_annotate_ui.py                    # http://localhost:7860

# CLI diff detection
python -m src.cli --photo photo.jpg --blueprint blueprint.png --board-size 29x29

# Training pipeline
python training/semi_auto_label.py --input training/photos --output training/dataset  # terminal labeling
python training/train.py --data training/dataset/data.yaml                           # train board detector
python training/validate.py --model training/runs/beadboard-v1/weights/best.pt       # validate model

# Bead-level training
python training/bead_train.py --data training/bead_dataset/data.yaml
python training/bead_validate.py --model <model_path>
```

## Architecture

```
src/
├── bead_grid.py        # BeadGridFitter — 豆子检测 + 网格编号（取代板子检测）
├── cli.py              # CLI entry point — orchestrates full detection pipeline
├── grid.py             # PerspectiveCorrector + GridExtractor — warp + sample colors per cell
├── color.py            # ColorMatcher — 221-color palette matching in LAB space + bead presence check
├── blueprint.py        # parse_blueprint — extract color grid from blueprint image
├── compare.py          # DiffComparator — grid comparison + diff annotation
├── bead_detect.py      # BeadDetector — YOLOv8n individual bead detection + filtering
├── bead_prelabel.py    # HoughCircles pre-labeling for bead training data
├── label_service.py    # Core labeling logic: FastSAM segmentation, candidate drawing, YOLO label export
└── label_ui.py         # Gradio 2-tab wizard UI (Upload → Annotate/Confirm)

training/
├── semi_auto_label.py  # Terminal-based semi-auto labeling tool (FastSAM)
├── train.py            # Board detection model training
├── validate.py         # Model validation
├── bead_train.py       # Bead detection model training
├── bead_validate.py    # Bead model validation
└── collect_helper.py   # Batch photo renaming

data/
├── colors.json         # 20-color palette (id, name, rgb) — expandable to 221
└── board_sizes.json    # 6 board size presets (14×14 to 29×58)
```

### Key Design Patterns

- **Lazy model loading**: `BeadDetector` and the label UI load YOLO/FastSAM models on first use, not at import time (model files are `.gitignore`d except `FastSAM-s.pt`)
- **Module-level singletons**: Label UI uses module-level dicts (`_segmentation_cache`, `_annotation_state`, `_pending_selection`) for Gradio state across callbacks
- **Board-via-beads**: Board localization is derived directly from the detected bead grid (BeadGridFitter) — no separate board-detection model. Bead-level detection (BeadDetector) remains for per-bead tasks.
- **LAB color space**: All color matching uses CIE LAB via `cv2.cvtColor(..., COLOR_RGB2Lab)` for perceptual accuracy
- **YOLO format**: Training labels use YOLO segmentation format (class x1 y1 x2 y2...) for boards, detection format (class cx cy w h) for beads

### Data Flow

1. **Bead grid fitting**: Image → `BeadGridFitter.fit()` → `GridResult` (cells with color, outline, confidence)
2. **Color matching**: Per-cell RGB → `ColorMatcher.match()` → palette names; `is_bead()` → presence check
3. **Diff comparison**: Photo grid vs blueprint grid → `DiffComparator.compare()` → list of mismatched cells

## Testing

- **Unit tests**: `tests/test_*.py` using pytest with mock fixtures (no model files needed)
- **Integration tests**: `tests/test_label_ui_integration.py` uses Playwright against a live Gradio server (port 7899)
- **Test fixtures**: `tests/fixtures/test_blueprint_5x5.png` — small grid for blueprint parsing tests
- **conftest.py**: Manages Gradio server lifecycle for Playwright tests (start, wait, teardown)

## Important Notes

- Model weight files (`*.pt`) are gitignored except `FastSAM-s.pt` — models must be trained before CLI use
- `training/` outputs (dataset/, runs/, photos/) are gitignored
- The project uses Python 3.10+ features (`tuple[int, int]` type hints, `|` union types)
- Images flow through the pipeline in OpenCV's **BGR** format; conversion to RGB happens only for color matching and display
- Gradio UI text and CLI output are in Chinese (中文)
