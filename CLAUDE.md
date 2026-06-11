# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Pintconut** (拼豆差异检测器) — compares photos of completed bead board artwork against blueprint designs, automatically detecting and annotating misplaced beads using computer vision.

Pipeline: Photo + Blueprint → Board Detection (YOLOv8-seg) → Perspective Correction → Grid Extraction → Color Matching (LAB space, 221-color palette) → Diff Annotation

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
python -m src.cli --photo photo.jpg --blueprint blueprint.png --board-size 29x29 --model models/beadboard-best.pt

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
├── cli.py              # CLI entry point — orchestrates full detection pipeline
├── detect.py           # BoardDetector — YOLOv8-seg board detection + corner extraction
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

- **Lazy model loading**: `BoardDetector`, `BeadDetector`, and the label UI load YOLO/FastSAM models on first use, not at import time (model files are `.gitignore`d except `FastSAM-s.pt`)
- **Module-level singletons**: Label UI uses module-level dicts (`_segmentation_cache`, `_annotation_state`, `_pending_selection`) for Gradio state across callbacks
- **Two-stage detection**: Board-level (YOLOv8-seg) then bead-level (YOLOv8n), each requiring separately trained models
- **LAB color space**: All color matching uses CIE LAB via `cv2.cvtColor(..., COLOR_RGB2Lab)` for perceptual accuracy
- **YOLO format**: Training labels use YOLO segmentation format (class x1 y1 x2 y2...) for boards, detection format (class cx cy w h) for beads

### Data Flow

1. **Board detection**: Image → `BoardDetector.detect()` → binary mask → `extract_corners()` → 4 corner points
2. **Perspective correction**: Image + corners → `PerspectiveCorrector.correct()` → warped board image
3. **Grid extraction**: Warped image + (rows, cols) → `GridExtractor.extract()` → (rows × cols × 3) RGB array
4. **Color matching**: RGB grid → `ColorMatcher.match()` → palette names; `is_bead()` → presence check
5. **Diff comparison**: Photo grid vs blueprint grid → `DiffComparator.compare()` → list of mismatched cells

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
