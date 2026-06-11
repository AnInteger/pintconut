# Obsidian Documentation for Pintconut — Design Spec

**Date**: 2026-06-11
**Status**: Approved

## Goal

Create 4 comprehensive Obsidian-flavored documentation files for Pintconut, targeting An and AI Agents as primary readers. Documentation focuses on current project state (no roadmap/history).

## Structure

```
docs/obs/
├── product/
│   ├── product-overview.md       # 功能概述 + 使用指南
│   └── labeling-tool-guide.md    # 标注工具用户手册
└── technical/
    ├── architecture.md           # 架构设计
    └── training-pipeline.md      # 训练流水线
```

## Obsidian Conventions

- **Frontmatter**: `title`, `tags`, `date`, `status` on every file
- **Wikilinks**: `[[filename]]` for cross-references between docs
- **Callouts**: `> [!info]`, `> [!warning]`, `> [!tip]` for emphasis
- **Mermaid**: Architecture diagrams and flowcharts
- **Filenames**: English, kebab-case

## Document Contents

### product/product-overview.md
- One-line description + core workflow (Mermaid pipeline diagram)
- CLI usage with all parameters
- Board size specifications table
- Color palette overview → links to [[architecture#颜色系统]]

### product/labeling-tool-guide.md
- Gradio Web UI complete operation guide
- Tab ① Upload: steps and behavior
- Tab ② Annotate: select → preview → confirm/reselect/skip flow
- Button reference (confirm, reselect, skip, export)
- Dataset output format → links to [[training-pipeline#数据集格式]]
- Troubleshooting callouts

### technical/architecture.md
- System module relationship diagram (Mermaid)
- Per-module breakdown: BoardDetector, PerspectiveCorrector, GridExtractor, ColorMatcher, DiffComparator, BeadDetector
- Complete data flow (Mermaid sequence)
- Color system: LAB space, matching algorithm, is_bead logic
- Design patterns: lazy model loading, module-level singletons, BGR/RGB conventions
- Cross-links to other docs

### technical/training-pipeline.md
- Two independent training tracks: board (YOLOv8-seg) + bead (YOLOv8n)
- Board training: semi_auto_label.py → train.py → validate.py
- Bead training: HoughCircles prelabel → bead_annotate_ui.py → bead_train.py → bead_validate.py
- Dataset formats: YOLO Segmentation vs YOLO Detection
- File structure conventions under training/
- Links to [[labeling-tool-guide]] and [[architecture]]

## Sources
- Source code: all files under `src/`, `training/`, `data/`
- Git history: 30 recent commits
- Existing docs: `docs/superpowers/specs/` (6 design specs)
- README.md
