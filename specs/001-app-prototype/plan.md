# Implementation Plan: Pintcount App 原型设计（v1.4 PRD）

**Branch**: `001-app-prototype` | **Date**: 2026-06-24 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-app-prototype/spec.md`

## Summary

基于 v1.4 PRD 产出 Pintcount 移动端 APP 的**静态 HTML 原型**：用既有设计令牌（2026-06-22 原型方向 C）绘制核心魔法流六屏（详细）+ 其余屏/边界态各一帧，加一个 index 导航枢纽。体现 v1.4 全部修订决策（合并流、de-tech、备豆统计、解析选色盘、去逃生舱、自动保存、首启权限、多色错误标记、通过态）。产物替换 `docs/mockups/`。

技术取向（见 research.md）：纯 HTML/CSS，共享 `styles.css` 承载设计令牌与公共组件（手机框/tab 栏/按钮/豆子马赛克纹理），每屏一个 HTML；多色错误标记映射到既有令牌（颜色错=红 `error`、空缺=蓝 `info`、多余=橙 `warn`），零新增颜色。

## Technical Context

**Language/Version**: HTML5 + CSS3；无框架。原生 JS 仅用于 index 导航高亮（可选，非必需）。

**Primary Dependencies**: 无（纯静态资源）。

**Storage**: N/A（静态视觉稿，无数据持久；所有内容为假数据）。

**Testing**: 人工视觉走查——浏览器打开 `docs/mockups/index.html`，按 [quickstart.md](./quickstart.md) 清单逐屏核对；无自动化测试。

**Target Platform**: 现代桌面浏览器（Chrome / Safari / Edge）；按 iPhone 390pt 逻辑分辨率绘制手机外框（沿用既有 `.phone390`）。

**Project Type**: static-mockup / prototype。

**Performance Goals**: N/A（静态页，本地秒开）。

**Constraints**:
- 复用既有设计令牌（`docs/superpowers/specs/2026-06-22-pintcount-prototype-design.md` §3：色板 / 字号 / 间距 4·8·12·16·20·24·32 / 圆角 / 阴影）。
- 全程无 emoji；图标用纯 CSS 几何形。
- 结果页去边框（软底面板 + 留白分组）。
- 多色错误标记：颜色错 `#EF4444`、空缺 `#38BDF8`、多余 `#F59E0B`，每处配编号 + 字形保色弱可读。
- 产物替换 `docs/mockups/`（旧版由 git 历史保留）。
- 用用户语言，不出现检测流水线 / CV 术语（与 PRD de-tech 一致）。

**Scale/Scope**: 14 屏/态 + 1 index + 1 共享 `styles.css`（共 16 个文件）。详见 [data-model.md](./data-model.md)。

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

`.specify/memory/constitution.md` 为**未填写的模板**（`[PROJECT_NAME]` / `[PRINCIPLE_1_NAME]` 等占位未填），无实际项目原则或治理约束 → **无 gate 适用**，通过。

## Project Structure

### Documentation (this feature)

```text
specs/001-app-prototype/
├── spec.md              # /speckit-specify 产物
├── plan.md              # 本文件（/speckit-plan 产物）
├── research.md          # Phase 0：决策记录（无 NEEDS CLARIFICATION）
├── data-model.md        # Phase 1：屏与状态清单（Screen & State Inventory）
├── quickstart.md        # Phase 1：走查验收指南
└── tasks.md             # /speckit-tasks 产物（本命令不创建）
```

### Source Code (docs/mockups/ — 替换既有)

```text
docs/mockups/
├── styles.css              # 共享：设计令牌 + 公共组件（手机框/tab栏/按钮/豆子马赛克/多色标记）
├── index.html              # 导航枢纽：分组卡片链接到每个 frame
├── home.html               # 首页（核心流·详细）
├── blueprint.html          # 选图纸·解析成功（核心流·详细）
├── capture.html            # 拍照（核心流·详细）
├── processing.html         # 检测中（核心流·详细）
├── results.html            # 结果·有错（核心流·详细）
├── results-pass.html       # 结果·通过 0 错（核心流·详细）
├── share.html              # 分享卡片（核心流·详细）
├── gallery.html            # 作品库列表（简略·一帧）
├── gallery-detail.html     # 作品详情 + 重扫（简略·一帧）
├── settings.html           # 设置·色盘品牌（简略·一帧）
├── first-run.html          # 首启权限（边界·一帧）
├── blueprint-lowconf.html  # 解析低置信度（边界·一帧）
├── unsupported.html        # 不支持场景（边界·一帧）
└── detection-failed.html   # 检测失败（边界·一帧）
```

**Structure Decision**: 共享 `styles.css`（令牌 + 公共组件）+ 每屏独立 HTML 引用之 + `index.html` 导航枢纽。既有 `docs/mockups/*.html` 为各自内联样式；改用共享 css 是因为本次 16 文件共享令牌，集中维护优于 16 份内联重复（见 research.md D3）。

## Complexity Tracking

无 Constitution 违规，无需填表。
