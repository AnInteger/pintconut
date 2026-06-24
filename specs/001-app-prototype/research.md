# Research — Pintcount 原型（Phase 0）

> spec 无 NEEDS CLARIFICATION（已在 /speckit-specify + /speckit-clarify 解决）。本文件记录影响实现的关键设计决策。

## D1：复用 2026-06-22 设计令牌
- **Decision**：色板 / 字号 / 间距 / 圆角 / 阴影 / 手机框 / 豆子马赛克纹理全部复用既有原型（方向 C）。
- **Rationale**：spec FR-007 + 用户「保持设计风格」；既有令牌已成熟。
- **Alternatives**：全新令牌——与既有 mockups 割裂，否决。

## D2：多色错误标记映射到既有令牌
- **Decision**：颜色错 = `#EF4444`（=error）、空缺 = `#38BDF8`（=info）、多余 = `#F59E0B`（=warn）。
- **Rationale**：零新增颜色，风格一致；三色在色相轮上区分强（红/蓝/橙）；颜色错沿用既有红、多余沿用既有橙，改动最小。每处配编号 + 字形保色弱可读（SC-005）。
- **Alternatives**：新引 3 色（如紫/青/黄）——增加调色板复杂度、与品牌色冲突，否决。

## D3：共享 styles.css + 每屏 HTML
- **Decision**：1 个 `styles.css`（令牌 + 公共组件：手机框 / tab 栏 / 按钮 / 马赛克 / 多色标记）+ 每屏 HTML 引用 + `index.html` 导航。
- **Rationale**：16 文件共享令牌，集中维护；改一处令牌全量生效，优于既有「每文件内联」在 16 文件下的重复。
- **Alternatives**：每文件内联（既有模式）——16 份重复、令牌漂移风险，否决。

## D4：结果「通过」态单独成帧
- **Decision**：`results.html`（有错）+ `results-pass.html`（通过 0 错）两帧。
- **Rationale**：通过态是重要情感时刻 + FR-003 / SC 要求；与有错合并信息密度过高。
- **Alternatives**：合并为一帧双态——评审时易混淆，否决。

## D5：无自动构建 / 无依赖
- **Decision**：纯静态 HTML/CSS，无 build step、无 JS 框架（index 导航可用最小原生 JS 或纯链接）。
- **Rationale**：原型只求浏览器可开可看；引入构建链徒增复杂。
- **Alternatives**：Vite / 静态站生成器——过度工程，否决。

## 无 NEEDS CLARIFICATION 残留
spec 经 specify（FR-010/011/012）+ clarify（产物落点 / 简略屏深度）已全部澄清，本 Phase 0 无悬而未决项。
