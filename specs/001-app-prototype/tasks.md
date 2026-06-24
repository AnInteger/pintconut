---
description: "Task list for Pintcount app prototype (v1.4 PRD)"
---

# Tasks: Pintcount App 原型设计（v1.4 PRD）

**Input**: Design documents from `/specs/001-app-prototype/`（plan.md, spec.md, data-model.md, research.md, quickstart.md）

**Prerequisites**: plan.md ✅、spec.md ✅、data-model.md ✅（屏与状态清单）、research.md ✅、quickstart.md ✅

**Tests**: 无自动化测试（静态原型；走查见 quickstart.md）。未要求 TDD，故不生成 test 任务。

**Organization**: 按用户故事分组（US1 核心魔法流 / US2 状态与边界 / US3 作品库与设置），每故事可独立实现与走查。

## Format: `[ID] [P?] [Story] Description`

- **[P]**：可并行（不同文件、无未完成依赖）
- **[Story]**：所属用户故事（US1/US2/US3）；Setup/Foundational/Polish 阶段无故事标签
- 描述含确切文件路径；逐屏内容详见 [data-model.md](./data-model.md)

## Path Conventions

静态原型，所有产物在 `docs/mockups/`：
- `docs/mockups/styles.css` — 共享设计令牌 + 公共组件
- `docs/mockups/index.html` — 导航枢纽
- `docs/mockups/<screen>.html` — 各屏/态

---

## Phase 1: Setup（清场）

**Purpose**: 移除与 v1.4 PRD 脱节的旧 mockups，为新稿让位（FR-009：替换 docs/mockups/，旧版 git 历史保留）。

- [ ] T001 删除旧 mockup 文件：`docs/mockups/visual-style.html`、`flow-storyboard.html`、`flow-storyboard-v2.html`、`real-scale-screens.html`、`real-scale-v2.html`、`real-scale-v3.html`、`index.html`（旧索引）

---

## Phase 2: Foundational（阻塞前置）

**Purpose**: 所有屏都依赖的共享样式，MUST 先完成。

- [ ] T002 创建 `docs/mockups/styles.css`：承载设计令牌（色板/字号/间距/圆角/阴影，见 data-model.md「设计令牌」）+ 公共组件类——`.phone390` 手机框、status/nav/tab 栏、主按钮(`.btn-primary` 渐变)/深色/文字按钮、`.pill` 胶囊、豆子马赛克纹理(`.beads`/`.b1`/`.b2`/`.b3`)、**多色错误标记**(`.mark-color` 红 `#EF4444`/`.mark-missing` 蓝 `#38BDF8`/`.mark-extra` 橙 `#F59E0B`，各含编号圆 + 字形)、`.soft-panel` 软底面板。无 emoji，图标纯 CSS 几何形。

---

## Phase 3: User Story 1 — 核心质检魔法流（Priority P1）

**Story goal**: 评审者能沿 `首页→选图纸→拍照→检测中→结果→分享` 走通完整 happy path，理解产品核心价值。
**Independent test**: 浏览器依次打开 home→blueprint→capture→processing→results→share，能完整理解「拍一张→对照图纸→标出错豆→分享」。
**Depends on**: T002。各屏互不依赖 → 全部 `[P]`。

- [ ] T003 [P] [US1] `home.html`：Status + 品牌 lockup + hero（`拼豆质检` + 副文案）+ `Count` 主按钮 + 最近作品横滑（马赛克缩略 + 结果徽章）+ tab bar（首页选中）
- [ ] T004 [P] [US1] `blueprint.html`：导航「图纸」+「换一张」+ 已解析图纸马赛克大图 + 尺寸徽章「29×29」+ 「N 种颜色」+ **可选色盘**（Perler 选中）+ **备豆统计**软底面板（色块+名称+数量）+ 色块条 + 主按钮「用这张图纸去拍照」
- [ ] T005 [P] [US1] `capture.html`：全屏深色取景器 + 四角粉色角标 + 「已锁定拼板」胶囊 + 底部快门（渐变圆）+「从相册选」+「翻转」
- [ ] T006 [P] [US1] `processing.html`：居中加载环 +「正在逐格比对」+ **用户语言步骤**（对齐拼板/读取每格/比对颜色，完成绿点）。**禁止** CV/流水线术语（FR-008）
- [ ] T007 [P] [US1] `results.html`（有错）：导航「检测结果」+「重扫」+ 成品图（马赛克）+ **多色编号标记**（①②红=颜色错、③蓝=空缺、④橙=多余）+ 结论胶囊「4 处错误」+ 小结 + 错误索引软底面板 + 底栏（**无「保存」**；重扫次级 + 分享主）（FR-003/FR-005）
- [ ] T008 [P] [US1] `results-pass.html`（通过 0 错）：结论胶囊 `success` 绿「质检通过」+ 净图（无标记）+ 底栏「分享」主
- [ ] T009 [P] [US1] `share.html`：卡片预览（成品缩略 + 结论角标 +「质检完成」+ 尺寸/色盘 +「用 Pintcount 一键质检」水印）+「分享到」+ 系统分享入口行（占位）

---

## Phase 4: User Story 2 — 状态与边界（Priority P2）

**Story goal**: 体现「魔法优先、不靠手动补救」——所有失败/边界态只有重试/提示，无手动校正工具。
**Independent test**: 逐个打开 4 个边界帧，确认**无手动逃生舱/无手动框选**，只有重试或明确提示。
**Depends on**: T002。各帧 `[P]`。

- [ ] T010 [P] [US2] `first-run.html`：开 App → 相机/相册权限请求卡（说明用途）→ 授权后进首页（FR-006，PRD §5.3）
- [ ] T011 [P] [US2] `blueprint-lowconf.html`：图纸区 + 温和提示「图纸不太清晰，建议重选/重截」+「重选图纸」。**无手动框选工具**（FR-004）
- [ ] T012 [P] [US2] `unsupported.html`：明确提示「这类图纸/拍摄场景暂不支持，建议换一张/重拍」+ 重选/重拍。**不出现手动逃生舱**（FR-004）
- [ ] T013 [P] [US2] `detection-failed.html`：「没认出来」+ 重拍 + 打光/角度建议。**不强制手动校正**（FR-004）

---

## Phase 5: User Story 3 — 作品库与设置（Priority P3）

**Story goal**: 补全三屏 IA，体现自动保存与重扫。
**Independent test**: 作品库可见自动保存的作品；作品详情可触发重扫（跳过图纸）；设置可切色盘。
**Depends on**: T002。各帧 `[P]`。

- [ ] T014 [P] [US3] `gallery.html`：导航「作品库」+ 作品卡（马赛克缩略 + 结果徽章 + 名称·尺寸 + 时间）+ tab bar（作品库选中）。体现**自动保存**（FR-005）
- [ ] T015 [P] [US3] `gallery-detail.html`：成品照 + 历次检测记录（时间线：错误数/通过 + 时间）+「重扫」按钮（跳过图纸直接进拍照）
- [ ] T016 [P] [US3] `settings.html`：分组列表——色盘品牌（Perler/Hama/Artkal 选中态高亮）+ 关于 + 隐私说明 + tab bar（设置选中）

---

## Phase 6: Polish & Cross-Cutting（收口）

**Purpose**: 串联导航 + 全局一致性 + 走查。

- [ ] T017 创建 `docs/mockups/index.html` 导航枢纽：分组卡片链接全部 14 帧（**核心魔法流**标注走查顺序 / **作品库·设置** / **边界态**）+ 顶部品牌 lockup + 一句说明
- [ ] T018 全局一致性巡检：共享令牌统一使用、各屏 tab bar 选中态正确、无 emoji、文案去「AI 味」且无 CV 术语、多色标记三类一致、结果页**无「保存」按钮**（自动保存）
- [ ] T019 按 quickstart.md 验收清单做最终视觉走查，修补缺口

---

## Dependencies（故事完成顺序）

```
T001 (清场) → T002 (styles.css) ─┬→ US1 (T003–T009) ─┐
                                  ├→ US2 (T010–T013)  ─┼→ T017 (index) → T018 → T019
                                  └→ US3 (T014–T016) ─┘
```

- US1 / US2 / US3 三个故事**相互独立**，均只依赖 T002，可任意顺序或并行。
- T017（index）依赖所有帧完成；T018/T019 依赖 T017。

## Parallel Execution

T002 完成后，**T003–T016 共 14 个帧任务全部可并行**（各为独立文件，仅共享 styles.css）。建议分 3 批对应 3 个故事，或一次铺开。

## Implementation Strategy（MVP 优先）

- **MVP = Phase 2 + US1**（T002 + T003–T009）：完成后即可走通核心魔法流，交付产品核心价值。
- US2（边界态）、US3（作品库/设置）为增量，可后续补。
- Polish（T017–T019）最后收口。
