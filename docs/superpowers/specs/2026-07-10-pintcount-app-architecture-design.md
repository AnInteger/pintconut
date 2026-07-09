# Pintcount APP 架构设计文档

> 日期：2026-07-10
> 状态：草案（待评审）
> 版本：v0.1
> 分支：`feat/pintcount-app`
> 文档类型：架构设计文档（Architecture Design Spec）
> 关联：
> - 产品需求：[`2026-06-18-pintcount-app-prd.md`](./2026-06-18-pintcount-app-prd.md)（v1.4）
> - 原型设计：[`2026-06-22-pintcount-prototype-design.md`](./2026-06-22-pintcount-prototype-design.md)（v0.1）

---

## 0. 文档目的与范围

把 PRD + 原型设计翻译成**可落地的工程架构**：分层、模块边界、数据流、错误处理、测试策略（TDD + Playwright）、技术栈、仓库结构、分阶段路线。

本 spec 写**全局架构 + 测试 + 路线**（"怎么搭"）。详细实现步骤由后续 `writing-plans` 阶段产出，且**只展开 Phase 1（垂直切片）**；Phase 2–4 各自再出计划，不一上来规划全部。

**核心架构决策（已与用户确认）**：检测流水线采用 **端侧 TS 核心 + ONNX Runtime** 方案（代号 **A1**）。完全符合 PRD「端侧、无后端、照片不出手机」。

---

## 1. 背景与约束

### 1.1 PRD 硬约束

- **端侧、无后端**：照片、图纸、作品记录全本地，不上传、不收集。隐私是卖点 + 零服务器成本。
- **跨平台**：React Native 写一份代码，**iOS 先发布、Android 随后**。
- **一键魔法流**：选图纸 → 拍成品照 → 自动检测 → 看差异 → 分享。默认全自动，**零手动干预**（不调角点、不填尺寸）。
- **魔法优先，无手动逃生舱**：失败只引导重试 / 提示不支持，绝不引入手动校正 UI。
- **作品自动入库**：检测完成自动保存，结果页只需"重扫 / 分享"。

### 1.2 现有 Python 流水线的角色

仓库 `src/*.py`（~1900 行）是算法的**蓝本**，APP 全新用 TypeScript 实现，**不移植代码本体**。逐模块对应关系：

| Python 模块 | 职责 | 移植到 TS |
|---|---|---|
| `color.py` | LAB 配色匹配 | 纯逻辑（只需 sRGB→LAB） |
| `blueprint.py` | 图纸 → RGB 网格 | 纯逻辑 |
| `compare.py` | 差异比对（LAB 距离） | 纯逻辑（画框交给 UI 层） |
| `bead_grid.py` | 豆框 → 网格拟合（向量投票 / 仿射 / 同伦） | 纯线性代数（配矩阵库做 SVD） |
| `bead_detect.py` | YOLO 豆子检测 | **唯一原生依赖** → ONNX Runtime RN |
| `grid.py`（老路径 warpPerspective） | 透视矫正 | **bead-grid 路径不用**，APP 不需要 |

**关键发现**：APP 走的 bead-grid 路径**完全不依赖 OpenCV 重算子**——~70% 是纯逻辑（可直接 TS，Jest 可测、浏览器可跑），唯一原生依赖是 YOLO 推理（ONNX）。FastSAM 只用在开发标注工具，**不进 APP**。这正是 A1 方案干净且可行的根本原因。

---

## 2. 目标与非目标

### 目标

- 一套可 TDD 的纯 TS 检测核心，算法与现有 Python 逐函数对拍一致。
- 一份 RN 代码出 iOS / Android / Web（Web 用于 Playwright "模拟测试"）。
- 失败处理是可测的纯逻辑，UI 只渲染状态。
- 垂直切片（P1）用最小表面积验证整套架构 + 测试链路。

### 非目标（本 spec 不覆盖）

- 真机 ONNX 推理的性能调优（P4）。
- App Store / Play 上架细节（P4）。
- v2 功能（自定义色盘、多分享模板、云备份、深色模式、图纸社区）。
- Phase 2–4 的逐步实现计划（各自后续出 plan）。

---

## 3. 架构：分层 + 依赖倒置

```
┌──────────────────────────────────────────────────────────┐
│  UI 层  (React Native + Expo Router)                      │
│  screens/ · components/ · tokens/ (取自原型设计稿)          │
└────────────────────┬─────────────────────────────────────┘
                     │ 依赖（平台无关）
                     ▼
┌──────────────────────────────────────────────────────────┐
│  应用层  (流程编排 / 状态)                                  │
│  flows/ 魔法流状态机 · stores/ 作品库·设置                   │
│  —— 纯 TS，不碰 React 原生，可 Jest 测                       │
└────────────────────┬─────────────────────────────────────┘
                     │ 依赖
                     ▼
┌──────────────────────────────────────────────────────────┐
│  检测核心  (纯 TS)  ★ TDD 主战场                             │
│  color/ · blueprint/ · grid/ · compare/ · pipeline/        │
│  —— 零平台依赖 → Node(Jest) 和浏览器都能跑                   │
└────────────────────┬─────────────────────────────────────┘
                     │ 依赖倒置（只认接口）
                     ▼
┌──────────────────────────────────────────────────────────┐
│  平台桥  (接口 ports + 两套实现)                             │
│  IVisionPort.detectBeads(img) → DetectedBead[]             │
│  ICameraPort · IStoragePort · ISharePort · IImagePort      │
│   ├─ native 实现: ONNX RN / expo-camera / 存储 / 分享        │
│   └─ mock 实现:  fixture 豆框 / 假存储 (Jest + Playwright)   │
└──────────────────────────────────────────────────────────┘
```

### 3.1 承重决策：平台边界用依赖倒置

检测核心和应用层只依赖**接口（port）**，不直接调用原生 API。

- 真机上：`native/` 适配器实现 ports（ONNX 推理、相机、本地存储、系统分享）。
- 测试 / Web 上：`mock/` 适配器用 fixture 实现 ports。

**这是 TDD（核心纯 TS 跑 Jest）和 Playwright（Web 上 mock 原生驱动整条魔法流）能干净成立的关键。** UI / 应用 / 核心三层完全平台无关，只有平台桥层分两套实现。

### 3.2 唯一最硬的接缝：`IVisionPort.detectBeads()`

它之上（网格拟合 / 配色 / 差异）全是纯 TS；它之下（YOLO 推理）是原生 ONNX。契约：

```ts
interface IVisionPort {
  detectBeads(image: ImageInput): Promise<DetectedBead[]>;
}
interface DetectedBead { cx: number; cy: number; w: number; h: number; conf: number; }
```

对应现有 `bead_grid.py` 的 `fit(detector=...)` 接缝——检测器可插拔。真机注入 ONNX 适配器，测试注入返回 fixture 豆框的 mock。

### 3.3 其它 ports

```ts
interface ICameraPort  { capture(): Promise<ImageInput>; pickFromGallery(): Promise<ImageInput | null>; }
interface IImagePort    { decode(path): Promise<PixelImage>; sampleColor(img, x, y, half): RGB; medianColor(img, region): RGB; }
interface IStoragePort  { loadWorks(): Promise<Work[]>; saveWork(w: Work): Promise<void>; loadSettings(): Promise<Settings>; saveSettings(s): Promise<void>; }
interface ISharePort    { share(card: ShareCard): Promise<void>; saveToGallery(img): Promise<void>; }
```

`IImagePort` 把"像素访问 / 中位数采样"隔离成接口（真机用 expo 图像 API，测试/Web 用 canvas / fixture）。

---

## 4. 模块边界（每个单元单一职责，可独立测）

### 检测核心 `app/core/`（纯 TS）

| 模块 | 职责 | 对外接口（设计契约） |
|---|---|---|
| `color/` | 色盘 + sRGB→LAB + 最近邻匹配 + `isBead` | `ColorMatcher.match(rgb): PaletteColor`、`isBead(rgb): boolean` |
| `blueprint/` | 图纸图 → RGB 网格 + 行列/颜色数/备豆统计 | `parseBlueprint(image, opts): BlueprintResult` |
| `grid/` | 豆框 → 轴向量投票 → 仿射/同伦标注 → cells | `GridFitter.fit(beads, boardSize?): GridResult` |
| `compare/` | 成品 cells vs 图纸网格 → `DiffResult[]` | `compareDiff(photoCells, blueprintGrid): DiffResult[]` |
| `pipeline/` | 编排上面，含失败判定 | `detect(photo, blueprint, palette, ports): DetectionResult` |

### 应用层 `app/app/`

- `flows/`：魔法流状态机（选图纸 → 拍照 → 检测中 → 结果 → 分享），含失败分流（见 §6）。
- `stores/`：作品库（自动存）、设置（色盘品牌）。

### 平台层 `app/platform/`

- `ports.ts`：接口定义（见 §3.2 / §3.3）。
- `native/`：ONNX（`onnxruntime-react-native`）+ `expo-camera` + 存储 + `expo-sharing` 适配器。
- `mock/`：fixture 适配器（Jest + Playwright 共用）。

### UI 层 `app/ui/`

- `screens/`：原型 8 屏（首页 / 选图纸 / 拍照 / 检测中 / 结果 / 分享 / 作品库 / 设置）+ 状态变体。
- `components/`：设计令牌组件（原型 §4：主按钮、编号红框、错误标记、卡片、栏）。
- `tokens/`：设计令牌（原型 §3：色板 / 字号 / 间距 / 圆角 / 阴影）。

---

## 5. 数据流（魔法流穿过各层）

```
[选图纸] image ──BlueprintParser──► BlueprintResult
                                   { rows, cols, cells: RGB[], colors[], beadCounts }
                                   + 选色盘（默认沿用上次）

[拍照]   image ──IVisionPort.detectBeads──► DetectedBead[]   (真机 ONNX / 测试 fixture)
              ──GridFitter.fit──► GridResult
                                   { rows, cols, cells: CellInfo[]（按豆位采样上色）,
                                     confidence: GridConfidence, truncation }

[检测]   cells ──ColorMatcher.match──► 色号
         ──compareDiff(photoCells, blueprintGrid)──► DiffResult[]
           ( 颜色错 / 空缺 / 多余 + 每格 is_reliable )

[判定]   DetectionResult + confidence ──decideFlowState──► FlowState
           happy → 结果屏 | 低置信 → 重试提示 | 反复失败 → 不支持提示

[结果]   FlowState.happy ──► 结果屏（编号红框 + 错误索引 + 计数；通过态=净图）
[分享]   ──► 分享卡片 ──ISharePort.share()──► 系统 Share Sheet
[作品库] ──► IStoragePort.saveWork()（自动落本地存储）
```

算法层与现有 Python 流水线**逐模块一一对应**，只是用 TS 表达 + 把 ONNX 接缝隔离出来。图纸随作品保留，重扫跳过图纸直接进 [拍照]。

---

## 6. 错误处理（魔法优先，不靠手动补救）

遵循 PRD §4.5 / 原型 §8 的铁律：**默认全自动，失败只引导重试 / 提示不支持，绝不引入手动校正 UI。**

### 6.1 失败判定是纯函数，UI 只渲染状态

每层吐出带置信度的类型化结果，应用层状态机据此分流：

| 信号源 | 置信度字段（Python 里已在算） |
|---|---|
| `BlueprintParser` | 解析置信度（网格检出？颜色数合理？） |
| `GridFitter` | `GridConfidence`（填充率 + 标注残差 → 高/中/低） |
| `DiffComparator` | 每格 `is_reliable`（≥ 0.8） |

分流为纯函数 `decideFlowState(result, confidence, retryCount): FlowState`，可单测：

| FlowState | 触发 | UI 表现 |
|---|---|---|
| `happy` | 置信度达标 | 结果屏（有错 / 通过两态） |
| `retry_blueprint` | 图纸低置信（可恢复） | "图纸不太清晰，建议重选 / 重截"（**无**手动框选） |
| `retry_capture` | 识别困难 / 检测失败 | 重拍 + 打光 / 角度引导（**无**手动校正） |
| `unsupported` | 反复失败 | "这类图纸 / 拍摄场景暂不支持"（**无**逃生舱） |
| `permission_needed` | 权限拒绝 | 引导去系统设置开启 |

### 6.2 强制约束

**组件库里不提供任何手动校正控件**（架构层面杜绝），保证"魔法流"定位不被破坏。`decideFlowState` 的所有分支都不产出"进入手动模式"的状态。

---

## 7. 测试策略（TDD + Playwright）

### 7.1 单元 — Jest，TDD 主战场

检测核心每个函数 TDD（先写失败测试再实现）。两类 fixture：

- **合成 fixture**：构造的网格 / 豆位（同 Python `test_bead_grid` 的合成夹具思路），测算法几何正确性。
- **移植一致性 fixture（关键）**：用现有 Python 流水线跑固定输入，抓取中间产物（豆框 / 轴 / 标签 / 网格 / diff）作 TS 断言金标 → 保证 TS 移植**逐算法**与 Python 对拍。`sRGB→LAB`、同伦拟合都与 `cv2` 参考比对。

### 7.2 组件 — Jest + React Native Testing Library

每屏给定 `DetectionResult` / store 状态渲染断言。重点：编号红框 + 错误索引、通过 / 有错两态、**断言不出现手动校正控件**。

### 7.3 E2E — Playwright + Expo Web（= "模拟测试"）

Expo Web 构建 + 注入 mock `IVisionPort / ICameraPort`（返回 fixture 豆框）→ 浏览器里走完整条魔法流：

- happy：`选图纸(fixture) → 拍照(mock 豆框) → 检测 → 结果(断言红框+计数) → 分享`。
- 失败：mock 低置信 → 断言重试提示；mock 反复失败 → 断言不支持提示 + 无手动工具。

这覆盖整条魔法流的 UI / 逻辑 / 状态，无需真机。

### 7.4 真机集成 — 手动 / 后期 Detox

端侧 ONNX 真实推理，真机 / 模拟器验。非 MVP 阻塞；逻辑 + 流程已被 7.1–7.3 覆盖。P4 评估 Detox。

### 7.5 fixture 单一真相源

少量真实拼豆照 + 图纸 + 人工核对的 ground-truth diff，**Python（一致性）/ TS（单元）/ Playwright（E2E）三处共用**同一份。

---

## 8. 技术栈

| 维度 | 选型 | 说明 |
|---|---|---|
| 框架 | **Expo**（managed + development build） | dev-client 承载 ONNX 原生模块 |
| 语言 | **TypeScript**（strict） | |
| 导航 | **Expo Router** | 文件式路由 |
| 端侧推理 | **onnxruntime-react-native** | YOLO 导出 ONNX，作 asset 打包 |
| 线性代数 | `ml-matrix`（或手写 SVD） | 同伦拟合；P0 定 |
| 状态 | Zustand（或 Context+reducer） | 轻量；flows 状态机 + stores；P0 定 |
| 单元/组件 | Jest + React Native Testing Library | |
| E2E | **Playwright** against Expo Web | 原生 ports mock |
| 图像 | expo-image / expo-image-manipulator | 解码 + 像素采样 |
| 存储 | AsyncStorage / expo-secure-store | 仅本地，不上云 |
| 分享 | expo-sharing + RN Share Sheet | |
| CI | GitHub Actions | 每 PR 跑 Jest + Playwright |

**ONNX 模型**：把 `yolo26n.pt`（或 `yolov8n.pt`）导出 ONNX，放进 `app/assets/models/`，作 app asset 打包（照片不出手机，端侧推理）。

---

## 9. 仓库结构（monorepo）

```
pintconut/
├── app/                      # ★ 新增：Pintcount RN app（本 spec 主体）
│   ├── core/                 # 纯 TS 检测核心（TDD）
│   │   ├── color/  blueprint/  grid/  compare/  pipeline/
│   ├── app/                  # 应用层 flows/ stores/
│   ├── platform/             # ports.ts + native/ + mock/
│   ├── ui/                   # screens/ components/ tokens/
│   ├── assets/               # palettes/（多品牌色盘） models/（ONNX） icons/
│   └── tests/                # Jest fixtures + Playwright specs
├── src/                      # 既有 Python CV（算法蓝本，不进 APP）
├── training/                 # 既有训练工具（开发资产）
├── data/                     # colors.json / board_sizes.json（参考）
└── docs/                     # PRD / 原型 / 本架构文档
```

- `app/` 与 Python `src/ training/` **并存**（monorepo）。Python 代码保留作算法参考；是否最终从 app 分支清掉，可选（开放问题 Q2）。
- **色盘**：APP 自带多品牌色盘（`app/assets/palettes/`，Perler / Artkal / Hama 等），扩展现有 `data/colors.json`（20 色 → 目标 221 色 / 多品牌）。色盘是"颜色 → 名称"查询表（PRD §4.1）。

---

## 10. 分阶段路线（垂直切片优先）

本 spec 只展开到**期级**；`writing-plans` 只把 **P1** 写成详细步骤，P2–4 各自后续出 plan。

| 期 | 内容 | 产出 / 验收 |
|---|---|---|
| **P0 基建** | `app/` Expo 脚手架 + 分层骨架 + ports/mock + Jest/Playwright 测试台 + 设计令牌（移植原型 §3）+ CI | 空壳能跑 Jest + Playwright |
| **P1 垂直切片（脊柱）★** | 检测核心 TDD（一致性 fixture）+ **结果屏**端到端（照→检测→结果，Playwright E2E，vision mock） | 一条屏打通整架构 + 测试链路 |
| **P2 魔法流 UI** | 选图纸 / 拍照 / 检测中 / 分享屏 + flows 状态机 + 失败态（§6）+ ONNX 原生适配（真机推理） | 完整魔法流（含失败处理） |
| **P3 生态** | 作品库（自动存）+ 设置（色盘）+ 分享卡片 + 权限 + 首启 | 三屏信息架构完整 |
| **P4 打磨发版** | 真机 CV 调优 + 性能 + App Store 就绪（iOS 先发） | 可上架 |

**P1 是命门**：它验证 ① TS 移植一致性（与 Python 对拍）② Playwright E2E 思路 ③ 整套分层架构 + 依赖倒置，用最小表面积。P1 通了，P2–4 是执行量。

---

## 11. 关键决策记录（ADR）

| # | 决策点 | 选择 | 理由 |
|---|---|---|---|
| 1 | 检测执行架构 | **端侧 TS 核心 + ONNX**（A1） | 符合 PRD 端侧/无后端；bead-grid 路径 ~70% 纯逻辑可 TS，唯一原生依赖是 YOLO；最契合 TDD+Playwright |
| 2 | 跨平台方案 | **Expo 单库双目标**（一份代码出 iOS/Android/Web） | Expo Web 让 Playwright 能驱动真实魔法流（原生 mock）；Expo 生态省事 |
| 3 | 平台边界 | **依赖倒置 + ports** | 核心/应用/UI 三层平台无关；TDD（Jest）+ Playwright（Web mock）干净成立 |
| 4 | 硬接缝 | `IVisionPort.detectBeads()` | 隔离 YOLO 原生推理，对应 Python `fit(detector=)` 接缝 |
| 5 | 失败处理 | **纯函数 `decideFlowState` + 无手动控件** | 魔法优先；失败逻辑可测；架构层杜绝手动校正 |
| 6 | 测试一致性 | **Python 中间产物作 TS 金标 fixture** | 保证逐算法移植正确 |
| 7 | 实现次序 | **垂直切片优先**（P1 = 核心 + 结果屏） | 最小表面积验证架构 + 测试链路 |
| 8 | Python 代码 | **并存作参考，不进 APP** | 用户明确"训练代码不用参考"；保留蓝本 |

---

## 12. 假设与开放问题

### 假设

- 「参考 Luma」= 借鉴一款精良跨平台 RN 消费级 App 的质感（组件系统 / 过渡 / 端侧 AI 的 UX），不锁定具体某款 Luma。
- RN 写一份跨平台代码，iOS 先发布、Android 随后（同 PRD）。
- 检测模型（`yolo26n.pt` / `yolov8n.pt`）已训练就绪，可导出 ONNX。
- 色盘以多品牌预设（Perler / Artkal / Hama）起步，自定义放 v2。

### 开放问题

- **Q1**（状态管理 / 线性代数库）：Zustand vs Context+reducer、`ml-matrix` vs 手写 SVD——P0 定。
- **Q2**（仓库）：Python `src/ training/` 是否最终从 app 分支清掉，还是长期并存作参考？
- **Q3**（色盘数据）：多品牌色盘的权威数据来源与格式（从现有 `data/colors.json` 扩展，还是另立 `app/assets/palettes/`）？P0 定。
- **Q4**（真机 E2E）：P4 是否引入 Detox 做真机 ONNX 推理回归？

---

> 本文档为 v0.1 草案。用户评审通过后，进入 `writing-plans` 阶段，**只展开 Phase 1（垂直切片）**为详细实现计划。
