# Pintcount 设计系统（Style Guide）

> 单一事实源。所有原型屏与未来 RN 实现 都以此为准。
> 版本：v3（2026-07-02）· 取代 `2026-06-22-pintcount-prototype-design.md` 的色/字/组件部分。
> 参考实现：`home-luma.html`

---

## 1. 设计立场

**「高级的多巴胺」**——克制的版面 + 来自产品本身（拼豆）的鲜艳色。

- 产品：拼豆熨烫前质检。端侧、无后端。受众：小红书 / 抖音年轻手作玩家。
- 核心隐喻：**界面由"豆"构成**。一颗豆（带高光 + 中心孔）是整个设计的原子单位。
- 取舍：**内容多彩，chrome 收敛**。多巴胺色用在内容（豆豆、色板、作品缩略图、错误标记）和 CTA；导航/底栏/容器保持中性。
- 只在一个地方大胆：**签名元素**（豆原子 + 拼板纹理 + 轮换 CTA），其余留白与排版保持克制。

## 2. 颜色

### 豆色板（多巴胺主色，4 个）
| 名 | 值 | 用途 |
|---|---|---|
| violet | `#6C4DF0` | 主锚色 / CTA 默认 / 品牌脊柱 |
| rose | `#F04E6E` | 颜色错 / 暖强调 |
| amber | `#E88F1E` | 多余 / 暖强调 |
| teal | `#0EB5A6` | 通过 / 冷强调 |
| sapphire（第 5，仅语义） | `#3A7BD5` | 空缺 |

### 签名渐变
`linear-gradient(135deg, #F04E6E 0%, #6C4DF0 100%)`（rose→violet）——**克制使用**：右上光晕、偶尔强调。CTA 不用渐变（用实色轮换）。

### 中性
| token | 值 | 用途 |
|---|---|---|
| `--bg` | `#FFFBF7` | 应用底（暖白） |
| `--surface` | `#FFFFFF` | 卡片表面 |
| `--ink` | `#1A1320` | 主文字（暖近黑） |
| `--muted` | `#7A7286` | 次要文字 |
| `--line` | `#EFE9F5` | 极浅分割 |

**禁止**：通用红 `#EF4444`、满屏彩虹、荧光饱和、彩虹渐变洪水。

## 3. 原子单位：一颗"豆"

每个点缀圆点都是一颗**带孔实体豆**——高光（左上 inset 白）+ 暗面（右下 inset 黑）+ 中心孔（小黑圆）。CSS `.bead`：

```css
.bead { position:relative; border-radius:50%;
  box-shadow: inset -2px -2px 3px rgba(0,0,0,.16), inset 2px 2px 3px rgba(255,255,255,.55); }
.bead::after { content:''; position:absolute; left:50%; top:50%; width:32%; height:32%;
  border-radius:50%; background: rgba(0,0,0,.18); transform: translate(-50%,-50%); }
/* 尺寸：.bead.xs 9px / .sm 11px / .md 18px */
```

用在：logo（2×2 豆阵）、色板条、项目符号、状态点、错误标记、tab 指示。

## 4. 签名元素

1. **拼板纹理（pegboard）**：极淡的孔位点阵背景，`radial-gradient(circle, rgba(26,19,32,.05) 1px, transparent 1.7px); background-size:13px 13px;`。用于 hero / 面板的氛围底，**不铺满**。
2. **右上光晕（glow）**：rose→violet 径向软光斑，置于 hero 右上角（用户偏好保留）。
3. **轮换 CTA**：实心单色，在 4 个豆色间缓慢轮换（12s，`prefers-reduced-motion` 自动停）——"每次随机一个色"。

## 5. 字体

| 角色 | 字体 | 用途 |
|---|---|---|
| 拉丁 / 数字 / 品牌名 | **Space Grotesk** 500/600/700 | wordmark、数字（错误数）、眉标（RECENT/YOUR BEADS）、日期 |
| 中文 + 正文 | **Inter** + PingFang SC（系统兜底） | 标题、正文、按钮文案 |

**字号阶**：display 42–44 / 800 / 字距 −1.6 · h1 22 · title 17 / 800 · body 15.5 / 500 · label 13 / 700 · eyebrow 10.5 / Space Grotesk / 大写字距 1 · caption 11.5–12 / Space Grotesk。

**禁止**：小于 11px；正文用渐变字；默认 Inter 扛全场（拉丁必须 Space Grotesk 出个性）。

## 6. 布局

- **设备**：iPhone 17 Pro，**402 × 874pt**，圆角 55px，固定高 flex-column；底栏（tabbar / screen-footer / home-bar）`margin-top:auto` 吸底。
- **间距**：4pt 网格；页边距 **22px**；段落/区块 24–44px；组件内 padding 12–16。
- **详情屏骨架**：`nav(chev+title+action) → hero(240px, r-card, 左下徽章) → meta 行(row-between) → 小节标签 + 软底面板 → screen-footer`。
- **首页骨架**：`hero(拼板底+光晕+豆logo+标题+副+色板条+轮换CTA) → 最近作品(装裱卡) → 悬浮玻璃 tab-island`。

## 7. 组件

| 组件 | 规格 |
|---|---|
| **CTA** | 高 58，r 20，实色轮换 4 豆色，白字，柔影；右侧箭头是个小白半透明"子珠"圆 |
| **tabbar** | **悬浮玻璃岛**：边距 52、毛玻璃、r 24；选中=**纯墨色**（图标+字），**无填充无描边**；图标=豆点语言（首页 1 豆 / 作品库 4 豆方阵 / 设置 豆环），currentColor |
| **主按钮** | 实色（豆色或墨色）；次级=文字/ghost；r/高 与 CTA 一致 |
| **装裱卡（作品）** | 白卡 + 软影 + r 20；缩略图 + 页脚（名 / 状态豆+Space Grotesk 数字 / 日期）；可首张 featured 加大 |
| **图上徽章** | 深色半透明胶囊 `rgba(26,19,32,.78)` + 模糊；状态用豆+数字 |
| **软底面板** | `--bg-soft`，r 20，无边框，成组信息（错误索引 / 历次 / 备豆统计） |
| **错误标记（结果页）** | 三类各一豆色 + 编号 + 字形（实心点 / 空心 / ＋），色弱可读：颜色错=rose、空缺=sapphire、多余=amber |

## 8. 动效

- CTA 12s 色轮换；扫描光带（capture / processing）；加载环（processing）。
- **克制**：不为装饰加动效；**全程** `prefers-reduced-motion` 兜底。

## 9. 文案

去"AI 味"、工具感、平实动词、句首大写。按钮说出会发生什么：**「开始质检」**（非 Submit）。同一动作全程同名。中文为主，拉丁眉标/数字用 Space Grotesk。空/错态给方向不卖惨。

## 10. 无障碍

最小可点 44pt；ink on 暖白达标，muted 仅次要；错误不只靠颜色（编号 + 字形）；动效遵 reduced-motion。

## 11. 技术

- 目标栈：**React Native + Expo**（iPhone 17 Pro 402×874）。
- 原型：静态 HTML（`docs/mockups/`），共享 `styles.css`（令牌 + 原子 + 组件）+ 各屏轻覆盖。
- 字体：Space Grotesk + Inter（Google Fonts）；中文系统 PingFang。

## 12. Do / Don't

**Do**：豆=原子；拼板纹理只用氛围；多巴胺给内容 + 轮换 CTA；拉丁用 Space Grotesk；留白慷慨；一色一职。

**Don't**：满屏彩虹；通用红错误；渐变洪水 chrome；emoji；AI 口号文案；默认模板腔。

## 13. 文件

| 路径 | 作用 |
|---|---|
| `docs/mockups/STYLE.md` | 本文件，设计系统 |
| `docs/mockups/styles.css` | 共享令牌 + 原子 + 组件（代码侧约束） |
| `docs/mockups/home.html` | v3 参考实现（首页，已迁移） |
| `docs/mockups/*.html` | 各屏（按本系统迁移） |
| `docs/superpowers/specs/2026-06-18-pintcount-app-prd.md` | 产品需求（v1.4） |
