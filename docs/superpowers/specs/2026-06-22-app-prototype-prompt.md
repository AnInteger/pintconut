# Pintcount App 原型设计图 — AI 生成 Prompt

> 日期：2026-06-22
> 关联：`docs/superpowers/specs/2026-06-18-pintcount-app-prd.md`
> 用途：给图像生成 AI 用的 prompt，用来产出 Pintcount App 的原型设计图（design showcase 风格）。
> 使用者：无设计经验的开发者（An）。直接复制下方 prompt 到对应 AI 即可。

---

## 0. 这份文档是什么

PRD 评审通过后进入 UI 设计阶段。本文档不设计 App 本身，而是**帮你用图像生成 AI 快速产出一张高保真原型设计图**，用来对齐视觉方向、对外展示、或作为后续正式设计的参考。

---

## 1. 已确定的设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| **AI 类型** | 图像生成类（非 UI 界面生成） | 要的是视觉调性 / 设计感，不是可交互原型 |
| **具体工具** | 通用版（中文核心 prompt + 各工具适配） | 不锁定单一工具，国产 / MJ / DALL·E 都能用 |
| **视觉风格** | 活力社交风（高饱和、强对比、年轻，像小红书 / 抖音） | 用户就在这两个平台，传播是增长引擎 |
| **画面构图** | 方案 C · hero 为主：中间一台 iPhone 显示「检测结果」屏 + 周围配色色卡 + 图标 | 一次只画一个主屏，AI 出片稳，又有 showcase 设计感 |
| **展示形态** | 横版 16:9，Dribbble / Behance 顶级设计稿风格 | 最像「专业设计提案」 |

### 品牌配色（已替你定，不喜欢可改）

| 角色 | 颜色 | 色号 |
|------|------|------|
| 主色 / CTA 按钮 | 活力珊瑚红 | `#FF5A5F` |
| 检测科技感 | 明亮青蓝 | `#2EC4F1` |
| 强调 / 正确 ✓ | 活力黄 | `#FFD23F` |
| 文字 | 墨黑 | `#1A1A2E` |
| 背景 | 米白 | `#FAF7F2` |

> 选色逻辑：珊瑚红呼应小红书 / 抖音（获客地），青蓝给「检测 / 科技可信感」，黄做「正确」强调；整体糖果感但不幼稚。

### 字体感

圆润粗体无衬线（中文像思源黑体 Heavy / 阿里普惠体 Heavy 的厚重感）。

---

## 2. 核心 Prompt（中文，国产模型直接用）

适用于 **即梦 / 文心一格 / 可灵** 等。直接复制粘贴：

```
一张高品质的移动 App 设计展示图，Dribbble / Behance 顶级设计稿风格。
画面中央是一台现代 iPhone（细窄边框、大圆角、柔和长投影），屏幕竖屏显示
「Pintcount 拼豆质检」App 的核心「检测结果」界面。横向 16:9 画布。

【手机屏幕内容 · 检测结果 hero 屏】
- 顶部导航栏：左侧返回箭头，中间品牌名「Pintcount」，右侧菜单图标。
- 结论标题区：超大粗体文字「发现 3 处错误」，下方一行小字「2 颜色错 · 1 空缺」。
- 主视觉区：一张俯拍的彩色拼豆成品图（像素马赛克图案、马卡龙糖果色），
  上面用醒目的红色方框圈出 3 处错误格子，红框带柔和外发光，一眼可读。
- 错误图例条：三个标签「🔴 颜色错」「⬜ 空缺」「➕ 多余」。
- 底部操作区：一个醒目的主按钮「生成分享卡片」（珊瑚红填充、大圆角、粗体白字），
  旁边一个次按钮「重扫复核」（描边样式）。

【画面其余部分 · 设计展示元素】
- 手机右侧整齐排列 5 个品牌配色色卡：纯色方块 + 下方色号文字
  （珊瑚红 #FF5A5F、青蓝 #2EC4F1、活力黄 #FFD23F、墨黑 #1A1A2E、米白 #FAF7F2）。
- 色卡旁边 2-3 个简洁的 App 功能图标：相机、盾牌对勾、分享。
- 留白干净、有呼吸感，像一张专业的设计提案展示页。

【视觉风格】
活力社交风——高饱和、强对比、年轻有张力，像小红书 / 抖音的设计语言，
但布局克制干净、不堆砌，保留质检工具的清晰与可信。
字体用圆润粗体无衬线（中文像思源黑体 Heavy 的厚重感）。
圆角卡片、柔和长投影、细节精致。

【配色】活力珊瑚红 #FF5A5F（主色/按钮）、明亮青蓝 #2EC4F1（检测科技感）、
活力黄 #FFD23F（强调/正确）、墨黑 #1A1A2E（文字）、米白 #FAF7F2（背景）。
整体明亮、年轻、糖果感但不幼稚。

【质感】UI 扁平但带细腻光影，手机屏幕清晰锐利，色卡是哑光纯色块，
背景米白带极淡暖色渐变。是「设计稿展示」的高级感，不是真实拍照。
8K、超高细节、专业 UI 设计稿、Dribbble 热门作品级别。
```

> 国产模型（即梦 / 文心 / 可灵）对中文 UI 文字识别最好，优先用它们。

---

## 3. 各工具适配

### 即梦 / 文心一格 / 可灵
- 用上面中文 prompt 原样。
- 即梦：画幅选「横图 / 16:9」，质量选高。
- 文心一格：选「高清」模式。

### Midjourney（文字必然糊，仅做调性 / 排版参考）
英文版 prompt，末尾加参数。UI 文字会乱码——接受它当占位，或后期覆盖替换。

```
A premium Dribbble-style mobile app design showcase, 16:9 canvas. Center: a modern
iPhone showing a bead-craft quality-check app "Pintcount" result screen — bold header
"Found 3 Errors", a top-down photo of colorful pixel-art fuse bead artwork with 3 glowing
red boxes marking wrong cells, a legend row, a vivid coral CTA button. Around the phone:
5 solid brand color swatches with hex labels, 2-3 minimal icons (camera, shield-check, share).
Energetic social-media aesthetic like Xiaohongshu/Douyin but clean restrained layout,
bold rounded sans-serif typography. Palette: coral #FF5A5F, cyan #2EC4F1, yellow #FFD23F,
ink #1A1A2E, off-white #FAF7F2. Flat UI, subtle soft shadows, matte chips, warm off-white
gradient bg, premium design-presentation look, ultra detailed, 8k --ar 16:9 --style raw --v 6.1
```

### DALL·E (ChatGPT)
- 用上面中文 prompt 直接发，它对布局理解较好。
- 中文字仍可能出错，多试几次。

---

## 4. 出图后小贴士（救命用）

1. **多生 6-10 次再挑**：同一 prompt 每次结果差很多，挑最对的那张，别指望一次成。
2. **文字糊是正常的**（尤其 Midjourney）：把标题「发现 3 处错误」、按钮「生成分享卡片」等关键文字，导出后用 **Canva**（免费、拖拽即可）加文本框覆盖成清晰中文，立刻专业。
3. **红框画歪 / 拼豆图太怪**：重生成；或单独再生成一张「干净的检测结果 hero 屏」，后期拼进展示图。
4. **想更稳**：在 prompt 最前面加一句参考，如「参考小红书 App 的设计风格」或「Dribbble 首页热门作品风格」。

---

## 5. 迭代与复用

- **换屏幕**：把「手机屏幕内容」整段替换成其他屏（如「首页 / 拍照入口」「作品库」「分享卡片」），其余不动，即可生成其他屏的展示图。
- **换风格**：把「视觉风格」段替换（如改成「干净工具风：大留白、克制配色、几何线条」），配色段同步调整。
- **想要更高完成度**：改用「方案 B · 分模块生成 + Canva 拼版」——分别生成各屏 / 色卡 / 图标，再拼成完整 showcase，成品最专业。

---

## 6. 决策记录

| # | 决策点 | 选择 | 理由 |
|---|--------|------|------|
| 1 | AI 类型 | 图像生成类（非 UI 界面生成） | 要视觉调性，不要可交互原型 |
| 2 | 工具 | 通用版（中文核心 + 各工具适配） | 不锁定工具，灵活 |
| 3 | 视觉风格 | 活力社交风 | 用户在抖音 / 小红书，传播驱动 |
| 4 | 构图方案 | C · hero 为主 + 轻量辅元素 | 兼顾设计感与出片稳定性 |
| 5 | 主色 | 珊瑚红 #FF5A5F | 呼应获客平台 + 活力感 |
