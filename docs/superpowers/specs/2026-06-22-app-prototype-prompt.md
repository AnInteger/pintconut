# Pintcount App 原型设计图 — AI 生成 Prompt

> 日期：2026-06-22
> 关联：`docs/superpowers/specs/2026-06-18-pintcount-app-prd.md`
> 用途：给图像生成 AI 用的 prompt，用来产出 Pintcount App 的原型设计图（design showcase 风格）。
> 使用者：无设计经验的开发者（An）。直接复制下方 prompt 到对应 AI 即可。

---

## 0. 这份文档是什么

PRD 评审通过后进入 UI 设计阶段。本文档不设计 App 本身，而是**帮你用图像生成 AI 快速产出一张高保真原型设计图**，用来对齐视觉方向、对外展示、或作为后续正式设计的参考。

> v2 修订（2026-06-22）：放宽配色（给方向不给色号，让 AI 自由发挥）；加到 2 台手机 + 饱满的结果屏，解决「颜色太死、hero 太空」的问题。

---

## 1. 已确定的设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| **AI 类型** | 图像生成类（非 UI 界面生成） | 要的是视觉调性 / 设计感，不是可交互原型 |
| **具体工具** | 通用版（中文核心 prompt + 各工具适配） | 不锁定单一工具，国产 / MJ / DALL·E 都能用 |
| **视觉风格** | 活力社交风（高饱和、强对比、年轻，像小红书 / 抖音） | 用户就在这两个平台，传播是增长引擎 |
| **画面构图** | 2 台 iPhone（大「检测结果」hero 屏 + 小「首页」屏）+ 周围图标 + 色卡 | hero 内容饱满、信息丰富，又不至于多屏混乱 |
| **展示形态** | 横版 16:9，Dribbble / Behance 顶级设计稿风格 | 最像「专业设计提案」 |

### 配色方向（给方向，不给色号 —— 让 AI 自由发挥）

- **基调**：高饱和、明亮、年轻的**糖果色系**，鲜亮但和谐。
- **主色**：暖色任选一种作品牌主色——**珊瑚红 / 品红 / 活力橙**皆可。
- **撞色**：配一道清凉的**青蓝或薄荷绿**做辅助 / 科技感。
- **背景**：明亮的**奶白 / 浅米**，不要灰暗。
- **不要**：莫兰迪低饱和、灰暗、脏色调。
- **重点**：拼豆成品图本身的**彩色像素**是画面里最丰富的色彩来源，让它当视觉焦点。

> 想锚定的话可参考：主色 `珊瑚红 #FF5A5F` 或 `活力品红 #FF4D6D`，辅助 `青蓝 #2EC4F1`——但**只是参考，别钉死**。

### 字体感

圆润粗体无衬线（中文像思源黑体 Heavy / 阿里普惠体 Heavy 的厚重感）。

---

## 2. 核心 Prompt（中文，国产模型直接用）

适用于 **即梦 / 文心一格 / 可灵** 等。直接复制粘贴：

```
一张高品质的移动 App 设计展示图，Dribbble / Behance 顶级设计稿风格，
活力、年轻、高饱和的糖果色系，明亮干净。

【整体构图】画面里有 2 台 iPhone，错落摆放、有前后层次和柔和投影：
- 左侧一台较大的手机（主角）：显示「检测结果」hero 屏。
- 右侧一台稍小的手机（配角）：显示「首页 / 一键拍照」入口屏。
手机周围点缀：品牌 App 图标、3-4 个功能小图标（相机、盾牌对勾、分享、网格）、
一组品牌主色色卡。留白干净有呼吸感，像专业的设计提案展示页。横向 16:9。

【大手机 · 检测结果 hero 屏（内容饱满，像真实界面，不要大片留白）】
- 顶部导航：返回箭头 + 品牌名「Pintcount」+ 菜单图标。
- 结论卡：超大粗体「发现 3 处错误」+ 副标题「2 颜色错 · 1 空缺」+ 醒目状态图标。
- 主视觉：俯拍的彩色拼豆成品图（像素马赛克、马卡龙糖果色），上面用醒目红框
  高亮 3 处错误格子（红框带柔和外发光），图上叠一层淡淡的网格线。
- 错误图例条：🔴 颜色错 · ⬜ 空缺 · ➕ 多余。
- 错误清单：2-3 行小列表，每行一个色块缩略图 + 错误类型 + 位置说明。
- 底部双按钮：主按钮「生成分享卡片」（主色填充、大圆角、粗体白字）+ 次按钮「重扫复核」。

【小手机 · 首页屏】
- 顶部品牌问候语。
- 中央一个超大圆形「一键拍照检测」主按钮（带相机图标，最醒目）。
- 下方「从相册选图纸」入口。
- 底部 Tab 栏：首页 / 作品库 / 设置。

【视觉风格 · 活力社交风】高饱和、强对比、年轻有张力，像小红书 / 抖音的设计语言，
但布局克制干净、不堆砌，保留质检工具的清晰可信。
圆润粗体无衬线字体，圆角卡片，柔和长投影，细节精致。

【配色 · 自由发挥的高饱和糖果色系】以暖色为基调——珊瑚红 / 品红 / 活力橙里任选
一种作品牌主色，撞一道清凉的青蓝或薄荷绿作辅助色，背景用明亮的奶白或浅米。
整体鲜亮、和谐、年轻，不要灰暗、不要莫兰迪低饱和。
拼豆成品图本身的彩色像素，是画面里最丰富的色彩来源，让它成为视觉焦点。

【质感】UI 扁平带细腻光影，手机屏清晰锐利，色卡是哑光纯色块，背景带极淡暖色渐变。
是「设计稿展示」的高级感，不是真实拍照。8K、超高细节、Dribbble 热门作品级别。
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
A premium Dribbble-style mobile app design showcase, 16:9, energetic young high-saturation
candy palette, bright and clean. Composition: two iPhones arranged with depth and soft
shadows — a larger phone (left, hero) showing a "Result" screen, a smaller phone (right)
showing a "Home / scan" screen. Around them: an app icon, 3-4 minimal function icons
(camera, shield-check, share, grid), a row of brand color swatches.

Large phone — Result screen (rich, realistic UI, no empty space): nav bar with back arrow +
brand "Pintcount" + menu; a bold headline card "Found 3 Errors" + subtitle "2 wrong color ·
1 missing" + status icon; main visual = a top-down photo of colorful pixel-art fuse bead
artwork with 3 glowing red boxes marking wrong cells and a faint grid overlay; a legend row
(wrong color / missing / extra); a 2-3 row error list; bottom dual buttons — primary
"Share Card" + secondary "Rescan".

Small phone — Home screen: greeting, a huge circular "One-Tap Scan" camera button center,
an "Import Blueprint" entry below, a bottom tab bar (Home / Gallery / Settings).

Energetic social-media aesthetic like Xiaohongshu/Douyin but clean restrained layout,
bold rounded sans-serif, rounded cards, soft long shadows. Color: warm-led vibrant candy
palette — pick coral / magenta / orange as the brand primary, clash with a cool cyan or mint
accent, bright off-white background; the bead artwork's own vivid pixels are the richest color
source and visual focus. Flat UI with subtle light, crisp screens, matte chips, ultra detailed,
8k --ar 16:9 --style raw --v 6.1
```

### DALL·E (ChatGPT)
- 用上面中文 prompt 直接发，它对布局理解较好。
- 中文字仍可能出错，多试几次。

---

## 4. 出图后小贴士（救命用）

1. **多生 6-10 次再挑**：同一 prompt 每次结果差很多，挑最对的那张，别指望一次成。
2. **文字糊是正常的**（尤其 Midjourney）：把标题「发现 3 处错误」、按钮「生成分享卡片」等关键文字，导出后用 **Canva**（免费、拖拽即可）加文本框覆盖成清晰中文，立刻专业。
3. **2 台手机画乱了**：比单台更易翻车。若两屏互相干扰、风格不统一，就**删掉小手机**，只保留大 hero 屏（回到单屏，出片最稳）。
4. **红框画歪 / 拼豆图太怪**：重生成；或单独再生成一张「干净的检测结果 hero 屏」，后期拼进展示图。
5. **颜色想更自由**：把【配色】段进一步简化成「高饱和糖果色系，色彩由你发挥」，完全交给 AI；想更收就把主色锚定成一个具体色相。
6. **想更稳**：在 prompt 最前面加一句参考，如「参考小红书 App 的设计风格」或「Dribbble 首页热门作品风格」。

---

## 5. 迭代与复用

- **换屏幕**：把「大手机 / 小手机」整段替换成其他屏（如「检测中」「作品库」「分享卡片」），其余不动，即可生成其他屏组合。
- **换风格**：把「视觉风格」段替换（如改成「干净工具风：大留白、克制配色、几何线条」），配色段同步调整。
- **想要更高完成度**：改用「方案 B · 分模块生成 + Canva 拼版」——分别生成各屏 / 色卡 / 图标，再拼成完整 showcase，成品最专业。

---

## 6. 决策记录

| # | 决策点 | 选择 | 理由 |
|---|--------|------|------|
| 1 | AI 类型 | 图像生成类（非 UI 界面生成） | 要视觉调性，不要可交互原型 |
| 2 | 工具 | 通用版（中文核心 + 各工具适配） | 不锁定工具，灵活 |
| 3 | 视觉风格 | 活力社交风 | 用户在抖音 / 小红书，传播驱动 |
| 4 | 构图 | 2 台 iPhone（大 hero 结果屏 + 小首页屏） | hero 饱满、信息丰富，又不混乱 |
| 5 | 配色 | 高饱和糖果色系（暖主调 + 冷撞色），给方向不给色号 | 给 AI 色彩自由，避免死板 |
