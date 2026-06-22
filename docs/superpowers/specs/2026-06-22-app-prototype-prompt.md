# Pintcount App 原型设计图 — AI 生成 Prompt

> 日期：2026-06-22
> 关联：`docs/superpowers/specs/2026-06-18-pintcount-app-prd.md`
> 用途：给图像生成 AI 的 prompt，产出 Pintcount App 原型设计图（design showcase 风格）。
> 使用者：无设计经验的开发者（An）。复制下方 prompt 到对应 AI 即可。

> v3 修订（2026-06-22）：去掉所有配色约束（只约束风格），并精简 prompt。

---

## 1. 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| AI 类型 | 图像生成类 | 要视觉调性，不要可交互原型 |
| 工具 | 通用版（中文核心 + 各工具适配） | 不锁定工具 |
| 视觉风格 | 活力社交风（年轻、强对比、像小红书 / 抖音） | 用户在这两个平台，传播驱动 |
| 配色 | **不约束，由 AI 自由发挥** | 给最大色彩自由 |
| 构图 | 2 台 iPhone（大「检测结果」hero + 小「首页」）+ 图标 | hero 饱满、信息丰富 |

---

## 2. 核心 Prompt（中文，国产模型直接用）

适用于 **即梦 / 文心一格 / 可灵**。直接复制：

```
一张高品质的移动 App 设计展示图，Dribbble 顶级设计稿风格，活力社交风
（年轻、强对比、像小红书 / 抖音的设计语言），明亮干净。横向 16:9。

【构图】画面里 2 台 iPhone 错落摆放、带柔和投影：左侧大的显示「检测结果」hero 屏，
右侧小的显示「首页」屏。周围点缀 App 图标和几个功能小图标（相机、盾牌对勾、分享）。

【大手机 · 检测结果屏（内容饱满，像真实界面）】
- 顶部导航：返回 + 品牌「Pintcount」+ 菜单。
- 结论卡：超大粗体「发现 3 处错误」+ 副标题「2 颜色错 · 1 空缺」。
- 主视觉：俯拍彩色拼豆成品图，红框高亮 3 处错误格子（带发光），叠淡网格线。
- 图例：🔴 颜色错 · ⬜ 空缺 · ➕ 多余。
- 错误清单：2-3 行，每行色块缩略 + 类型 + 位置。
- 底部双按钮：「生成分享卡片」(主) + 「重扫复核」(次)。

【小手机 · 首页屏】问候语 + 中央超大圆形「一键拍照检测」按钮 +「从相册选图纸」
入口 + 底部 Tab（首页 / 作品库 / 设置）。

【质感】扁平 UI 带细腻光影，圆角卡片，柔和长投影，圆润粗体无衬线字体，
手机屏清晰锐利。设计稿展示的高级感，非真实拍照。8K、超高细节。
```

> 配色完全交给 AI；若想收一点，可在开头加一句主色方向（如「以暖色为主调」）。
> 国产模型（即梦 / 文心 / 可灵）对中文 UI 文字识别最好，优先用。

---

## 3. 各工具适配

- **即梦 / 文心一格 / 可灵**：上面中文 prompt 原样用。即梦选「横图 / 16:9」+ 高质量。
- **Midjourney**（文字必然糊，仅做调性参考）。英文版，末尾加参数：

```
A premium Dribbble-style mobile app design showcase, 16:9. Energetic social-media aesthetic
—young, high-contrast, like Xiaohongshu/Douyin design language, bright and clean.
Composition: two iPhones with soft shadows, arranged with depth — a larger phone (left) showing
a "Result" screen, a smaller phone (right) showing a "Home" screen. Around them: an app icon and
a few minimal function icons (camera, shield-check, share).
Large phone — Result screen (rich, realistic UI): nav bar (back + brand "Pintcount" + menu); bold
headline card "Found 3 Errors" + subtitle "2 wrong color · 1 missing"; main visual = top-down photo
of colorful pixel-art fuse bead artwork with 3 glowing red boxes marking wrong cells and a faint
grid overlay; a legend row; a 2-3 row error list; bottom dual buttons — primary "Share Card" +
secondary "Rescan".
Small phone — Home screen: greeting, a huge circular "One-Tap Scan" button center, an "Import
Blueprint" entry, a bottom tab bar (Home / Gallery / Settings).
Flat UI with subtle light, rounded cards, soft long shadows, bold rounded sans-serif, crisp screens.
Premium design-presentation look, not a photo. Ultra detailed, 8k --ar 16:9 --style raw --v 6.1
```

- **DALL·E (ChatGPT)**：用中文 prompt 直接发，中文字多试几次。

---

## 4. 出图小贴士

1. **多生 6-10 次再挑**：同 prompt 每次差很多，挑最对的。
2. **文字糊正常**（尤其中文 / MJ）：关键文字用 **Canva**（免费拖拽）加文本框覆盖成清晰中文。
3. **2 台手机画乱了**：删掉小手机，只留大 hero 屏，出片最稳。
4. **想更稳**：prompt 开头加「参考小红书 App 设计风格」或「Dribbble 热门作品风格」。

---

## 5. 迭代复用

- **换屏幕**：替换「大手机 / 小手机」段落为其他屏（检测中 / 作品库 / 分享卡片）。
- **换风格**：替换风格句（如「干净工具风：大留白、克制、几何线条」）。
- **要更高完成度**：分模块生成各屏 / 图标，再用 Canva 拼版。
