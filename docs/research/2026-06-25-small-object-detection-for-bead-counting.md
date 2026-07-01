# 小物体识别算法地图 —— 面向"拼豆检测统计"的实战决策指南

> 产出：deep-research 工作流（106 agents，24 源，25 条论断经对抗式核实，21 确认 / 4 否决）。
> 定位：实战决策导向，少公式，多直觉与"什么时候用哪个"。日期：2026-06-25。

---

## TL;DR —— 三句话 + 一个关键 reframe

1. **通用小物体识别（SOD）是 CV 公认的开放难题**：在 COCO 上，中/大物体 AP 已逼近 70%/80%（基本解决），但小物体（<32×32 像素）AP_S 仍只有 ~30–36%，落后约 30 个百分点。即便 transformer 检测器时代也未弥合。
2. **但你的拼豆问题不是"通用 SOD"，而是"规则结构 + 可独立检测"的小物体计数**——这是一个被严重低估的优势。对这类问题，**最强大的杠杆不是更牛的检测器，而是网格/晶格几何先验**。你系统里的 `bead_grid.py` 网格拟合正是这一类，方向对。
3. **人群计数的 SOTA（密度图法）不适用于你**：它的优势在"密集 + 互相遮挡 + 无规律结构"，而拼豆是稀疏可分 + 强规则网格。对拼豆，**检测 + 几何网格拟合**比密度图更自然。别被"计数 SOTA"带偏。

> **Reframe**：把你的系统从"我在做小物体检测"重新理解为"我在做『规则网格上的圆形物体计数与编号』"——后者有专属、更便宜的解法（几何拟合），不必去卷通用 SOD 的军备竞赛。

---

## 1. 第一性原理：为什么小物体识别难？

直觉版：**相机拍到的小物体，本质上携带的信息就少**。一颗 20×20 像素的豆子，只有 400 个像素来表达"这是一颗豆 + 它的颜色 + 它的边界"；一张 200×200 的大物体有 40000 个像素。信息量差 100 倍，再加上模糊、噪声，小物体天然是"低信噪比"问题。

对**深度学习检测器**，这变成两个具体的失败模式（高置信度，多份综述一致）：

- **下采样把小物体"卷没了"**：CNN 靠反复卷积 + 缩放提取特征，每缩一次分辨率减半。一个 32×32 的物体经过几层 stride-2 下采样后，在特征图上可能只剩 2×2 个点，细节（纹理、边缘）几乎消失。
- **anchor 分配饿死小物体**：标准 anchor-based 检测器（Faster R-CNN / YOLO 老）按"框与 anchor 的 IoU"分配正样本。小物体的框小，稍有偏移 IoU 就暴跌，导致训练时几乎没有正样本喂给它。

> 对你的系统：YOLOv8n 是 anchor-free 的（缓解了第二个问题），但下采样问题依然存在。如果照片里豆子很小（<32px），YOLOv8n 直接跑全图大概率会漏——这就是 SAHI/P2 头存在的意义（见 §5）。

而对**经典 CV（HoughCircles）**，难点不同：它不依赖学习，但对**光照、反光、预处理参数极其敏感**——有研究专门针对圆形物体检测得出结论：常规算法（Hough）受环境光和反光影响显著，这是催生深度学习替代方案的主要原因。这正是你 Hough 路线的软肋。

---

## 2. 标准管线：典型流程分 5 步

业界推荐的 SOD 实用管线（MDPI 2025 PRISMA 综述，高置信度）：

| 阶段 | 在干什么 | 典型坑 |
|---|---|---|
| ① 预处理 | 去噪、白平衡、受控 resize、光照归一化 | 过度缩放会把小物体直接抹掉；激进去噪丢失边缘 |
| ② 候选区域/定位 | 找到"哪里可能有物体"：selective search / anchor / anchor-free 关键点 / FPN 多尺度特征 | 小物体在低层高分辨率特征图上才看得清，深层看不到 |
| ③ 检测/分类 | 对候选区域分类 + 回归框（或直接输出关键点） | 类别不平衡、小物体正样本少 |
| ④ 后处理 | NMS 去重、阈值过滤、跨尺度融合 | NMS 的 IoU 阈值对小框很敏感；低置信度阈值=高召回但多误检 |
| ⑤ 计数/结构化 | 数框、密度图积分、或网格编号 | 通用 NMS 不知道你的网格规则——这是你能加先验的地方 |

**关键洞察**：步骤 ⑤ 在通用 SOD 里是"数框"这么简单；但在拼豆里，你有**网格结构**这个强先验，可以把"几何网格拟合"提升为检测的后处理甚至先验，同时修正检测漏检和编号错位（见 §5、§8）。

---

## 3. 算法地图：经典 CV vs 深度学习（你的系统已标出）

### 经典 CV 一族

| 方法 | 直觉 | 擅长 | 弱点 | 数据/算力 |
|---|---|---|---|---|
| **HoughCircles**（你的 `detect_beads`）| 对每个像素"投票"它属于哪个圆的边缘，累计票数找圆 | 圆形、边界清晰、尺寸已知 | 对光照/反光/参数极敏感；密集时易漏/重复 | 零标注，CPU 即可 |
| **EDCircles** | 参数无关、实时（640×480 仅 10–20ms）的圆检测，Hough 的现代替代 | 同上，但无需手调参 | 仍是基于边缘，光照问题仍在 | 零标注，CPU |
| **轮廓 + 连通域** | 二值化后找闭合轮廓 | 对比分明的物体 | 颜色相近时粘连 | 零标注 |
| **模板匹配** | 拿一个"标准豆"模板在图上滑窗比对 | 形状高度一致、尺度固定 | 尺度/旋转变化即失效 | 零标注 |
| **分水岭** | 把图像当地形，"注水"分割粘连区域 | 粘连物体分离 | 对噪声敏感、易过分割 | 零标注 |
| **传统 ML（SVM/HOG）** | 手工特征 + 分类器 | 中等复杂度、可解释 | 特征工程重、泛化差 | 少量标注 |

### 深度学习一族

| 家族 | 代表 | 直觉 | 擅长 | 弱点 | 代价 |
|---|---|---|---|---|---|
| **two-stage** | Faster R-CNN | 先找候选框→再精分类/回归 | 精度高、定位准 | 慢、小物体仍弱 | 中量标注 + GPU |
| **one-stage（anchor）** | YOLOv3/SSD/RetinaNet | 一次前推断出所有框 | 快 | 小物体天然弱（anchor 问题） | 中量 + GPU |
| **one-stage（anchor-free）** | **YOLOv8/v9/X**（你的 `bead_detect`）、CenterNet、FCOS | 直接预测中心点 + 尺寸，不依赖 anchor | 小物体比 anchor-based 友好、快 | 仍受下采样限制 | 中量 + GPU |
| **transformer** | DETR / RT-DETR / DINO | 用注意力全局匹配，端到端无 NMS | 大物体强、训练稳 | 小物体收敛慢、训练贵 | 大量 + 大 GPU |

> **你的系统地图定位**：你同时跑了**经典 HoughCircles 路线**（`bead_grid.detect_beads`）和 **anchor-free 深度路线**（`bead_detect`，YOLOv8n），再用 **几何网格拟合**（`bead_grid` 的 lattice/affine/homography）做后处理编号。这是一个"经典 + DL + 几何后处理"的**混合架构**——研究证据明确支持这种混合方向（见 §7）。

---

## 4. 计数专题：三大家族 + 为什么密度图不适合你

密集小物体计数有三大方法族（NeurIPS 2020 DMCount、Springer 2024 综述，高置信度）：

1. **先检测后计数（detect-then-count）**：检测每个个体→数框。**需要框标注**；密集遮挡下失效。
2. **直接计数回归**：图→直接输出一个数字。黑盒、可解释性差。
3. **密度图估计（density map）**：图→"每个像素有多少个体"的热力图→积分得到总数。目前人群计数 SOTA 基于此。优势：**抗遮挡、不需要早期二值化决策**。

**但有一个被研究明确标注的适用范围限定（关键）**：密度图法的鲁棒性优势针对的是**密集、无规律、互相遮挡**的场景（人群）。在**稀疏、可独立检测、有规则结构**的场景下，detect-then-count 反而更准、更可解释。**拼豆属于后者**——每颗豆可独立检测，网格结构强。

> **结论**：别把人群计数的密度图 SOTA 直接搬过来。你的"检测 + 网格编号"路线对拼豆是更合适的方法族。

---

## 5. 拼豆专属技术：这几个真正值得你知道

### 5.1 SAHI（切片推理）——如果你坚持 YOLO 路线，这是最便宜的升级
SAHI = Slicing Aided Hyper Inference。**检测器无关的插件**：把大图切成重叠小块→每块单独检测→NMS 合并。本质是给小物体"放大镜"，不改模型即可叠加到任何检测器上。

- 纯推理（不重训）：**+5.1%–6.8% AP**（FCOS/VFNet/TOOD，VisDrone/xView 数据集，原文逐字核实）。
- 叠加"切片微调"累计：**+12.7%–14.5% AP**。
- **三个实操铁律**：① 切片**必须重叠**（否则边界物体被切断，性能反降——这是被消融实验证实的"边界效应"）；② 必须做**跨片 NMS** 去重；③ 若模型**没在切片上训练过，SAHI 可能反而变差**。

### 5.2 网格/晶格几何拟合 —— 你的"秘密武器"，研究证实方向对
对规则网格上的物体，**几何拟合是最自然、最省标注**的解法：检测出大部分豆子中心点→拟合规则网格（你的 affine/homography lattice）→**用网格补回漏检的豆子、修正编号**。这等于把"网格规则"当作强先验注入，既提升召回又保证编号一致性。这正是 `bead_grid.py` 在做的事。

> 一个可探索方向（研究的 open question）：能否把网格几何从"后处理"提升为"检测的强先验"——即让检测器输出时就受网格约束？目前没有专门针对"规则网格小圆形物体"的 benchmark 论文做这个，对你可能是差异化创新点。

### 5.3 EDCircles —— HoughCircles 的现代替代
如果你保留经典路线，可考虑把 HoughCircles 换成 **EDCircles**（TPAMI 2013，但仍是现役基线）：参数无关、640×480 仅 10–20ms。能省掉你 `detect_beads` 里手调 `param2`/半径范围的痛苦。不过它仍基于边缘，光照问题没有根治。

### 5.4 P2 检测头 / 高分辨率特征 —— DL 路线治"下采样丢信息"
SOD 四大解法主线之首就是**提升特征图分辨率**（加 P2 stride-4 头、高分辨率 FPN、超分辨率分支），在 WIDER FACE / TinyPerson 上实验证明显著涨点。YOLOv8 有 YOLOv8-P2 变体专门为小物体设计。如果你的豆子落 <32px，这比 SAHI 更治本（SAHI 是推理期补救，P2 是架构期根治）。

---

## 6. 数据 / 训练 / 评估实战

**标注**：小物体的框标注必须**像素级贴边**——IoU 对小框极其敏感，1–2 像素偏移就能让 IoU 暴跌、直接拉低 mAP。你的 `bead_annotate_ui.py` 在做这件事，方向对。半自动标注（Hough 预标注 + 人工修正）能大幅降本，你已经在这么干。

**数据增强（重要陷阱）**：
- YOLO 默认开 **Mosaic**（mosaic=1.0），官方称对"小物体高度有效"——**但有显著反面证据**：Mosaic 会把部分小物体缩小、甚至切断框导致假阳性。Ultralytics 默认 `close_mosaic=10`（最后 10 epoch 关掉）正是为此。**训练拼豆时务必监控 Mosaic 对豆子的影响，别盲信默认值。**
- **Copy-Paste 增强对纯检测格式（class cx cy w h）是空操作**——它只在分割格式下生效。你的 YOLO 检测管线用 Copy-Paste 不会有效果。（源码 + 官方文档 + Issue #18073 核实）

**评估**：别只看 mAP。要分尺寸看 **AP_S / AR_S**（小物体精度/召回），并单独看**计数误差**（|预测数 − 真实数|，或 MAE）。你的 `tests/validation/` 已经在做 ground-truth 标注（commit 43f67ca），可以同时产出这两类指标。

---

## 7. 决策框架：什么时候用什么（含拼豆推荐路线）

### 通用决策树
- 物体**规则、对比度高、有强几何结构** → **经典 CV + 几何拟合**往往已够，且省标注省算力。
- 物体**多变、光照/颜色/遮挡复杂** → **上深度学习**，买的是泛化能力。
- 两者都不极端 → **混合**：经典 CV 生成候选（便宜高召回）+ DL 精修（抗干扰）+ 几何后处理（规则化）。

### 对拼豆的推荐路线（confidence：medium——无拼豆专用 benchmark，基于通识 + 你架构推导）
> 你现有的"经典 HoughCircles / YOLOv8n 双路线 + 网格拟合"混合架构，**方向是证据支持的**。具体建议优先级：

1. **把"网格几何拟合"当核心，而不是配角**。它是你相对通用 SOD 系统的最大差异化优势。投入让它在漏检时能"补豆"、在编号错位时能"纠偏"。
2. **给 HoughCircles 路线补一个"光照鲁棒"的兜底**——要么 EDCircles，要么在光照差时自动切换到 YOLO 路线。研究的核心反方观点就是"经典方法死于光照/反光"。
3. **如果豆子经常 <32px 且你依赖 YOLO**：先试 **SAHI**（最便宜，不重训），不行再上 **YOLOv8-P2**（治本但要重训）。务必重叠切片 + 跨片 NMS。
4. **训练监控**：盯住 Mosaic 是否在帮倒忙；用 AP_S + 计数误差双指标评估，别只看 mAP。
5. **别走密度图计数路线**——对拼豆是错配的方法族。

---

## 8. 对你现有系统的逐模块诊断

| 你的模块 | 在地图上的位置 | 状态 | 最值得动的点 |
|---|---|---|---|
| `bead_grid.detect_beads`（HoughCircles 暗环） | 经典 CV 圆检测 | 方向对，但光照敏感 | 兜底方案：光照差时切 EDCircles 或 YOLO |
| `bead_detect`（YOLOv8n） | anchor-free DL 单阶段 | 对小物体比老 YOLO 友好 | 若漏小豆→SAHI/P2；监控 Mosaic |
| `bead_grid` 仿射/单应 lattice 拟合 | **几何先验（你的核心差异化）** | **方向最对** | 把它做强：漏检补全 + 编号纠偏 |
| `color.py`（LAB 221 色匹配） | 颜色分类（独立于检测） | 成熟 | 注意光照影响 LAB——可考虑与检测联合 |
| `bead_annotate_ui.py` 标注 | 小物体像素级标注 | 方向对 | 半自动标注降本，同时产出计数 ground-truth |

**一句话**：你的检测器不是短板，**网格几何拟合的深度**和**Hough 的光照兜底**才是 ROI 最高的两个改进点。

---

## 9. 可信度与盲区（诚实地标注）

- **SAHI 的 +6% 等数字来自航拍数据集（VisDrone/xView）上的 FCOS/VFNet/TOOD**，既不是 YOLOv8n、也不是拼豆。迁移到拼豆的实际增益未知，且模型没在切片上训过时 SAHI 可能反降。
- **"密度图优于检测计数"严格限于密集人群**，对稀疏可分的拼豆不适用——已在 §4 标注。
- **"Mosaic 对小物体有效"是官方说法，但有反面证据**——实战需监控，别盲信。
- **没有任何针对"规则网格小圆形物体 + 晶格编号"的专用 benchmark 或论文**。§7/§8 的拼豆推荐（confidence medium）是基于通识 + 你架构的合理推导，未经专门实验验证。
- 时效性：核心结论（下采样丢信息、anchor 分配、SAHI、密度图适用范围）是 2020–2025 跨多份综述的稳定共识，不易过时；具体 AP 数字会随新模型演进。

---

## 10. 来源（权威，已核实）

**综述（小物体检测）**
- Nikouei et al. 2025, *Small Object Detection: A Comprehensive Survey on Deep Learning Methods* — arXiv:2503.20516
- MDPI Applied Sciences 2025, *Unified Practical Pipeline for SOD* (PRISMA) — mdpi.com/2076-3417/15/22/11882
- AIMS MBE 2023, *SOD 四视角分类* — aimspress.com/article/doi/10.3934/mbe.2023282
- PAR 2023, *Deep Learning for Small and Tiny Object Detection: A Survey* — par.pl
- Chen 2022, *Four Pillars for SOD* — ScienceDirect

**计数 / 密集小物体**
- Wang et al. NeurIPS 2020, *DMCount*（密度图 vs 检测计数）— proceedings.neurips.cc
- arXiv:2508.16970（TOD 数据集统计：AI-TOD/TinyPerson/JHU-Crowd++）
- Springer 2024 人群计数综述 — link.springer.com/article/10.1007/s44336-024-00011-8

**SAHI / 切片推理**
- Akbas & Akhan, ICIP 2022, *SAHI*（+6.8% 等原始数字）— arXiv:2202.06934
- Ultralytics 官方 SAHI 文档 — docs.ultralytics.com/guides/sahi-tiled-inference
- obss/sahi GitHub

**经典 vs 深度学习**
- O'Mahony et al. 2019, *Deep Learning vs Traditional Computer Vision* — arXiv:1910.13796
- Roboflow 博客 — blog.roboflow.com/deep-learning-vs-traditional-computer-vision/

**圆形物体 / 拼豆领域（最相关）**
- BeadNet-v2（现役拼豆检测网络）— github.com/TimScherr/BeadNet-v2
- EDCircles（参数无关实时圆检测，TPAMI）— publications.ri.cmu.edu / ScienceDirect S0031320312004268
- StackOverflow *Optimizing image recognition of colored beads in OpenCV*（最接近的公开讨论）— stackoverflow.com/q/54749320
- 圆形物体检测：常规算法受光照/反光影响的实证研究 — ScienceDirect S0031320325010684

**Ultralytics 工程实战**
- *How to improve model mAP on small objects* — ultralytics.com/blog
- *YOLO Data Augmentation*（Mosaic / copy-paste 行为）— docs.ultralytics.com/guides/yolo-data-augmentation
- *Object Counting* — docs.ultralytics.com/guides/object-counting
