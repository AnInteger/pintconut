# 🫘 Pintconut — 拼豆差异检测器

比较拼豆作品照片与拼豆图纸，自动检测并标注出拼错的位置。

## 工作原理

```
拼豆照片 + 图纸 → 定位拼板 → 透视校正 → 网格提取 → 颜色对比 → 标注差异
```

1. **定位拼板** — 用 YOLOv8-seg（迁移学习）从照片中定位拼板区域
2. **透视校正** — 将倾斜的拼板校正为标准矩形
3. **网格提取** — 按拼板尺寸等分为网格，每格采样颜色
4. **颜色对比** — 在 LAB 色彩空间逐格对比照片与图纸（221 色盘）
5. **标注差异** — 在照片上用红框标记拼错的位置

## 快速开始

### 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 方式一：Gradio Web UI（推荐）

```bash
python src/label_ui.py
```

浏览器打开 `http://localhost:7860`，按四步向导操作：

| 步骤 | 操作 |
|------|------|
| ① 上传照片 | 拖拽或选择多张拼板照片 |
| ② 自动分割 | FastSAM 自动识别候选区域 |
| ③ 逐张确认 | 点选正确的拼板区域 |
| ④ 导出数据集 | 生成 YOLO 格式数据集 |

### 方式二：命令行

**半自动标注：**

```bash
python training/semi_auto_label.py --input training/photos --output training/dataset
```

**训练模型：**

```bash
python training/train.py --data training/dataset/data.yaml
```

**验证模型：**

```bash
python training/validate.py --model training/runs/beadboard-v1/weights/best.pt --image training/test_images/
```

**检测差异：**

```bash
python -m src.cli \
  --photo photo.jpg \
  --blueprint blueprint.png \
  --board-size 29x29 \
  --model models/beadboard-best.pt \
  --output result.jpg
```

## 项目结构

```
pintconut/
├── src/
│   ├── detect.py          # 阶段①：拼板定位（YOLOv8-seg）
│   ├── grid.py            # 阶段②：透视校正 + 网格提取
│   ├── blueprint.py       # 图纸解析
│   ├── color.py           # 221色盘颜色匹配
│   ├── compare.py         # 阶段③：差异对比 + 标注
│   ├── label_service.py   # 标注核心逻辑（FastSAM 分割 + YOLO 标签生成）
│   ├── label_ui.py        # Gradio 标注 Web UI
│   └── cli.py             # 命令行入口
├── training/
│   ├── semi_auto_label.py # 终端版半自动标注工具
│   ├── train.py           # 模型训练脚本
│   ├── validate.py        # 模型验证脚本
│   └── collect_helper.py  # 照片批量重命名
├── data/
│   ├── colors.json        # 221色盘定义
│   └── board_sizes.json   # 拼板尺寸规格
├── tests/                 # 单元测试（28 个）
├── docs/                  # 设计文档和实现计划
└── requirements.txt
```

## 技术栈

- **Python 3.10+**
- **ultralytics** — YOLOv8-seg 拼板检测 + FastSAM 半自动标注
- **OpenCV** — 透视校正、网格提取、标注绘制
- **NumPy** — LAB 色彩空间颜色匹配
- **Gradio** — 标注工具 Web UI

## 拼板尺寸

内置常见规格：

| 名称 | 尺寸 |
|------|------|
| Square Mini | 14×14 |
| Square Small | 21×21 |
| Square Medium | 29×29 |
| Square Large | 40×40 |
| Rectangle Small | 21×29 |
| Rectangle Large | 29×58 |

## License

MIT
