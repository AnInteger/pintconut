"""集中路径常量 —— 代码从这里导入，不在各模块硬编码路径。

需要新路径时加到这里，其他模块 `from src.paths import XXX`。
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 原始照片（统一存放，标注工具从这里选图）
PHOTOS_DIR = os.path.join(BASE_DIR, "data", "photos")

# YOLO 标注/训练数据集
DATASET_DIR = os.path.join(BASE_DIR, "training", "bead_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

# 配置
COLORS_PATH = os.path.join(BASE_DIR, "data", "colors.json")
BOARD_SIZES_PATH = os.path.join(BASE_DIR, "data", "board_sizes.json")
