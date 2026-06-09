"""
YOLO 检测模型训练脚本 - 用于珠子检测。

用法:
    python training/bead_train.py --data training/bead_dataset
    python training/bead_train.py --data training/bead_dataset --model yolov8n.pt --epochs 150 --batch 16
"""

import argparse
import os
import sys


def generate_data_yaml(dataset_dir):
    """生成 data.yaml 配置文件（如果不存在）。

    包含单个类别 "bead"，指向 dataset_dir 下的 train/valid 子目录。

    Args:
        dataset_dir: 数据集根目录路径。

    Returns:
        str: data.yaml 文件路径。
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"data.yaml 已存在: {yaml_path}")
        return yaml_path

    images_train = os.path.join(dataset_dir, "images", "train")
    images_valid = os.path.join(dataset_dir, "images", "valid")

    content = f"""# Bead 检测数据集
path: {os.path.abspath(dataset_dir)}
train: {os.path.abspath(images_train)}
val: {os.path.abspath(images_valid)}

nc: 1
names:
  0: bead
"""
    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"data.yaml 已生成: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="YOLO 检测模型训练 - 珠子检测")
    parser.add_argument(
        "--data", required=True,
        help="数据集目录路径（包含 images/train, images/valid 子目录）",
    )
    parser.add_argument(
        "--model", default="yolov8n.pt",
        help="基础模型权重 (默认: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=150,
        help="训练轮数 (默认: 150)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="输入图片尺寸 (默认: 640)",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="批次大小 (默认: 16)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="训练设备: cpu 或 0,1,2... (默认: cpu)",
    )
    parser.add_argument(
        "--project", default="training/runs",
        help="输出项目目录 (默认: training/runs)",
    )
    parser.add_argument(
        "--name", default="bead-v1",
        help="实验名称 (默认: bead-v1)",
    )
    args = parser.parse_args()

    # 验证数据集目录结构
    dataset_dir = args.data
    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] 数据集目录不存在: {dataset_dir}")
        sys.exit(1)

    train_dir = os.path.join(dataset_dir, "images", "train")
    valid_dir = os.path.join(dataset_dir, "images", "valid")

    for dir_path, label in [(train_dir, "images/train"), (valid_dir, "images/valid")]:
        if not os.path.isdir(dir_path):
            print(f"[ERROR] 缺少 {label} 目录: {dir_path}")
            sys.exit(1)

    # 生成 data.yaml
    data_yaml = generate_data_yaml(dataset_dir)

    # 加载模型并训练
    from ultralytics import YOLO

    print(f"\n加载模型: {args.model}")
    model = YOLO(args.model)

    print(f"开始训练: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}")
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        single_cls=True,
        patience=50,
    )

    # 输出最佳权重路径
    best_weights = os.path.join(args.project, args.name, "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"\n训练完成! 最佳权重: {best_weights}")
    else:
        # ultralytics 可能使用不同的输出路径
        save_dir = getattr(results, "save_dir", None)
        if save_dir:
            alt_path = os.path.join(save_dir, "weights", "best.pt")
            if os.path.exists(alt_path):
                print(f"\n训练完成! 最佳权重: {alt_path}")
            else:
                print(f"\n训练完成! 请在 {args.project}/{args.name}/ 下查找 best.pt")
        else:
            print(f"\n训练完成! 请在 {args.project}/{args.name}/ 下查找 best.pt")


if __name__ == "__main__":
    main()
