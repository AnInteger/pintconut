"""
YOLO 检测模型验证脚本 - 对测试图片运行推理并可视化结果。

用法:
    python training/bead_validate.py --model training/runs/bead-v1/weights/best.pt --image test_photos/
    python training/bead_validate.py --model best.pt --image test.png --output training/validation_results --conf 0.5
"""

import argparse
import os
import sys

import cv2
import numpy as np


def validate_single(model, image_path, output_dir, conf_threshold):
    """对单张图片运行推理并保存可视化结果。

    Args:
        model: YOLO 模型实例。
        image_path: 图片路径。
        output_dir: 输出目录。
        conf_threshold: 置信度阈值。

    Returns:
        int: 检测到的目标数量。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERROR] 无法读取: {image_path}")
        return 0

    results = model(image, conf=conf_threshold)
    filename = os.path.basename(image_path)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        print(f"  [SKIP] 未检测到目标: {filename}")
        # 保存原图到输出目录
        output_path = os.path.join(output_dir, f"no_detect_{filename}")
        cv2.imwrite(output_path, image)
        return 0

    # 在图片上绘制检测框
    annotated = image.copy()
    boxes = results[0].boxes

    for box in boxes:
        # 获取边界框坐标和置信度
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        # 绘制边界框
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # 添加置信度标签
        label = f"{conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(int(y1) - 5, label_size[1] + 5)

        # 标签背景
        cv2.rectangle(
            annotated,
            (int(x1), label_y - label_size[1] - 3),
            (int(x1) + label_size[0] + 5, label_y + 3),
            (0, 255, 0),
            -1
        )

        # 标签文字
        cv2.putText(
            annotated, label,
            (int(x1) + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

    # 添加总检测数量文字
    n_boxes = len(boxes)
    cv2.putText(
        annotated, f"Detected: {n_boxes} beads",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    output_path = os.path.join(output_dir, f"detected_{filename}")
    cv2.imwrite(output_path, annotated)
    print(f"  [OK] {filename} -> {output_path} ({n_boxes} 个珠子)")
    return n_boxes


def main():
    parser = argparse.ArgumentParser(description="YOLO 检测模型验证 - 推理 + 可视化")
    parser.add_argument(
        "--model", required=True,
        help="模型权重文件路径 (如 best.pt)",
    )
    parser.add_argument(
        "--image", required=True,
        help="测试图片路径（单张图片或目录）",
    )
    parser.add_argument(
        "--output", default="training/validation_results",
        help="输出目录 (默认: training/validation_results)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="置信度阈值 (默认: 0.5)",
    )
    args = parser.parse_args()

    # 检查模型文件
    if not os.path.isfile(args.model):
        print(f"[ERROR] 模型文件不存在: {args.model}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    from ultralytics import YOLO

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    print("模型加载完成\n")

    # 收集测试图片
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    if os.path.isfile(args.image):
        image_paths = [args.image]
    elif os.path.isdir(args.image):
        image_paths = sorted(
            os.path.join(args.image, f)
            for f in os.listdir(args.image)
            if f.lower().endswith(exts)
        )
    else:
        print(f"[ERROR] 路径不存在: {args.image}")
        sys.exit(1)

    if not image_paths:
        print("[ERROR] 没有找到测试图片")
        sys.exit(1)

    print(f"共 {len(image_paths)} 张测试图片\n")

    # 推理
    total_beads = 0
    success_count = 0
    for img_path in image_paths:
        n_beads = validate_single(model, img_path, args.output, args.conf)
        total_beads += n_beads
        if n_beads > 0:
            success_count += 1

    print(f"\n验证完成:")
    print(f"  - {success_count}/{len(image_paths)} 张成功检测到目标")
    print(f"  - 总共检测到 {total_beads} 个珠子")
    print(f"  - 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
