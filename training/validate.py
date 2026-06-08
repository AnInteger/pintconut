"""
YOLO 分割模型验证脚本 - 对测试图片运行推理并可视化结果。

用法:
    python training/validate.py --model training/runs/beadboard-v1/weights/best.pt --image test_photos/
    python training/validate.py --model best.pt --image test.png --output training/validation_results
"""

import argparse
import os
import sys

import cv2
import numpy as np


def validate_single(model, image_path, output_dir):
    """对单张图片运行推理并保存可视化结果。

    Args:
        model: YOLO 模型实例。
        image_path: 图片路径。
        output_dir: 输出目录。

    Returns:
        bool: 是否成功检测到目标。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERROR] 无法读取: {image_path}")
        return False

    results = model(image)
    filename = os.path.basename(image_path)

    if results[0].masks is None or len(results[0].masks) == 0:
        print(f"  [SKIP] 未检测到目标: {filename}")
        # 保存原图到输出目录
        output_path = os.path.join(output_dir, f"no_detect_{filename}")
        cv2.imwrite(output_path, image)
        return False

    # 在图片上绘制 mask
    annotated = image.copy()
    masks = results[0].masks.data.cpu().numpy()
    h, w = image.shape[:2]

    for mask in masks:
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        # 绿色半透明覆盖
        colored = np.zeros_like(image)
        colored[:] = (0, 200, 0)
        colored_mask = cv2.bitwise_and(colored, colored, mask=mask_binary)
        # 叠加到原图
        mask_inv = cv2.bitwise_not(mask_binary)
        bg = cv2.bitwise_and(annotated, annotated, mask=mask_inv)
        annotated = cv2.addWeighted(bg, 1.0, colored_mask, 0.4, 0)
        # 画轮廓
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)

    # 添加文字
    n_masks = len(masks)
    cv2.putText(
        annotated, f"Detected: {n_masks}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )

    output_path = os.path.join(output_dir, f"detected_{filename}")
    cv2.imwrite(output_path, annotated)
    print(f"  [OK] {filename} -> {output_path} ({n_masks} 个区域)")
    return True


def main():
    parser = argparse.ArgumentParser(description="YOLO 分割模型验证 - 推理 + 可视化")
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
    success = 0
    for img_path in image_paths:
        if validate_single(model, img_path, args.output):
            success += 1

    print(f"\n验证完成: {success}/{len(image_paths)} 张成功检测到目标")
    print(f"结果保存在: {args.output}")


if __name__ == "__main__":
    main()
