"""
Command-line interface for Pintconut bead diff detector.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from src.color import ColorMatcher
from src.bead_detect import BeadDetector
from src.grid import PerspectiveCorrector
from src.detect import BoardDetector


def parse_board_size(size_str: str) -> tuple[int, int]:
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid board size format: {size_str}. Expected 'ROWSxCOLS' like '29x29'.")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Pintconut - 拼豆差异检测器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.cli --photo IMG_001.jpg --blueprint pattern.png --board-size 29x29
  python -m src.cli --photo IMG_001.jpg
        """,
    )
    parser.add_argument("--photo", required=True, help="拼豆照片路径")
    parser.add_argument("--blueprint", help="图纸图片路径（可选）")
    parser.add_argument("--board-size", help="拼板尺寸，格式为 ROWSxCOLS（如 29x29）")
    parser.add_argument("--model", default="models/beadboard-best.pt", help="拼板检测模型权重路径")
    parser.add_argument("--bead-model", default="models/bead-best.pt", help="豆子检测模型权重路径")
    parser.add_argument("--output", default="annotated_result.jpg", help="输出图片路径")
    parser.add_argument("--color-tolerance", type=float, default=30.0, help="颜色匹配容差（LAB距离）")

    args = parser.parse_args()

    if not os.path.exists(args.photo):
        print(f"❌ 照片文件不存在: {args.photo}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}", file=sys.stderr)
        print(f"   请先训练模型，参见 training/train.py", file=sys.stderr)
        sys.exit(1)

    # Optional blueprint validation
    if args.blueprint and not os.path.exists(args.blueprint):
        print(f"❌ 图纸文件不存在: {args.blueprint}", file=sys.stderr)
        sys.exit(1)

    print(f"📷 照片: {args.photo}")
    if args.blueprint:
        print(f"📋 图纸: {args.blueprint}")
    if args.board_size:
        rows, cols = parse_board_size(args.board_size)
        print(f"📐 拼板: {rows}×{cols}")
    print()

    print("🔍 检测拼板位置...")
    photo = cv2.imread(args.photo)
    if photo is None:
        print(f"❌ 无法读取照片: {args.photo}", file=sys.stderr)
        sys.exit(1)

    detector = BoardDetector(model_path=args.model)
    mask = detector.detect(photo)

    if mask is None:
        print("❌ 未能在照片中检测到拼板", file=sys.stderr)
        sys.exit(1)

    print("   ✅ 检测到拼板区域")

    print("📐 透视校正...")
    corners = detector.extract_corners(mask)
    if corners is None:
        print("❌ 无法提取拼板角点", file=sys.stderr)
        sys.exit(1)

    corrector = PerspectiveCorrector()
    # Use a default output size if board size not provided
    if args.board_size:
        rows, cols = parse_board_size(args.board_size)
        output_w = cols * 20
        output_h = rows * 20
    else:
        output_w = 800
        output_h = 800
    corrected = corrector.correct(photo, corners, output_size=(output_w, output_h))
    print(f"   校正后尺寸: {corrected.shape[1]}×{corrected.shape[0]}")

    print("🔎 检测豆子...")
    color_matcher = ColorMatcher()
    bead_detector = BeadDetector(model_path=args.bead_model)

    # Create board mask from the detected mask (resized to corrected image size)
    board_mask_resized = cv2.resize(mask, (output_w, output_h))

    beads = bead_detector.detect_beads(
        corrected,
        color_matcher=color_matcher,
        board_mask=board_mask_resized,
    )

    if not beads:
        print("❌ 未能在照片中检测到豆子", file=sys.stderr)
        sys.exit(1)

    print(f"   ✅ 检测到 {len(beads)} 个豆子")

    print("\n📊 豆子统计:")
    stats = BeadDetector.count_beads(beads)
    print(f"   总计: {stats['total']} 个豆子")
    if stats['by_color']:
        print(f"   颜色分布:")
        for color, count in sorted(stats['by_color'].items(), key=lambda x: -x[1]):
            print(f"   - {color}: {count}")

    print(f"\n💾 保存标注图到 {args.output}...")
    annotated = corrected.copy()

    for bead in beads:
        if bead.get("is_bead", True):
            xyxy = bead["xyxy"]
            x1, y1, x2, y2 = xyxy
            color_name = bead.get("color_name", "Unknown")

            # Draw green box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw color name text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_color = (255, 255, 255)
            bg_color = (0, 0, 0)

            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(color_name, font, font_scale, 1)

            # Draw background rectangle for text
            cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), bg_color, -1)

            # Draw text
            cv2.putText(annotated, color_name, (x1 + 2, y1 - 3), font, font_scale, text_color, 1)

    cv2.imwrite(args.output, annotated)
    print(f"   ✅ 已保存")


if __name__ == "__main__":
    main()
