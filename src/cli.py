"""
Command-line interface for Pintconut bead diff detector.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from src.color import ColorMatcher
from src.blueprint import parse_blueprint
from src.grid import PerspectiveCorrector, GridExtractor
from src.detect import BoardDetector
from src.compare import DiffComparator


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
        """,
    )
    parser.add_argument("--photo", required=True, help="拼豆照片路径")
    parser.add_argument("--blueprint", required=True, help="图纸图片路径")
    parser.add_argument("--board-size", required=True, help="拼板尺寸，格式为 ROWSxCOLS（如 29x29）")
    parser.add_argument("--model", default="models/beadboard-best.pt", help="模型权重路径")
    parser.add_argument("--output", default="annotated_result.jpg", help="输出图片路径")
    parser.add_argument("--color-tolerance", type=float, default=30.0, help="颜色匹配容差（LAB距离）")

    args = parser.parse_args()

    if not os.path.exists(args.photo):
        print(f"❌ 照片文件不存在: {args.photo}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.blueprint):
        print(f"❌ 图纸文件不存在: {args.blueprint}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}", file=sys.stderr)
        print(f"   请先训练模型，参见 training/train.py", file=sys.stderr)
        sys.exit(1)

    rows, cols = parse_board_size(args.board_size)

    print(f"📷 照片: {args.photo}")
    print(f"📋 图纸: {args.blueprint}")
    print(f"📐 拼板: {rows}×{cols}")
    print()

    print("📋 解析图纸...")
    blueprint_grid = parse_blueprint(args.blueprint, rows=rows, cols=cols)
    print(f"   图纸网格: {blueprint_grid.shape}")

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
    output_w = cols * 20
    output_h = rows * 20
    corrected = corrector.correct(photo, corners, output_size=(output_w, output_h))
    print(f"   校正后尺寸: {corrected.shape[1]}×{corrected.shape[0]}")

    print("🔲 提取网格...")
    extractor = GridExtractor()
    photo_grid = extractor.extract(corrected, rows=rows, cols=cols)
    print(f"   照片网格: {photo_grid.shape}")

    print("🔎 对比差异...")
    comparator = DiffComparator(color_tolerance=args.color_tolerance)
    diffs = comparator.compare(photo_grid, blueprint_grid)

    if not diffs:
        print("\n🎉 完美！没有发现差异！")
    else:
        print(f"\n⚠️  发现 {len(diffs)} 处差异：")
        for d in diffs:
            print(f"   行{d['row']+1} 列{d['col']+1}: "
                  f"实际={d['photo_color']} 应为 {d['blueprint_color']}")

    print(f"\n💾 保存标注图到 {args.output}...")
    annotated = comparator.annotate(corrected, diffs, rows=rows, cols=cols)
    cv2.imwrite(args.output, annotated)
    print(f"   ✅ 已保存")


if __name__ == "__main__":
    main()
