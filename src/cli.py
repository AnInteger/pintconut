"""
Command-line interface for Pintconut bead diff detector.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from src.color import ColorMatcher
from src.compare import DiffComparator
from src.bead_grid import BeadGridFitter, GridFitError


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
    parser.add_argument("--output", default="annotated_result.jpg", help="输出图片路径")
    parser.add_argument("--color-tolerance", type=float, default=30.0, help="颜色匹配容差（LAB距离）")

    args = parser.parse_args()

    if not os.path.exists(args.photo):
        print(f"❌ 照片文件不存在: {args.photo}", file=sys.stderr)
        sys.exit(1)

    # Optional blueprint validation
    if args.blueprint and not os.path.exists(args.blueprint):
        print(f"❌ 图纸文件不存在: {args.blueprint}", file=sys.stderr)
        sys.exit(1)

    print(f"📷 照片: {args.photo}")
    if args.blueprint:
        print(f"📋 图纸: {args.blueprint}")
    board_size = None
    if args.board_size:
        rows, cols = parse_board_size(args.board_size)
        board_size = (rows, cols)
        print(f"📐 拼板: {rows}×{cols}（提供，用于截断补全）")
    print()

    photo = cv2.imread(args.photo)
    if photo is None:
        print(f"❌ 无法读取照片: {args.photo}", file=sys.stderr)
        sys.exit(1)

    _run_bead_grid(args, photo, board_size)


def _run_bead_grid(args, photo, board_size):
    fitter = BeadGridFitter()
    try:
        result = fitter.fit(photo, board_size=board_size)
    except GridFitError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    conf = result.confidence
    print(f"   ✅ 拟合豆子网格: {result.rows}×{result.cols}")
    print(f"   豆子数={conf.bead_count}  填充率={conf.grid_fill_ratio:.2f}  "
          f"透视分级={conf.perspective_tier}  置信度={conf.level}")
    if result.truncation.is_truncated:
        print(f"   截断: {', '.join(result.truncation.clipped_edges)}")

    annotated = photo.copy()

    if args.blueprint:
        print("📊 对比图纸...")
        from src.blueprint import parse_blueprint
        blueprint_img = cv2.imread(args.blueprint)
        bp_grid = parse_blueprint(blueprint_img, result.rows, result.cols)
        comparator = DiffComparator(color_tolerance=args.color_tolerance)
        diffs = comparator.compare_with_confidence(result.cells, bp_grid)
        reliable = [d for d in diffs if d.is_reliable]
        unreliable = [d for d in diffs if not d.is_reliable]
        print(f"   可靠差异 {len(reliable)} 处；边缘/低置信 {len(unreliable)} 处")
        annotated = comparator.annotate_with_confidence(photo, diffs, result.rows, result.cols)
    else:
        cv2.polylines(annotated, [result.outline.astype(np.int32)], True, (0, 255, 0), 3)

    print(f"\n💾 保存到 {args.output}...")
    cv2.imwrite(args.output, annotated)
    print("   ✅ 已保存")


if __name__ == "__main__":
    main()
