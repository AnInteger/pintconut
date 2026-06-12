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
from src.grid import PerspectiveCorrector, GridExtractor
from src.detect import BoardDetector
from src.edge_refiner import RefinerConfig, DetectionError
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
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="使用旧版管线（不做边缘精化，用于对比或回退）",
    )

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
    board_size = None
    if args.board_size:
        rows, cols = parse_board_size(args.board_size)
        board_size = (rows, cols)
        print(f"📐 拼板: {rows}×{cols}")
    print()

    print("🔍 检测拼板位置...")
    photo = cv2.imread(args.photo)
    if photo is None:
        print(f"❌ 无法读取照片: {args.photo}", file=sys.stderr)
        sys.exit(1)

    detector = BoardDetector(model_path=args.model)

    if args.legacy or board_size is None:
        _run_legacy(args, photo, detector, board_size)
    else:
        _run_refined(args, photo, detector, board_size)


def _run_refined(
    args,
    photo: np.ndarray,
    detector: BoardDetector,
    board_size: tuple[int, int],
) -> None:
    """New pipeline with edge refinement."""
    rows, cols = board_size

    try:
        detection = detector.refine(photo, board_size)
    except DetectionError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    conf = detection.confidence
    print(f"   ✅ 检测到拼板区域")
    print(f"   置信度: 对象={conf.q_object:.2f}  检测={conf.q_detection:.2f}  "
          f"综合={conf.total:.2f} ({conf.level})")

    if conf.total < 0.5:
        print("   ⚠ 置信度较低，结果可能不准确，建议重新拍摄")

    clipped = [e for e in detection.edges if e.is_clipped]
    if clipped:
        sides = [e.clip_side or f"边{e.edge_id}" for e in clipped]
        print(f"   截断边: {', '.join(sides)}")

    # Perspective correction
    print("📐 透视校正...")
    output_w = cols * 20
    output_h = rows * 20
    corrector = PerspectiveCorrector()
    corrected, transform_matrix = corrector.correct_with_matrix(
        photo, detection.corners, output_size=(output_w, output_h),
    )
    print(f"   校正后尺寸: {corrected.shape[1]}×{corrected.shape[0]}")

    # Warp visibility mask into corrected coordinates
    visibility_corrected = cv2.warpPerspective(
        detection.visibility_mask.astype(np.uint8),
        transform_matrix,
        (output_w, output_h),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)

    # Bead detection
    print("🔎 检测豆子...")
    color_matcher = ColorMatcher()
    bead_detector = BeadDetector(model_path=args.bead_model)
    board_mask_resized = cv2.resize(
        (detection.visibility_mask.astype(np.uint8) * 255),
        (output_w, output_h),
    )
    beads = bead_detector.detect_beads(
        corrected, color_matcher=color_matcher, board_mask=board_mask_resized,
    )
    if not beads:
        print("❌ 未能在照片中检测到豆子", file=sys.stderr)
        sys.exit(1)
    print(f"   ✅ 检测到 {len(beads)} 个豆子")

    # Blueprint comparison (if provided)
    if args.blueprint:
        print("📊 对比图纸...")
        from src.blueprint import parse_blueprint

        blueprint_img = cv2.imread(args.blueprint)
        bp_grid = parse_blueprint(blueprint_img, rows, cols)

        extractor = GridExtractor()
        cells = extractor.extract_with_visibility(
            corrected, rows, cols, visibility_corrected,
        )

        comparator = DiffComparator(color_tolerance=args.color_tolerance)
        diffs = comparator.compare_with_confidence(cells, bp_grid)

        reliable = [d for d in diffs if d.is_reliable]
        unreliable = [d for d in diffs if not d.is_reliable]
        print(f"   发现 {len(reliable)} 处可靠差异")
        if unreliable:
            print(f"   发现 {len(unreliable)} 处边缘区域差异（低置信度）")

        annotated = comparator.annotate_with_confidence(corrected, diffs, rows, cols)
    else:
        annotated = corrected.copy()

    # Draw bead boxes
    for bead in beads:
        if bead.get("is_bead", True):
            xyxy = bead["xyxy"]
            x1, y1, x2, y2 = xyxy
            color_name = bead.get("color_name", "Unknown")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(color_name, font, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(annotated, color_name, (x1 + 2, y1 - 3), font, font_scale, (255, 255, 255), 1)

    print(f"\n💾 保存标注图到 {args.output}...")
    cv2.imwrite(args.output, annotated)
    print(f"   ✅ 已保存")


def _run_legacy(
    args,
    photo: np.ndarray,
    detector: BoardDetector,
    board_size: tuple[int, int] | None,
) -> None:
    """Legacy pipeline (detect → extract_corners → correct)."""
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
    if board_size:
        rows, cols = board_size
        output_w, output_h = cols * 20, rows * 20
    else:
        output_w, output_h = 800, 800
    corrected = corrector.correct(photo, corners, output_size=(output_w, output_h))
    print(f"   校正后尺寸: {corrected.shape[1]}×{corrected.shape[0]}")

    print("🔎 检测豆子...")
    color_matcher = ColorMatcher()
    bead_detector = BeadDetector(model_path=args.bead_model)
    board_mask_resized = cv2.resize(mask, (output_w, output_h))
    beads = bead_detector.detect_beads(
        corrected, color_matcher=color_matcher, board_mask=board_mask_resized,
    )
    if not beads:
        print("❌ 未能在照片中检测到豆子", file=sys.stderr)
        sys.exit(1)
    print(f"   ✅ 检测到 {len(beads)} 个豆子")

    print("\n📊 豆子统计:")
    stats = BeadDetector.count_beads(beads)
    print(f"   总计: {stats['total']} 个豆子")
    if stats['by_color']:
        print("   颜色分布:")
        for color, count in sorted(stats['by_color'].items(), key=lambda x: -x[1]):
            print(f"   - {color}: {count}")

    print(f"\n💾 保存标注图到 {args.output}...")
    annotated = corrected.copy()
    for bead in beads:
        if bead.get("is_bead", True):
            xyxy = bead["xyxy"]
            x1, y1, x2, y2 = xyxy
            color_name = bead.get("color_name", "Unknown")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), _ = cv2.getTextSize(color_name, font, font_scale, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 0), -1)
            cv2.putText(annotated, color_name, (x1 + 2, y1 - 3), font, font_scale, (255, 255, 255), 1)

    cv2.imwrite(args.output, annotated)
    print("   ✅ 已保存")


if __name__ == "__main__":
    main()
