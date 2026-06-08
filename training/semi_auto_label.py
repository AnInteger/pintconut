"""
半自动标注工具 - 使用 FastSAM 分割图片，人工选择拼板区域，保存 YOLO 分割标签。

用法:
    python training/semi_auto_label.py --input training/photos --output training/dataset

操作:
    0-9  : 选择对应编号的候选区域作为拼板
    s    : 跳过当前图片
    q    : 退出程序
"""

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import FastSAM

# 候选区域颜色表 (BGR)
COLORS = [
    (0, 255, 0),    # 绿
    (255, 0, 0),    # 蓝
    (0, 0, 255),    # 红
    (255, 255, 0),  # 青
    (0, 255, 255),  # 黄
    (255, 0, 255),  # 品红
    (128, 255, 0),  # 浅绿
    (0, 128, 255),  # 橙
    (255, 128, 0),  # 浅蓝
    (128, 0, 255),  # 紫
]


def is_rectangular(mask, threshold=0.80):
    """判断 mask 是否接近矩形。

    通过比较 mask 面积与其凸包面积的比值来判断。比值越接近 1.0，
    说明形状越接近凸多边形（矩形是凸多边形的一种）。

    Args:
        mask: 二值 mask (uint8, 值为 0 或 1)。
        threshold: 矩形度阈值，默认 0.80。

    Returns:
        (bool, float): 是否满足矩形度要求，以及实际的矩形度比值。
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return False, 0.0
    contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(contour)
    if mask_area < 100:
        return False, 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False, 0.0
    ratio = mask_area / hull_area
    return ratio > threshold, ratio


def _draw_candidates(image, candidates):
    """在图片上绘制所有候选区域并标注编号。"""
    display = image.copy()
    for i, cand in enumerate(candidates):
        color = COLORS[i % len(COLORS)]
        mask = cand["mask"]
        # 半透明填充
        overlay = display.copy()
        overlay[mask > 0] = color
        cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)
        # 轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, color, 2)
        # 编号
        if contours:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 30, 30 + i * 40
        else:
            cx, cy = 30, 30 + i * 40
        label = str(i)
        cv2.putText(
            display, label, (cx - 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA,
        )
        cv2.putText(
            display, label, (cx - 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA,
        )
    return display


def _save_label(mask, image_path, output_images_dir, output_labels_dir, img_w, img_h):
    """将选中的 mask 保存为 YOLO 分割标签格式。

    YOLO 分割格式: class x1 y1 x2 y2 ... (归一化坐标)

    同时将原图复制到 output_images_dir。

    Args:
        mask: 二值 mask (uint8, 与原图同尺寸)。
        image_path: 原始图片路径。
        output_images_dir: 输出图片目录。
        output_labels_dir: 输出标签目录。
        img_w: 图片宽度。
        img_h: 图片高度。
    """
    basename = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]

    # 复制原图到输出目录
    import shutil
    dest_image = os.path.join(output_images_dir, basename + ext)
    if not os.path.exists(dest_image):
        shutil.copy2(image_path, dest_image)

    # 生成 YOLO 分割标签
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"  [WARN] 无有效轮廓，跳过标签保存: {basename}")
        return

    contour = max(contours, key=cv2.contourArea)
    # 简化轮廓点数，避免标签过大
    epsilon = 0.005 * cv2.arcLength(contour, True)
    contour = cv2.approxPolyDP(contour, epsilon, True)

    label_path = os.path.join(output_labels_dir, basename + ".txt")
    with open(label_path, "w") as f:
        coords = []
        for point in contour:
            x = point[0][0] / img_w
            y = point[0][1] / img_h
            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")
        f.write("0 " + " ".join(coords) + "\n")

    print(f"  [OK] 标签已保存: {label_path}")


def process_image(model, image_path, output_images_dir, output_labels_dir):
    """处理单张图片: 分割 -> 筛选候选 -> 用户选择 -> 保存标签。

    Args:
        model: FastSAM 模型实例。
        image_path: 输入图片路径。
        output_images_dir: 输出图片目录。
        output_labels_dir: 输出标签目录。

    Returns:
        str: "saved", "skipped", 或 "quit"
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERROR] 无法读取图片: {image_path}")
        return "skipped"

    h, w = image.shape[:2]
    img_area = h * w
    filename = os.path.basename(image_path)
    print(f"\n处理: {filename} ({w}x{h})")

    # FastSAM 推理
    results = model(image, device="cpu", retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

    if results[0].masks is None:
        print("  [WARN] 未检测到任何区域，跳过")
        return "skipped"

    masks = results[0].masks.data.cpu().numpy()

    # 筛选候选区域
    candidates = []
    for i, m in enumerate(masks):
        mask_resized = cv2.resize(m, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        area = np.sum(mask_binary)
        area_ratio = area / img_area

        # 面积过滤: 5% ~ 95%
        if area_ratio < 0.05 or area_ratio > 0.95:
            continue

        # 矩形度过滤
        rect_ok, rect_ratio = is_rectangular(mask_binary, threshold=0.5)
        if not rect_ok:
            continue

        candidates.append({
            "index": i,
            "mask": mask_binary,
            "area": area,
            "area_ratio": area_ratio,
            "rect_ratio": rect_ratio,
        })

    # 最多保留 10 个候选（对应按键 0-9）
    candidates = sorted(candidates, key=lambda c: c["area"], reverse=True)[:10]

    if not candidates:
        print("  [WARN] 没有符合条件的候选区域，跳过")
        return "skipped"

    print(f"  找到 {len(candidates)} 个候选区域:")
    for j, c in enumerate(candidates):
        print(
            f"    [{j}] 面积={c['area_ratio']:.1%}, 矩形度={c['rect_ratio']:.2f}"
        )

    # 显示候选区域
    display = _draw_candidates(image, candidates)
    cv2.imshow("Semi-Auto Label - press 0-9/s/q", display)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            return "quit"

        if key == ord("s"):
            print("  跳过")
            cv2.destroyAllWindows()
            return "skipped"

        if ord("0") <= key <= ord("9"):
            idx = key - ord("0")
            if idx < len(candidates):
                selected = candidates[idx]
                print(
                    f"  选中 [{idx}]: 面积={selected['area_ratio']:.1%}, "
                    f"矩形度={selected['rect_ratio']:.2f}"
                )
                _save_label(
                    selected["mask"], image_path,
                    output_images_dir, output_labels_dir, w, h,
                )
                cv2.destroyAllWindows()
                return "saved"
            else:
                print(f"  [{idx}] 不在范围内 (0-{len(candidates) - 1})")

    cv2.destroyAllWindows()
    return "skipped"


def main():
    parser = argparse.ArgumentParser(
        description="半自动标注工具: FastSAM 分割 + 人工选择拼板区域",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
操作说明:
  0-9  选择对应编号的候选区域
  s    跳过当前图片
  q    退出程序
        """,
    )
    parser.add_argument(
        "--input", default="training/photos",
        help="输入图片目录 (默认: training/photos)",
    )
    parser.add_argument(
        "--output", default="training/dataset",
        help="输出数据集目录 (默认: training/dataset)",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output

    if not os.path.isdir(input_dir):
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 创建输出目录结构
    output_images_dir = os.path.join(output_dir, "images", "train")
    output_labels_dir = os.path.join(output_dir, "labels", "train")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # 收集图片文件
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    images = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(exts)
    )
    if not images:
        print(f"[ERROR] 输入目录中没有图片: {input_dir}")
        sys.exit(1)

    print(f"共 {len(images)} 张图片待处理")
    print("加载 FastSAM-s 模型...")
    model = FastSAM("FastSAM-s.pt")
    print("模型加载完成\n")

    saved = 0
    skipped = 0

    for img_file in images:
        img_path = os.path.join(input_dir, img_file)
        result = process_image(model, img_path, output_images_dir, output_labels_dir)

        if result == "quit":
            break
        elif result == "saved":
            saved += 1
        else:
            skipped += 1

    print(f"\n完成: 已标注 {saved}, 跳过 {skipped}, 共处理 {saved + skipped}/{len(images)}")


if __name__ == "__main__":
    main()
