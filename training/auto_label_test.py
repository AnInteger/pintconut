"""
FastSAM 自动标注测试脚本
用 FastSAM 分割图片，自动找到拼板区域，验证自动标注的可行性。
"""

import os
import sys
import numpy as np
import cv2
from ultralytics import FastSAM

# 输入输出目录
INPUT_DIR = "training/test_images"
OUTPUT_DIR = "training/auto_label_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_rectangular(mask, threshold=0.85):
    """判断 mask 是否接近矩形（凸包面积 vs mask 面积）"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0
    contour = max(contours, key=cv2.contourArea)
    mask_area = cv2.contourArea(contour)
    if mask_area < 1000:  # 太小，忽略
        return False, 0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    ratio = mask_area / hull_area if hull_area > 0 else 0
    return ratio > threshold, ratio


def find_beadboard(results, img_shape):
    """从 FastSAM 结果中筛选出拼板区域"""
    h, w = img_shape[:2]
    img_area = h * w
    candidates = []

    if results[0].masks is None:
        print("  ⚠️  未检测到任何区域")
        return None

    masks = results[0].masks.data.cpu().numpy()
    print(f"  检测到 {len(masks)} 个区域")

    for i, mask in enumerate(masks):
        # resize mask to original image size
        mask_resized = cv2.resize(mask, (w, h))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        area = np.sum(mask_binary)
        area_ratio = area / img_area

        # 筛选条件
        is_rect, rect_ratio = is_rectangular(mask_binary)

        # 拼板通常占画面 15%-90%
        size_ok = 0.15 < area_ratio < 0.90
        rect_ok = is_rect

        print(f"  区域 {i:2d}: 面积占比={area_ratio:.1%}, 矩形度={rect_ratio:.2f}, "
              f"大小OK={size_ok}, 矩形OK={rect_ok}")

        if size_ok and rect_ok:
            candidates.append({
                "index": i,
                "mask": mask_binary,
                "area": area,
                "rect_ratio": rect_ratio,
                "area_ratio": area_ratio,
            })

    if not candidates:
        print("  ⚠️  没有找到符合条件的区域")
        return None

    # 按面积排序，取最大的
    candidates.sort(key=lambda c: c["area"], reverse=True)
    best = candidates[0]
    print(f"  ✅ 选中区域 {best['index']}: 面积占比={best['area_ratio']:.1%}, 矩形度={best['rect_ratio']:.2f}")
    return best


def save_result(image, result_info, output_path):
    """保存标注结果图片"""
    overlay = image.copy()
    mask = result_info["mask"]

    # 绿色半透明填充
    overlay[mask > 0] = [0, 255, 0]
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # 画轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 3)

    # 画凸包（蓝色）
    hull = cv2.convexHull(contours[0])
    cv2.drawContours(blended, [hull], -1, (255, 0, 0), 2)

    # 添加文字标注
    cv2.putText(blended,
                f"Board: {result_info['area_ratio']:.0%} area, rect={result_info['rect_ratio']:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(output_path, blended)
    print(f"  💾 保存到 {output_path}")


def main():
    print("=" * 60)
    print("FastSAM 自动标注测试")
    print("=" * 60)

    # 加载模型（首次会自动下载权重）
    print("\n📦 加载 FastSAM-s 模型...")
    model = FastSAM("FastSAM-s.pt")
    print("✅ 模型加载完成")

    # 处理每张测试图片
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\n🔍 共 {len(images)} 张测试图片\n")

    success_count = 0
    for img_file in images:
        img_path = os.path.join(INPUT_DIR, img_file)
        print(f"📷 处理: {img_file}")

        image = cv2.imread(img_path)
        if image is None:
            print(f"  ❌ 无法读取图片")
            continue

        # FastSAM 推理
        results = model(image, device="cpu", retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

        # 找拼板
        result_info = find_beadboard(results, image.shape)

        if result_info:
            output_path = os.path.join(OUTPUT_DIR, f"labeled_{img_file}")
            save_result(image, result_info, output_path)
            success_count += 1
        else:
            # 保存原图 + 所有检测到的区域用于调试
            output_path = os.path.join(OUTPUT_DIR, f"failed_{img_file}")
            debug_img = results[0].plot()
            cv2.imwrite(output_path, debug_img)
            print(f"  💾 调试图保存到 {output_path}")

        print()

    print("=" * 60)
    print(f"📊 结果: {success_count}/{len(images)} 张成功定位拼板")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
