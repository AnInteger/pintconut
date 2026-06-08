"""
照片批量重命名工具 - 收集图片并统一命名。

用法:
    python training/collect_helper.py --input /path/to/raw/photos
    python training/collect_helper.py --input ~/Downloads/boards --output training/photos --prefix board
    python training/collect_helper.py --input ./tmp --move --prefix img

默认行为是复制文件到输出目录，使用 --move 参数改为移动。
"""

import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(
        description="照片批量重命名工具: 收集图片并统一为 {prefix}_{NNNN}{ext} 格式",
    )
    parser.add_argument(
        "--input", required=True,
        help="输入图片目录路径（必需）",
    )
    parser.add_argument(
        "--output", default="training/photos",
        help="输出目录 (默认: training/photos)",
    )
    parser.add_argument(
        "--prefix", default="board",
        help="文件名前缀 (默认: board)",
    )
    parser.add_argument(
        "--move", action="store_true",
        help="使用移动而非复制（默认: 复制）",
    )
    args = parser.parse_args()

    input_dir = args.input
    output_dir = args.output
    prefix = args.prefix
    do_move = args.move

    # 验证输入目录
    if not os.path.isdir(input_dir):
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 收集图片文件
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    image_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith(exts)
    )

    if not image_files:
        print(f"[ERROR] 输入目录中没有图片: {input_dir}")
        sys.exit(1)

    # 检查输出目录中已有文件的最大编号，避免覆盖
    existing_max = 0
    for f in os.listdir(output_dir):
        if f.startswith(prefix + "_"):
            try:
                num_part = os.path.splitext(f[len(prefix) + 1 :])[0]
                num = int(num_part)
                existing_max = max(existing_max, num)
            except (ValueError, IndexError):
                pass

    start_num = existing_max + 1
    action = "移动" if do_move else "复制"

    print(f"找到 {len(image_files)} 张图片")
    print(f"起始编号: {start_num:04d}")
    print(f"操作: {action}")
    print()

    count = 0
    for img_file in image_files:
        src = os.path.join(input_dir, img_file)
        ext = os.path.splitext(img_file)[1].lower()
        # 保持原始扩展名（统一为小写）
        new_name = f"{prefix}_{start_num + count:04d}{ext}"
        dst = os.path.join(output_dir, new_name)

        if do_move:
            shutil.move(src, dst)
        else:
            shutil.copy2(src, dst)

        print(f"  {img_file} -> {new_name}")
        count += 1

    print(f"\n完成: {action}了 {count} 张图片到 {output_dir}")


if __name__ == "__main__":
    main()
