"""交互式调 conf/iou,实时看 YOLO 检测效果(拖滑块即时重画).

用法:
  python src/bead_tune.py <model.pt> [图或目录]
  python src/bead_tune.py               # 默认最新 best.pt + data/photos

交互:
  拖滑块 conf / iou -> 即时推理 + 重画检测框 + 显示检出数
  n / p 切换图片
  q 退出
"""
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Rectangle
from ultralytics import YOLO

# 禁用 matplotlib 默认快捷键(避免和 n/p/q 冲突)
for _k in ("save", "pan", "zoom", "quit", "back", "forward", "grid", "home"):
    plt.rcParams[f"keymap.{_k}"] = []


def find_latest_model():
    cands = sorted(glob.glob("runs/detect/**/weights/best.pt", recursive=True),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else find_latest_model()
    if not model_path or not os.path.exists(model_path):
        sys.exit(f"找不到模型(传 <model.pt>,或先训练生成 best.pt)")
    target = sys.argv[2] if len(sys.argv) > 2 else "data/photos"
    if os.path.isdir(target):
        images = sorted(p for p in glob.glob(os.path.join(target, "*"))
                        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                        and cv2.imread(p) is not None)
    else:
        images = [target]
    if not images:
        sys.exit("没找到图")

    print(f"模型: {model_path}")
    print(f"图: {len(images)} 张 (n/p 切图, 拖滑块调 conf/iou, q 退出)")
    model = YOLO(model_path)
    idx = [0]
    boxes_artists = []

    fig, ax = plt.subplots(figsize=(11, 9))
    plt.subplots_adjust(bottom=0.15)
    ax_conf = fig.add_axes([0.2, 0.09, 0.6, 0.025])
    ax_iou = fig.add_axes([0.2, 0.04, 0.6, 0.025])
    s_conf = Slider(ax_conf, "conf", 0.01, 0.5, valinit=0.08)
    s_iou = Slider(ax_iou, "iou", 0.3, 0.85, valinit=0.6)

    def redraw():
        for a in boxes_artists:
            try:
                a.remove()
            except Exception:
                pass
        boxes_artists.clear()
        img = cv2.imread(images[idx[0]])
        r = model(img, conf=s_conf.val, iou=s_iou.val, verbose=False)
        boxes = r[0].boxes
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            boxes_artists.append(ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                       fill=False, edgecolor="lime", lw=1)))
        ax.set_title(f"[{idx[0]+1}/{len(images)}] {os.path.basename(images[idx[0]])}  "
                     f"|  conf={s_conf.val:.2f} iou={s_iou.val:.2f}  ->  检出 {len(boxes)} 颗",
                     fontsize=10)
        fig.canvas.draw_idle()

    def load_img():
        img = cv2.imread(images[idx[0]])
        ax.clear()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_axis_off()
        redraw()

    def on_key(event):
        if event.key == "n" and idx[0] < len(images) - 1:
            idx[0] += 1
            load_img()
        elif event.key == "p" and idx[0] > 0:
            idx[0] -= 1
            load_img()

    s_conf.on_changed(lambda _: redraw())
    s_iou.on_changed(lambda _: redraw())
    fig.canvas.mpl_connect("key_press_event", on_key)
    load_img()
    plt.show()


if __name__ == "__main__":
    main()
