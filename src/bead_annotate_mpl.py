"""matplotlib 纯手动点豆标注 — 可缩放 + 画圆 + 实时预览。

不跑算法（脏图 ballooning 不可靠）。点中心→点豆缘→准圆，像 gt_annotate。
WSLg 下 TkAgg 可靠 + 自带缩放工具栏（放大镜看清豆子，点击坐标不受缩放影响）。

交互：
  左键 点1: 点豆子中心（出现红十字）
  左键 点2: 点豆缘 -> 半径 = 中心到该点距离 -> 锁定蓝圆
            （鼠标移动时实时显示金黄虚线预览圆 + 半径数值）
  右键    : 删除离光标最近的豆
  n 下一张 / p 上一张 / u 撤销 / c 清空 / s 保存(导出YOLO) / q 退出
  工具栏  : 🔍 放大镜(框选放大) / ✋ 平移 —— 看清豆子用，不影响点击坐标

用法:
  python src/bead_annotate_mpl.py <目录>     # 批量遍历
  python src/bead_annotate_mpl.py <image>    # 单张
  python src/bead_annotate_mpl.py            # 默认 training/photos/
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 禁用 matplotlib 默认工具栏快捷键(s=save图/p=pan/o=zoom/q=quit...), 和标注键冲突
for _k in ("save", "pan", "zoom", "quit", "back", "forward", "grid", "grid_minor",
           "fullscreen", "home", "yscale", "xscale", "copy"):
    plt.rcParams[f"keymap.{_k}"] = []

from src.bead_label_service import export_yolo
from src.paths import DATASET_DIR

img = name = ax = fig = None
boxes = []        # [{cx, cy, r, source, warn}]
pending = None    # (cx, cy) of center awaiting edge click
images = []       # list of (path, split) — 要标的图(train/valid)
idx = [0]
cur_split = "train"
_press = [None]   # (display_x, display_y, button) at press — tell click from drag
mouse = [0, 0]
_last_draw = [0.0]   # on_move 限频用(避免 draw_idle 阻塞按键)
_pred_model = [None]  # YOLO 预填模型(lazy 加载)


preview_artists = []   # 预览(中心十字+金黄虚线圆+r)单独管理, on_move 只更新它不全重画


def _draw_preview():
    """只重画预览, 不动已标 boxes —— on_move 调这个省卡。"""
    for a in preview_artists:
        try:
            a.remove()
        except Exception:
            pass
    preview_artists.clear()
    if pending is None:
        return
    cx, cy = pending[0], pending[1]
    pr = int(((cx - mouse[0]) ** 2 + (cy - mouse[1]) ** 2) ** 0.5)
    preview_artists.append(ax.add_patch(Circle((cx, cy), pr, fill=False, color="gold", lw=1, ls="--")))
    preview_artists.extend(ax.plot([cx], [cy], "+", color="red", ms=14, mew=2))
    preview_artists.append(ax.text(cx + 6, cy - 6, f"r={pr}", color="gold", fontsize=9))


def redraw():
    for p in list(ax.patches):
        p.remove()
    for ln in list(ax.lines):
        ln.remove()
    for t in list(ax.texts):
        t.remove()
    preview_artists.clear()                        # patches 已清, 预览跟踪也清(下面 _draw_preview 重画)
    for b in boxes:
        c = "deepskyblue"
        ax.add_patch(Circle((b["cx"], b["cy"]), b["r"], fill=False, color=c, lw=2))
        ax.plot(b["cx"], b["cy"], "o", color=c, ms=3)
        ax.text(b["cx"] + b["r"] + 2, b["cy"], str(b["r"]), color=c, fontsize=7)
    _draw_preview()
    cur = f"[{idx[0]+1}/{len(images)}] {os.path.basename(images[idx[0]][0])}  " if images else ""
    ax.set_title(f"{cur}{len(boxes)} beads  |  LEFT=center->edge  RIGHT=delete  "
                 f"n/p u/c/s/q", fontsize=9)
    fig.canvas.draw_idle()


def _load_existing(name, shape, split):
    """加载该图已保存的 YOLO 标注,方便继续标/微调。"""
    lp = os.path.join(DATASET_DIR, "labels", split, name + ".txt")
    if not os.path.exists(lp):
        return []
    H, W = shape[:2]
    out = []
    for line in open(lp):
        p = line.split()
        if len(p) == 5:
            _c, cx, cy, w, h = p
            cx, cy, w = float(cx) * W, float(cy) * H, float(w) * W
            out.append({"cx": int(round(cx)), "cy": int(round(cy)), "r": int(round(w / 2)),
                        "source": "manual", "warn": False})
    return out


def load_current():
    global img, name, boxes, pending, cur_split
    path, cur_split = images[idx[0]]
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    boxes = _load_existing(name, img.shape, cur_split)
    pending = None
    ax.clear()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    redraw()


def _do_label(button, x, y):
    """左键1=点中心(新建)/点已有框内(选中改); 左键2=点豆缘定半径; 右键=删最近。"""
    global pending
    if button == 1:
        if pending is None:
            hit = None
            for i, b in enumerate(boxes):           # 落在已有框内 -> 选中改半径
                if (b["cx"] - x) ** 2 + (b["cy"] - y) ** 2 < b["r"] * b["r"]:
                    hit = i
                    break
            if hit is not None:
                b = boxes[hit]
                pending = (b["cx"], b["cy"], hit)
            else:
                pending = (x, y, None)              # 新建
        else:
            cx, cy, idx = pending
            r = int(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5)
            if r >= 3:
                if idx is None:
                    boxes.append({"cx": cx, "cy": cy, "r": r, "source": "manual", "warn": False})
                else:
                    boxes[idx].update(r=r, source="manual")   # 改选中框半径
            pending = None
    elif button == 3 and boxes:
        i = min(range(len(boxes)),
                key=lambda k: (boxes[k]["cx"] - x) ** 2 + (boxes[k]["cy"] - y) ** 2)
        boxes.pop(i)
        pending = None
    redraw()


def on_press(event):
    if event.inaxes is ax and event.xdata is not None:
        _press[0] = (event.x, event.y, event.button)


def on_release(event):
    p = _press[0]
    _press[0] = None
    if p is None or event.inaxes is not ax or event.xdata is None:
        return
    dx, dy = event.x - p[0], event.y - p[1]
    if dx * dx + dy * dy > 25:                     # 拖拽 = zoom/pan, 忽略
        return
    _do_label(p[2], int(event.xdata), int(event.ydata))


def prefill():
    """SAHI 切图预填(按 f 触发, 密集小豆子切图放大检出)"""
    global pending
    if img is None:
        return
    pending = None
    boxes.clear()                                   # 清空重预填(避免和已有重复)
    if _pred_model[0] is None:
        import glob as _glob
        cands = sorted(_glob.glob("runs/detect/**/weights/best.pt", recursive=True),
                       key=os.path.getmtime, reverse=True)
        if not cands:
            print("没找到 best.pt, 先训练一个")
            return
        from sahi import AutoDetectionModel
        print(f"SAHI 预填加载模型: {cands[0]}")
        _pred_model[0] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=cands[0],
            confidence_threshold=0.08, device="cuda:0", image_size=1280)
    from sahi.predict import get_sliced_prediction
    result = get_sliced_prediction(img, _pred_model[0], slice_height=512, slice_width=512,
                                   overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0)
    n = 0
    for pred in result.object_prediction_list:
        x1, y1, x2, y2 = pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
        boxes.append({"cx": int((x1 + x2) / 2), "cy": int((y1 + y2) / 2),
                      "r": int((x2 - x1) / 2), "source": "auto", "warn": False})
        n += 1
    redraw()
    print(f"SAHI 切图预填 {n} 颗 (右键删误报, u撤销, 点中心+豆缘补漏, s保存)")


def on_move(event):
    if event.inaxes is ax and event.xdata is not None:
        mouse[0], mouse[1] = event.xdata, event.ydata
        if pending is not None:                    # 预览限频(50ms), 避免 draw_idle 阻塞按键
            now = time.time()
            if now - _last_draw[0] > 0.05:
                _draw_preview()
                fig.canvas.draw_idle()
                _last_draw[0] = now


def on_key(event):
    global pending
    if event.key == "n" and idx[0] < len(images) - 1:
        idx[0] += 1
        load_current()
    elif event.key == "p" and idx[0] > 0:
        idx[0] -= 1
        load_current()
    elif event.key == "u":
        if pending is not None:
            pending = None                         # 先取消待定点
        elif boxes:
            boxes.pop()
        redraw()
    elif event.key == "c":
        boxes.clear()
        pending = None
        redraw()
    elif event.key == "f":
        prefill()
    elif event.key == "s" and boxes:
        xyxy = [{"xyxy": [b["cx"] - b["r"], b["cy"] - b["r"], b["cx"] + b["r"], b["cy"] + b["r"]],
                 "cx": b["cx"], "cy": b["cy"], "width": 2 * b["r"], "height": 2 * b["r"]}
                for b in boxes]
        img_dir = os.path.join(DATASET_DIR, "images", cur_split)
        lbl_dir = os.path.join(DATASET_DIR, "labels", cur_split)
        ip, lp, n = export_yolo(img, xyxy, name, img_dir, lbl_dir)
        print(f"\n[saved] {n} beads:")
        print(f"   image:  {ip}")
        print(f"   labels: {lp}\n")
        ax.set_title(f"[saved] {n} beads -> {os.path.basename(lp)}   (n=next)",
                     fontsize=9, color="green")
        fig.canvas.draw_idle()
    elif event.key == "q":
        plt.close(fig)


def main():
    global ax, fig, images
    # 默认扫 bead_dataset/images/{train,valid}(要标的图);或传单张图
    images = []
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        images = [(sys.argv[1], "train")]
    else:
        base_imgs = os.path.join(DATASET_DIR, "images")
        for split in ["train", "valid"]:
            d = os.path.join(base_imgs, split)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    images.append((os.path.join(d, f), split))
    images = [(p, s) for p, s in images if cv2.imread(p) is not None]
    if not images:
        sys.exit(f"没找到要标的图(在 {DATASET_DIR}/images/{{train,valid}})")
    fig, ax = plt.subplots()
    load_current()
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("LEFT=center, LEFT=edge(=radius)  RIGHT=delete  u=undo c=clear s=save n=next p=prev q=quit")
    try:
        fig.canvas.manager.window.attributes('-topmost', True)   # 窗口置顶, 确保弹到最前
    except Exception:
        pass
    plt.show()


if __name__ == "__main__":
    main()
