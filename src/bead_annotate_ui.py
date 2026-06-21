"""Gradio UI for bead annotation — bead-grid prelabel + grid-assisted fill + box edit + export.

Thin UI over src.bead_label_service. Reads photos from training/photos/, exports YOLO
detection labels to training/bead_dataset/{images,labels}/train/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from src.bead_label_service import prelabel, holes_to_boxes, export_yolo, match_cell_colors
from src.color import ColorMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS_DIR = os.path.join(BASE_DIR, "training", "photos")
DATASET_DIR = os.path.join(BASE_DIR, "training", "bead_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")

_cm = None


def _color_matcher():
    global _cm
    if _cm is None:
        _cm = ColorMatcher()
    return _cm


# box source -> BGR draw color
SRC_BGR = {"detect": (0, 255, 0), "autofill": (0, 255, 255), "manual": (255, 128, 0)}


def _list_photos():
    if not os.path.isdir(PHOTOS_DIR):
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(f for f in os.listdir(PHOTOS_DIR) if os.path.splitext(f)[1].lower() in exts)


def _stats():
    n = len(os.listdir(LABELS_DIR)) if os.path.isdir(LABELS_DIR) else 0
    return f"数据集: {n} 张已标注 / {len(_list_photos())} 张照片"


def _box_label(b):
    return f"#{b['id']} {b['source']} @({b['cx']},{b['cy']})"


def _choices(state):
    return [_box_label(b) for b in state["boxes"]]


def _draw(state, show_color):
    img = state.get("img_bgr")
    if img is None:
        return None
    disp = img.copy()
    if show_color and state.get("result") is not None:
        for cc in match_cell_colors(state["result"], _color_matcher()):
            x, y = cc["xy"]
            bgr = [int(v) for v in reversed(cc["rgb"])]
            cv2.circle(disp, (int(x), int(y)), 5, bgr, -1)
    res = state.get("result")
    if res is not None and getattr(res, "outline", None) is not None:
        cv2.polylines(disp, [res.outline.astype(np.int32)], True, (0, 0, 255), 2)
    for b in state["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), SRC_BGR.get(b["source"], (255, 255, 255)), 2)
    return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)


def _new_state(img_bgr=None, name=""):
    return {"img_bgr": img_bgr, "boxes": [], "holes": [], "result": None,
            "name": name, "next_id": 0}


def _stamp_ids(next_id: int, boxes: list[dict]) -> tuple[list[dict], int]:
    """Assign each box a unique stable 'id'; return (new_boxes, next_id)."""
    out = []
    for b in boxes:
        nb = dict(b)
        nb["id"] = next_id
        next_id += 1
        out.append(nb)
    return out, next_id


def h_load(name, state):
    path = os.path.join(PHOTOS_DIR, name or "")
    img = cv2.imread(path) if name else None
    if img is None:
        return None, _new_state(), f"❌ 读不到 {name}", gr.update(), _stats()
    state = _new_state(img, os.path.splitext(name)[0])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), state, f"✅ 已加载 {name}", gr.update(choices=_choices(state)), _stats()


def h_prelabel(state, show_color):
    img = state.get("img_bgr")
    if img is None:
        return None, state, "❌ 请先加载照片", gr.update(), _stats()
    r = prelabel(img)
    boxes, nid = _stamp_ids(state.get("next_id", 0), r.boxes)
    state = {**state, "boxes": boxes, "holes": r.holes, "result": r.result, "next_id": nid}
    if r.fit_ok:
        msg = f"✅ 检出 {len(r.boxes)} 颗；可补漏 {len(r.holes)} 处"
    else:
        msg = f"⚠️ 检出 {len(r.boxes)} 颗，豆子太少无法拟合网格（仅显示框）"
    return _draw(state, show_color), state, msg, gr.update(choices=_choices(state)), _stats()


def h_autofill(state, show_color):
    if not state.get("holes"):
        return _draw(state, show_color), state, "⚠️ 无可补漏（先预标注）", gr.update(), _stats()
    new_boxes, nid = _stamp_ids(state.get("next_id", 0), holes_to_boxes(state["holes"]))
    state = {**state, "boxes": state["boxes"] + new_boxes, "holes": [], "next_id": nid}
    return _draw(state, show_color), state, f"✅ 已补漏，共 {len(state['boxes'])} 个框", gr.update(choices=_choices(state)), _stats()


def h_delete(state, sel, show_color):
    sel = sel or []
    keep = [b for b in state["boxes"] if _box_label(b) not in sel]
    state = {**state, "boxes": keep}
    return _draw(state, show_color), state, f"剩余 {len(keep)} 个框", gr.update(choices=_choices(state)), _stats()


def h_add_click(evt, state, radius, show_color):
    img = state.get("img_bgr")
    if img is None or evt is None:
        return _draw(state, show_color), state, gr.update()
    x, y = evt.index
    cx, cy = int(round(x)), int(round(y))
    r = int(radius)
    manual = [{"xyxy": [cx - r, cy - r, cx + r, cy + r],
               "cx": cx, "cy": cy, "width": 2 * r, "height": 2 * r, "source": "manual"}]
    new_boxes, nid = _stamp_ids(state.get("next_id", 0), manual)
    state = {**state, "boxes": state["boxes"] + new_boxes, "next_id": nid}
    return _draw(state, show_color), state, gr.update(choices=_choices(state))


def h_export(state, name_override):
    img = state.get("img_bgr")
    if img is None:
        return "❌ 未加载照片", _stats()
    if not state["boxes"]:
        return "❌ 无框可导出（先预标注/补漏/加框）", _stats()
    name = (name_override or state.get("name") or "").strip()
    if not name:
        return "❌ 缺导出名", _stats()
    img_path, lbl_path, n = export_yolo(img, state["boxes"], name, IMAGES_DIR, LABELS_DIR)
    return f"✅ 导出 {n} 框 → {lbl_path}", _stats()


def build_ui():
    with gr.Blocks(title="Pintconut 珠子标注") as app:
        gr.Markdown("# 🟡 Pintconut 珠子标注 (bead-grid)\n"
                    "加载 → 预标注 → (可选)网格补漏 → 校正 → 导出。  "
                    "框色：🟢检出 / 🟡网格补 / 🔵手动。勾选「叠加匹配色」可评估算法效果。")
        state = gr.State(_new_state())

        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Dropdown(choices=_list_photos(), label="照片 (training/photos/)")
                load_btn = gr.Button("📥 加载")
                show_color = gr.Checkbox(value=True, label="叠加匹配色(评估)")
                prelabel_btn = gr.Button("🔍 预标注", variant="primary")
                autofill_btn = gr.Button("🟡 网格补漏")
                box_list = gr.CheckboxGroup(choices=[], label="框列表 (勾选→删除)")
                delete_btn = gr.Button("🗑️ 删除选中")
                radius = gr.Number(value=10, label="点击加框半径(px)")
                name_tb = gr.Textbox(label="导出名 (留空用照片名)")
                export_btn = gr.Button("💾 导出 YOLO", variant="secondary")
            with gr.Column(scale=1):
                canvas = gr.Image(label="标注画布 (点击=加🔵框)", type="numpy")
                status = gr.Textbox(label="状态", interactive=False)
                stats = gr.Textbox(label="统计", interactive=False)

        # 先声明所有组件，再统一接线
        load_btn.click(h_load, [photo, state], [canvas, state, status, box_list, stats])
        prelabel_btn.click(h_prelabel, [state, show_color], [canvas, state, status, box_list, stats])
        autofill_btn.click(h_autofill, [state, show_color], [canvas, state, status, box_list, stats])
        delete_btn.click(h_delete, [state, box_list, show_color], [canvas, state, status, box_list, stats])
        canvas.select(h_add_click, [state, radius, show_color], [canvas, state, box_list])
        export_btn.click(h_export, [state, name_tb], [status, stats])
        app.load(fn=lambda: gr.update(choices=_list_photos()), outputs=photo)
        app.load(fn=_stats, outputs=stats)
    return app


if __name__ == "__main__":
    build_ui().launch(share=False)
