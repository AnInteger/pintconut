"""Gradio UI for bead annotation — corner-click grid generation + box edit + export.

Thin UI over src.bead_label_service. User clicks 4 corner beads + picks board size ->
geometry generates all bead boxes -> correct -> export YOLO labels.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gradio as gr
import numpy as np

from src.bead_label_service import generate_grid_boxes, preview_box_colors, export_yolo
from src.color import ColorMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTOS_DIR = os.path.join(BASE_DIR, "training", "photos")
DATASET_DIR = os.path.join(BASE_DIR, "training", "bead_dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")
BOARD_SIZES_PATH = os.path.join(BASE_DIR, "data", "board_sizes.json")


def _load_presets():
    try:
        with open(BOARD_SIZES_PATH) as f:
            data = json.load(f)
        labels = [f"{p['name']} ({p['rows']}×{p['cols']})" for p in data]
        return labels, data
    except Exception:
        return [], []


PRESET_CHOICES, PRESET_DATA = _load_presets()


def _parse_preset(label):
    for p in PRESET_DATA:
        if f"{p['name']} ({p['rows']}×{p['cols']})" == label:
            return p["rows"], p["cols"]
    return None


_cm = None


def _color_matcher():
    global _cm
    if _cm is None:
        _cm = ColorMatcher()
    return _cm


# box source -> BGR draw color
SRC_BGR = {"generated": (0, 255, 0), "manual": (255, 0, 0)}


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


def _new_state(img_bgr=None, name="", rows=29, cols=29):
    return {"img_bgr": img_bgr, "boxes": [], "corners": [], "rows": rows, "cols": cols,
            "name": name, "next_id": 0}


def _stamp_ids(next_id: int, boxes: list[dict]) -> tuple[list[dict], int]:
    out = []
    for b in boxes:
        nb = dict(b)
        nb["id"] = next_id
        next_id += 1
        out.append(nb)
    return out, next_id


def _draw(state, show_color):
    img = state.get("img_bgr")
    if img is None:
        return None
    disp = img.copy()
    if show_color and state["boxes"]:
        for cc in preview_box_colors(img, state["boxes"], _color_matcher()):
            x, y = cc["xy"]
            bgr = [int(v) for v in reversed(cc["rgb"])]
            cv2.circle(disp, (int(x), int(y)), 4, bgr, -1)
    corners = state.get("corners", [])
    if len(corners) >= 2:
        cv2.polylines(disp, [np.array(corners, dtype=np.int32)],
                      len(corners) >= 4, (0, 0, 255), 2)
    for i, (x, y) in enumerate(corners):
        cv2.circle(disp, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(disp, str(i + 1), (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    for b in state["boxes"]:
        x1, y1, x2, y2 = b["xyxy"]
        cv2.rectangle(disp, (x1, y1), (x2, y2), SRC_BGR.get(b["source"], (255, 255, 255)), 2)
    return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)


def h_load(name, state):
    path = os.path.join(PHOTOS_DIR, name or "")
    img = cv2.imread(path) if name else None
    if img is None:
        return None, _new_state(), f"❌ 读不到 {name}", gr.update(), _stats()
    state = _new_state(img, os.path.splitext(name)[0])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), state, f"✅ 已加载 {name}", gr.update(choices=_choices(state)), _stats()


def h_preset(label):
    rc = _parse_preset(label or "")
    if rc:
        return rc[0], rc[1]
    return gr.update(), gr.update()


def h_reset(state, show_color):
    state = {**state, "corners": [], "boxes": []}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), "已重置角点与框"


def h_click(evt, state, mode, radius, show_color):
    img = state.get("img_bgr")
    if img is None or evt is None:
        return _draw(state, show_color), state, gr.update(), ""
    x, y = evt.index
    if mode == "角点":
        corners = list(state.get("corners", []))
        if len(corners) < 4:
            corners.append((float(x), float(y)))
            state = {**state, "corners": corners}
        return _draw(state, show_color), state, gr.update(), f"已点 {len(state['corners'])}/4 角豆"
    cx, cy = int(round(x)), int(round(y))
    r = int(radius)
    manual = [{"xyxy": [cx - r, cy - r, cx + r, cy + r],
              "cx": cx, "cy": cy, "width": 2 * r, "height": 2 * r, "source": "manual"}]
    nb, nid = _stamp_ids(state.get("next_id", 0), manual)
    state = {**state, "boxes": state["boxes"] + nb, "next_id": nid}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), ""


def h_generate(state, rows, cols, show_color):
    corners = state.get("corners", [])
    rows_i, cols_i = int(rows or 0), int(cols or 0)
    if len(corners) != 4:
        return _draw(state, show_color), state, gr.update(), "❌ 请先在「角点」模式点满 4 颗角豆"
    if rows_i < 2 or cols_i < 2:
        return _draw(state, show_color), state, gr.update(), "❌ 板尺寸太小（rows/cols ≥ 2）"
    try:
        boxes = generate_grid_boxes(np.array(corners, dtype=np.float32), rows_i, cols_i)
    except cv2.error:
        return _draw(state, show_color), state, gr.update(), "❌ 4 颗角豆共线，请重置后重新点"
    nb, nid = _stamp_ids(state.get("next_id", 0), boxes)
    state = {**state, "boxes": nb, "next_id": nid, "rows": rows_i, "cols": cols_i}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), f"✅ 生成 {len(nb)} 颗豆位"


def h_delete(state, sel, show_color):
    sel = sel or []
    keep = [b for b in state["boxes"] if _box_label(b) not in sel]
    state = {**state, "boxes": keep}
    return _draw(state, show_color), state, gr.update(choices=_choices(state)), f"剩余 {len(keep)} 个框"


def h_export(state, name_override):
    img = state.get("img_bgr")
    if img is None:
        return "❌ 未加载照片", _stats()
    if not state["boxes"]:
        return "❌ 无框可导出（先生成网格或加框）", _stats()
    name = (name_override or state.get("name") or "").strip()
    if not name:
        return "❌ 缺导出名", _stats()
    img_path, lbl_path, n = export_yolo(img, state["boxes"], name, IMAGES_DIR, LABELS_DIR)
    return f"✅ 导出 {n} 框 → {lbl_path}", _stats()


def build_ui():
    with gr.Blocks(title="Pintconut 珠子标注") as app:
        gr.Markdown("# 🟡 Pintconut 珠子标注 (角点定位)\n"
                    "加载 → 「角点」模式点 4 颗角豆(左上→右上→右下→左下) → 选板尺寸 → 生成网格 → "
                    "「加框」模式校正 → 导出。框色：🟢生成 / 🔵手动。")
        state = gr.State(_new_state())

        with gr.Row():
            with gr.Column(scale=1):
                photo = gr.Dropdown(choices=_list_photos(), label="照片 (training/photos/)")
                load_btn = gr.Button("📥 加载")
                preset_dd = gr.Dropdown(choices=PRESET_CHOICES, label="板尺寸预设 (选则自动填下方)",
                                        value=PRESET_CHOICES[2] if len(PRESET_CHOICES) > 2 else None)
                with gr.Row():
                    rows_num = gr.Number(value=29, label="rows", precision=0)
                    cols_num = gr.Number(value=29, label="cols", precision=0)
                mode = gr.Radio(["角点", "加框"], value="角点", label="点击模式")
                show_color = gr.Checkbox(value=True, label="叠加匹配色(评估)")
                generate_btn = gr.Button("🎯 生成网格", variant="primary")
                reset_btn = gr.Button("♻️ 重置角点/框")
                box_list = gr.CheckboxGroup(choices=[], label="框列表 (勾选→删除)")
                delete_btn = gr.Button("🗑️ 删除选中")
                radius = gr.Number(value=10, label="加框半径(px, 仅加框模式)")
                name_tb = gr.Textbox(label="导出名 (留空用照片名)")
                export_btn = gr.Button("💾 导出 YOLO", variant="secondary")
            with gr.Column(scale=1):
                canvas = gr.Image(label="标注画布 (角点模式=点角豆 / 加框模式=加框)", type="numpy")
                status = gr.Textbox(label="状态", interactive=False)
                stats = gr.Textbox(label="统计", interactive=False)

        # 先声明组件，再统一接线
        load_btn.click(h_load, [photo, state], [canvas, state, status, box_list, stats])
        preset_dd.change(h_preset, [preset_dd], [rows_num, cols_num])
        generate_btn.click(h_generate, [state, rows_num, cols_num, show_color],
                           [canvas, state, box_list, status])
        reset_btn.click(h_reset, [state, show_color], [canvas, state, box_list, status])
        delete_btn.click(h_delete, [state, box_list, show_color], [canvas, state, box_list, status])
        canvas.select(h_click, [state, mode, radius, show_color], [canvas, state, box_list, status])
        export_btn.click(h_export, [state, name_tb], [status, stats])
        app.load(fn=lambda: gr.update(choices=_list_photos()), outputs=photo)
        app.load(fn=_stats, outputs=stats)
    return app


if __name__ == "__main__":
    build_ui().launch(share=False)
