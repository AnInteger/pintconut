"""Tests for the bead annotation UI handlers.

The crash-reproduction test guards the regression where clicking the canvas to
mark corners raised AttributeError: Gradio only injects a gr.SelectData event
object when the handler parameter is *annotated* as gr.SelectData. Without the
annotation, `evt` was bound positionally to the first input (the state dict),
and `evt.index` blew up.
"""
import inspect

import cv2
import numpy as np
from gradio.events import EventData


def test_h_click_annotates_evt_as_selectdata():
    """Reproduces the corner-click crash.

    Gradio injects the click event (SelectData) ONLY when the parameter carries
    the `: gr.SelectData` type hint. Assert that hint is present, else clicking
    raises AttributeError on `evt.index`.
    """
    from src.bead_annotate_ui import h_click

    ann = inspect.get_annotations(h_click)
    assert "evt" in ann, "h_click must annotate evt as gr.SelectData"
    assert issubclass(ann["evt"], EventData), "evt must be an EventData subclass"


class _FakeSelectData:
    """Minimal stand-in for gr.SelectData: index == (x, y) for an image click."""

    def __init__(self, x, y):
        self.index = (x, y)


def test_h_click_corner_mode_appends_click():
    from src.bead_annotate_ui import h_click, _new_state

    state = _new_state(img_bgr=np.zeros((600, 600, 3), dtype=np.uint8), rows=6, cols=6)
    _canvas, state2, _boxlist, status = h_click(
        _FakeSelectData(123.0, 456.0), state, "角点", False)
    assert len(state2["corners"]) == 1
    assert state2["corners"][0] == (123.0, 456.0)
    assert "1" in status and "4" in status   # "已点 1/4 ..."


def test_h_click_caps_at_four_corners():
    from src.bead_annotate_ui import h_click, _new_state

    state = _new_state(img_bgr=np.zeros((400, 400, 3), dtype=np.uint8), rows=4, cols=4)
    pts = [(10, 10), (300, 10), (300, 300), (10, 300), (50, 50)]  # 5th should be ignored
    for x, y in pts:
        _img, state, _b, _s = h_click(_FakeSelectData(x, y), state, "角点", False)
    assert len(state["corners"]) == 4


def _gmag_ring(size, cx, cy, ring_r, val=50.0):
    g = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    g[np.abs(r - ring_r) < 1.5] = val
    return g


def test_h_click_dianzhu_center_adds_auto_box():
    from src.bead_annotate_ui import h_click, _new_state
    state = _new_state(img_bgr=np.zeros((120, 120, 3), dtype=np.uint8))
    state["gmag"] = _gmag_ring(120, 60, 60, ring_r=15)   # clean ring → r≈15, warn=False
    _img, state2, _b, _s = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)
    assert len(state2["boxes"]) == 1
    b = state2["boxes"][0]
    assert b["source"] == "auto"
    assert b["cx"] == 60 and b["cy"] == 60
    assert b["warn"] is False
    assert state2["pending"] is not None      # awaiting possible edge-override


def test_h_click_dianzhu_edge_override():
    from src.bead_annotate_ui import h_click, _new_state
    state = _new_state(img_bgr=np.zeros((120, 120, 3), dtype=np.uint8))
    state["gmag"] = _gmag_ring(120, 60, 60, ring_r=15)
    _, state, _, _ = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)   # center
    # 2nd click 10px from center, inside pending box (r=15) → edge override
    _, state2, _, _ = h_click(_FakeSelectData(70.0, 60.0), state, "点豆", False)
    b = state2["boxes"][0]
    assert b["source"] == "manual"
    assert b["width"] == 20                    # 2 * 10
    assert state2["pending"] is None


def test_h_click_dianzhu_outside_is_new_bead():
    from src.bead_annotate_ui import h_click, _new_state
    state = _new_state(img_bgr=np.zeros((160, 160, 3), dtype=np.uint8))
    # two rings so both centers find a clean edge
    g = _gmag_ring(160, 60, 60, ring_r=15) + _gmag_ring(160, 110, 60, ring_r=15)
    state["gmag"] = g
    _, state, _, _ = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)   # bead 1
    # 2nd click 50px away → outside pending box (r=15) → new bead
    _, state2, _, _ = h_click(_FakeSelectData(110.0, 60.0), state, "点豆", False)
    assert len(state2["boxes"]) == 2
    assert state2["pending"] is not None       # new pending on bead 2


def test_h_load_sets_gmag(tmp_path, monkeypatch):
    import src.bead_annotate_ui as ui
    monkeypatch.setattr(ui, "PHOTOS_DIR", str(tmp_path))
    cv2.imwrite(str(tmp_path / "t.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    _img, state, _status, _bl, _st = ui.h_load("t.png", ui._new_state())
    assert state["gmag"] is not None
    assert state["gmag"].shape == (20, 20)
    assert state["pending"] is None


def test_h_reset_clears_pending_no_crash():
    from src.bead_annotate_ui import h_click, h_reset, _new_state
    state = _new_state(img_bgr=np.zeros((120, 120, 3), dtype=np.uint8))
    state["gmag"] = _gmag_ring(120, 60, 60, ring_r=15)
    _, state, _, _ = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)   # sets pending
    assert state["pending"] is not None
    _img, state, _bl, _s = h_reset(state, False)
    assert state["pending"] is None
    assert state["boxes"] == []
    # previously: next 点豆 click IndexError'd on boxes[-1]; now it starts fresh
    _, state2, _b, _s = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)
    assert len(state2["boxes"]) == 1


def test_h_delete_clears_pending():
    from src.bead_annotate_ui import h_click, h_delete, _new_state, _box_label
    state = _new_state(img_bgr=np.zeros((160, 160, 3), dtype=np.uint8))
    state["gmag"] = _gmag_ring(160, 60, 60, ring_r=15) + _gmag_ring(160, 110, 60, ring_r=15)
    _, state, _, _ = h_click(_FakeSelectData(60.0, 60.0), state, "点豆", False)    # bead1
    _, state, _, _ = h_click(_FakeSelectData(110.0, 60.0), state, "点豆", False)   # bead2 (pending)
    assert len(state["boxes"]) == 2
    sel = [_box_label(state["boxes"][1])]   # delete the pending bead
    _img, state, _bl, _s = h_delete(state, sel, False)
    assert state["pending"] is None          # was dangling before the fix
    assert len(state["boxes"]) == 1


def test_h_generate_clears_pending():
    from src.bead_annotate_ui import h_click, h_generate, _new_state
    state = _new_state(img_bgr=np.zeros((160, 160, 3), dtype=np.uint8), rows=4, cols=4)
    state["gmag"] = _gmag_ring(160, 80, 80, ring_r=15)
    _, state, _, _ = h_click(_FakeSelectData(80.0, 80.0), state, "点豆", False)   # stale pending
    assert state["pending"] is not None
    state["corners"] = [(20, 20), (140, 20), (140, 140), (20, 140)]
    _img, state, _bl, _s = h_generate(state, 4, 4, False)
    assert state["pending"] is None          # was dangling before the fix
    assert len(state["boxes"]) == 16
