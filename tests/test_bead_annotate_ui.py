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
        _FakeSelectData(123.0, 456.0), state, "角点", 10, False)
    assert len(state2["corners"]) == 1
    assert state2["corners"][0] == (123.0, 456.0)
    assert "1" in status and "4" in status   # "已点 1/4 ..."


def test_h_click_caps_at_four_corners():
    from src.bead_annotate_ui import h_click, _new_state

    state = _new_state(img_bgr=np.zeros((400, 400, 3), dtype=np.uint8), rows=4, cols=4)
    pts = [(10, 10), (300, 10), (300, 300), (10, 300), (50, 50)]  # 5th should be ignored
    for x, y in pts:
        _img, state, _b, _s = h_click(_FakeSelectData(x, y), state, "角点", 10, False)
    assert len(state["corners"]) == 4


def test_h_click_box_mode_adds_manual_box():
    from src.bead_annotate_ui import h_click, _new_state

    state = _new_state(img_bgr=np.zeros((400, 400, 3), dtype=np.uint8))
    _img, state, _b, _s = h_click(_FakeSelectData(200.0, 150.0), state, "加框", 12, False)
    assert len(state["boxes"]) == 1
    b = state["boxes"][0]
    assert b["source"] == "manual"
    assert b["cx"] == 200 and b["cy"] == 150
    assert b["width"] == 24 and b["height"] == 24   # 2 * radius(12)


def test_h_load_sets_gmag(tmp_path, monkeypatch):
    import src.bead_annotate_ui as ui
    monkeypatch.setattr(ui, "PHOTOS_DIR", str(tmp_path))
    cv2.imwrite(str(tmp_path / "t.png"), np.zeros((20, 20, 3), dtype=np.uint8))
    _img, state, _status, _bl, _st = ui.h_load("t.png", ui._new_state())
    assert state["gmag"] is not None
    assert state["gmag"].shape == (20, 20)
    assert state["pending"] is None
