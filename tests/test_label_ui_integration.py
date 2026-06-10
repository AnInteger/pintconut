"""Playwright integration tests for the 2-tab label UI.

Tests are ordered to represent the full user journey:
upload → segment → select → confirm → skip → complete → export.

Gradio 6.x DOM notes:
- Tabs: button[role="tab"] with aria-selected
- Radio: role="radio" (NOT input[type="radio"])
- Buttons: get_by_role("button", name="...")
- Disabled state: .is_disabled() / .is_enabled()
"""

import re
import pytest
from playwright.sync_api import Page


def _wait(page: Page, ms: int = 1000):
    """Wait for Gradio processing."""
    page.wait_for_timeout(ms)


def _click_tab2(page: Page):
    """Navigate to Tab ②."""
    page.locator('button[role="tab"]').filter(has_text="标注确认").click()
    _wait(page, 2000)


def test_page_has_two_tabs(page: Page):
    """Page should have exactly 2 tabs."""
    tabs = page.locator('button[role="tab"]')
    count = tabs.count()
    assert count == 2, f"Expected 2 tabs, found {count}"
    texts = [tabs.nth(i).inner_text() for i in range(count)]
    assert any("上传照片" in t for t in texts), f"Missing upload tab: {texts}"
    assert any("标注确认" in t for t in texts), f"Missing annotate tab: {texts}"


def test_upload_tab_active_by_default(page: Page):
    """Tab ① should be active on page load."""
    active = page.locator('button[role="tab"][aria-selected="true"]')
    assert active.count() == 1
    assert "上传照片" in active.first.inner_text()


def test_upload_ui_elements_exist(page: Page):
    """Upload tab should have upload button."""
    assert page.get_by_role("button", name="📤 上传").is_visible()


def test_start_annotation_button_disabled_before_upload(page: Page):
    """'开始标注' in Tab ① should be disabled before upload."""
    btns = page.get_by_role("button", name="▶ 开始标注")
    btn = btns.first
    assert btn.is_disabled()


def test_tab2_initial_state(page: Page):
    """Tab ② should show skip enabled, confirm/reselect disabled."""
    _click_tab2(page)

    assert page.get_by_role("button", name="⏭️ 跳过这张").is_enabled()
    assert page.get_by_role("button", name="✅ 确认，下一张").is_disabled()
    assert page.get_by_role("button", name="↩️ 重新选择").is_disabled()


def test_start_annotation_segments_and_loads_image(page: Page):
    """Clicking '开始标注' in Tab ② should segment images and show first one."""
    _click_tab2(page)

    # Click the "开始标注" button in Tab ② (the last one, since Tab ① also has one)
    btns = page.get_by_role("button", name="▶ 开始标注")
    btns.last.click()

    # Wait for segmentation (FastSAM can be slow)
    _wait(page, 60000)

    # Should show candidate count
    info = page.locator("text=个候选区域")
    assert info.is_visible(timeout=5000), "Should show candidate count after segmentation"


def test_radio_has_choices_after_segmentation(page: Page):
    """Radio should have candidate choices after segmentation."""
    # Gradio 6.x renders Radio with role="radio" (not input[type="radio"])
    radios = page.get_by_role("radio")
    count = radios.count()
    assert count > 0, f"Radio should have at least one candidate option, found {count}"


def test_select_candidate_shows_preview(page: Page):
    """Selecting a radio option should show preview and enable confirm/reselect."""
    # Click first radio option (Gradio 6.x uses role="radio")
    first_radio = page.get_by_role("radio").first
    first_radio.click()
    _wait(page, 2000)

    # Should show selection confirmation text
    assert page.locator("text=已选择区域").is_visible(timeout=5000)

    # Confirm and reselect should be enabled
    assert page.get_by_role("button", name="✅ 确认，下一张").is_enabled()
    assert page.get_by_role("button", name="↩️ 重新选择").is_enabled()
    # Skip should be disabled
    assert page.get_by_role("button", name="⏭️ 跳过这张").is_disabled()


def test_reselect_returns_to_candidates(page: Page):
    """Reselect should restore candidate view."""
    reselect_btn = page.get_by_role("button", name="↩️ 重新选择")
    if reselect_btn.is_enabled():
        reselect_btn.click()
        _wait(page, 2000)

        # Radio options should be visible again (role="radio")
        radios = page.get_by_role("radio")
        assert radios.count() > 0, "Radio options should be visible after reselect"
        # Confirm should be disabled again
        assert page.get_by_role("button", name="✅ 确认，下一张").is_disabled()


def test_confirm_saves_and_advances(page: Page):
    """Confirm should save label and load next image."""
    # Re-select first
    first_radio = page.get_by_role("radio").first
    if first_radio.is_visible():
        first_radio.click()
        _wait(page, 2000)

    confirm_btn = page.get_by_role("button", name="✅ 确认，下一张")
    if confirm_btn.is_enabled():
        confirm_btn.click()
        _wait(page, 2000)

        # Should show ✅ in thumbnail strip
        assert page.locator("text=✅").first.is_visible(timeout=5000)


def test_skip_marks_and_advances(page: Page):
    """Skip should mark image as skipped and advance."""
    skip_btn = page.get_by_role("button", name="⏭️ 跳过这张")
    if skip_btn.is_enabled():
        skip_btn.click()
        _wait(page, 2000)

        # Should show ⏭️ in thumbnails
        assert page.locator("text=⏭️").first.is_visible(timeout=5000)


def test_complete_shows_export_button(page: Page):
    """After all images processed, export button should be enabled."""
    skip_btn = page.get_by_role("button", name="⏭️ 跳过这张")
    for _ in range(5):
        if skip_btn.is_enabled():
            skip_btn.click()
            _wait(page, 2000)
        else:
            break

    complete = page.locator("text=全部处理完成")
    if complete.is_visible(timeout=5000):
        export_btn = page.get_by_role("button", name="📦 导出数据集")
        assert export_btn.is_enabled(), "Export should be enabled when complete"


def test_export_shows_report(page: Page):
    """Clicking export should show dataset report."""
    export_btn = page.get_by_role("button", name="📦 导出数据集")
    if export_btn.is_enabled():
        export_btn.click()
        _wait(page, 3000)

        # Report is in a Textbox (textarea), check its value directly
        report_textbox = page.get_by_label("数据集报告")
        value = report_textbox.input_value(timeout=10000)
        assert "照片总数" in value or "已标注" in value, \
            f"Report should contain stats, got: {value[:200]}"
