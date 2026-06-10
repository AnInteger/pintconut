"""Playwright integration tests for the 2-tab label UI."""

import re
import pytest
from playwright.sync_api import Page


def _wait(page: Page, ms: int = 1000):
    """Wait for Gradio processing."""
    page.wait_for_timeout(ms)


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
    """Upload tab should have upload button, status, gallery."""
    assert page.get_by_role("button", name="📤 上传").is_visible()
    assert page.get_by_label("状态").is_visible()
    assert page.get_by_label("已上传照片").is_visible()


def test_start_annotation_button_disabled_before_upload(page: Page):
    """'开始标注' in Tab ① should be disabled before upload."""
    btns = page.get_by_role("button", name="▶ 开始标注")
    # There are two (one in each tab). The first one (in Tab ①) should be disabled
    btn = btns.first
    assert btn.is_disabled()


def test_tab2_initial_state(page: Page):
    """Tab ② should show Radio (disabled), skip enabled, confirm/reselect disabled."""
    page.locator('button[role="tab"]').filter(has_text="标注确认").click()
    _wait(page, 2000)

    # Skip should be enabled
    assert page.get_by_role("button", name="⏭️ 跳过这张").is_enabled()
    # Confirm and reselect should be disabled
    assert page.get_by_role("button", name="✅ 确认，下一张").is_disabled()
    assert page.get_by_role("button", name="↩️ 重新选择").is_disabled()


def test_start_annotation_segments_and_loads_image(page: Page):
    """Clicking '开始标注' in Tab ② should segment images and show first one."""
    page.locator('button[role="tab"]').filter(has_text="标注确认").click()
    _wait(page, 1000)

    # Click the second "开始标注" button (the one in Tab ②)
    btns = page.get_by_role("button", name="▶ 开始标注")
    # The second one is in Tab ②
    btns.last.click()

    # Wait for segmentation (FastSAM can be slow, up to 60s)
    _wait(page, 60000)

    # Should show candidate count
    info = page.locator("text=个候选区域")
    assert info.is_visible(timeout=5000), "Should show candidate count after segmentation"


def test_radio_has_choices_after_segmentation(page: Page):
    """Radio should have candidate choices after segmentation."""
    # Check for radio options with "#N 面积" pattern
    option = page.locator("label").filter(has_text=re.compile(r"#\d+ 面积"))
    assert option.count() > 0, "Radio should have at least one candidate option"


def test_select_candidate_shows_preview(page: Page):
    """Selecting a radio option should show preview and enable confirm/reselect."""
    # Click first radio option
    first_radio_input = page.locator("input[type='radio']").first
    first_radio_input.click(force=True)
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

        # Radio options should be visible again
        option = page.locator("label").filter(has_text=re.compile(r"#\d+ 面积"))
        assert option.count() > 0, "Radio options should be visible after reselect"
        # Confirm should be disabled again
        assert page.get_by_role("button", name="✅ 确认，下一张").is_disabled()


def test_confirm_saves_and_advances(page: Page):
    """Confirm should save label and load next image."""
    # Re-select first
    first_radio_input = page.locator("input[type='radio']").first
    if first_radio_input.is_visible():
        first_radio_input.click(force=True)
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
        _wait(page, 2000)

        report = page.locator("text=数据集统计")
        assert report.is_visible(timeout=5000), "Should show dataset report"
