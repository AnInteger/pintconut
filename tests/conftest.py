"""Pytest fixtures for Playwright integration tests of the Gradio label UI."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

import pytest
from playwright.sync_api import Page, Browser

# Port for test Gradio server (avoid conflict with dev server on 7860)
TEST_SERVER_PORT = 7899
TEST_SERVER_URL = f"http://127.0.0.1:{TEST_SERVER_PORT}"

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def gradio_server():
    """Start Gradio server for testing, yield URL, then shut down."""
    # Kill any existing process on the test port
    try:
        subprocess.run(["fuser", "-k", f"{TEST_SERVER_PORT}/tcp"],
                       capture_output=True, timeout=5)
    except Exception:
        pass
    time.sleep(0.5)

    env = os.environ.copy()
    env["GRADIO_SERVER_PORT"] = str(TEST_SERVER_PORT)
    env["GRADIO_SERVER_NAME"] = "127.0.0.1"

    proc = subprocess.Popen(
        [os.path.join(PROJECT_ROOT, ".venv/bin/python"),
         os.path.join(PROJECT_ROOT, "src/label_ui.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    # Wait for server to be ready (up to 30 seconds)
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{TEST_SERVER_URL}/")
            break
        except urllib.error.URLError:
            time.sleep(1)
    else:
        proc.terminate()
        stdout = proc.stdout.read().decode()
        stderr = proc.stderr.read().decode()
        raise RuntimeError(
            f"Gradio server failed to start.\nstdout: {stdout}\nstderr: {stderr}"
        )

    yield TEST_SERVER_URL

    # Cleanup: send SIGTERM
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def browser(gradio_server):
    """Launch a Chromium browser for the test module."""
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch()
    yield browser
    browser.close()
    pw.stop()


@pytest.fixture(scope="module")
def page(browser, gradio_server):
    """Provide a shared Playwright page for the entire test module.

    Tests are ordered as a sequential user journey, so they share state.
    """
    page = browser.new_page()
    page.goto(gradio_server, wait_until="domcontentloaded")
    # Wait for Gradio to fully render (WS keeps network alive, so networkidle never fires)
    page.wait_for_selector('button[role="tab"]', timeout=15000)
    page.wait_for_timeout(2000)
    yield page
    page.close()
