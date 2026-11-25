import time
from typing import Any, Callable, Optional, Tuple

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

from scraper.models import STOCK_UNKNOWN

DEFAULT_PLAYWRIGHT_TIMEOUT_MS = 15000


def fetch_html_tier3(
    url: str,
    timeout_ms: int = DEFAULT_PLAYWRIGHT_TIMEOUT_MS,
    stock_checker: Optional[Callable[[Any], str]] = None,
) -> Tuple[Optional[str], str, Optional[str]]:
    """Fetch fully rendered HTML using headless Chromium via Playwright.

    If `stock_checker` is provided, runs it against the live Playwright `page`
    so element visibility (`is_visible`, `is_enabled`) can be evaluated directly.
    Returns (html, live_stock_status, error_message).
    """
    deadline = time.monotonic() + max(1.0, timeout_ms / 1000.0)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-IN",
            )
            page = context.new_page()
            try:
                remaining_ms = max(1000, int((deadline - time.monotonic()) * 1000))
                page.goto(url, wait_until="domcontentloaded", timeout=remaining_ms)

                remaining_ms = int((deadline - time.monotonic()) * 1000)
                if remaining_ms > 500:
                    try:
                        page.wait_for_load_state(
                            "networkidle",
                            timeout=min(2500, remaining_ms),
                        )
                    except PlaywrightTimeoutError:
                        # Many e-commerce sites keep analytics/chat sockets open; proceed with current DOM.
                        pass

                live_stock = STOCK_UNKNOWN
                if stock_checker is not None:
                    live_stock = stock_checker(page)

                html = page.content()
                return html, live_stock, None
            finally:
                context.close()
                browser.close()
    except Exception as exc:
        return None, STOCK_UNKNOWN, str(exc)
