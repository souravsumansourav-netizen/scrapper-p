from typing import Any, Dict
from bs4 import BeautifulSoup, Tag

from scraper.models import (
    STOCK_IN_STOCK_ONLINE,
    STOCK_IN_STOCK_STORE_ONLY,
    STOCK_OUT_OF_STOCK,
    STOCK_UNKNOWN,
)
from scraper.normalizer import normalize_currency, normalize_price
from scraper.parsers.base import first_text


def _is_hidden_or_modal(el: Tag) -> bool:
    """Return True if element or an ancestor is a hidden container or modal template."""
    for parent in [el, *el.parents]:
        if not isinstance(parent, Tag):
            continue
        classes = " ".join(parent.get("class") or []).lower()
        if any(c in classes for c in ("modal", "d-none", "hide", "hidden")):
            return True
        style = (parent.get("style") or "").replace(" ", "").lower()
        if "display:none" in style:
            return True
    return False


def _is_disabled(el: Tag) -> bool:
    classes = " ".join(el.get("class") or []).lower()
    return el.has_attr("disabled") or "disabled" in classes or el.get("aria-disabled") == "true"


def detect_tanishq_stock_playwright(page: Any) -> str:
    """Inspect live Playwright DOM visibility/enabled state for Tanishq stock."""
    # 1. Active, visible Add to Cart / Add to Bag / Buy Now always wins over Find In Store
    add_selectors = [
        "button.add-to-cart",
        "button[class*='add-to-cart']",
        "button.add-to-bag",
        "button[class*='add-to-bag']",
    ]
    for sel in add_selectors:
        loc = page.locator(sel)
        count = loc.count()
        for i in range(min(count, 5)):
            item = loc.nth(i)
            try:
                if not item.is_visible():
                    continue
                txt = (item.inner_text() or "").strip().lower()
                if any(k in txt for k in ("add to cart", "add to bag", "buy now")):
                    return STOCK_IN_STOCK_ONLINE if item.is_enabled() else STOCK_OUT_OF_STOCK
            except Exception:
                continue

    # 2. Check visible Find In Store button
    store_selectors = [
        "button.find-in-store-button",
        "button.fis-store-pdp-btn",
        "button:has-text('Find In Store')",
        "button:has-text('Find nearest Store')",
    ]
    for sel in store_selectors:
        loc = page.locator(sel)
        count = loc.count()
        for i in range(min(count, 3)):
            item = loc.nth(i)
            try:
                if item.is_visible():
                    return STOCK_IN_STOCK_STORE_ONLY
            except Exception:
                continue

    # 3. Check visible unavailability alert
    unavail_loc = page.locator(
        "text=/cannot be added to your cart|currently out of stock|currently unavailable|sold out/i"
    )
    for i in range(min(unavail_loc.count(), 3)):
        try:
            if unavail_loc.nth(i).is_visible():
                return STOCK_OUT_OF_STOCK
        except Exception:
            continue

    return STOCK_UNKNOWN


def parse_tanishq(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Domain-specific parser for tanishq.co.in product pages.

    Tanishq injects and toggles Add to Cart vs. Find In Store via JS, so if an
    explicit Add to Cart button is not present in raw HTML, escalate to Tier 3
    Playwright to inspect live element visibility.
    """
    title = first_text(
        soup,
        [
            "h1.product-name",
            "h1.pdp-product-name",
            "h1[class*='product']",
            "h1",
        ],
    )

    price_text = first_text(
        soup,
        [
            ".prices .sales .value",
            ".price .sales .value",
            "span.sales .value",
            ".product-price .value",
            "[data-price]",
            ".pdp-price",
        ],
    )
    price = normalize_price(price_text)
    currency = normalize_currency(None, price_text) or ("INR" if price is not None else None)

    stock_status = STOCK_UNKNOWN
    for btn in soup.select("button.add-to-cart, button[class*='add-to-cart']"):
        if _is_hidden_or_modal(btn):
            continue
        btn_text = btn.get_text(" ", strip=True).lower()
        if any(k in btn_text for k in ("add to cart", "add to bag", "buy now")):
            stock_status = STOCK_OUT_OF_STOCK if _is_disabled(btn) else STOCK_IN_STOCK_ONLINE
            break

    return {
        "title": title,
        "price": price,
        "currency": currency,
        "stock_status": stock_status,
        "requires_tier3": stock_status == STOCK_UNKNOWN,
    }
