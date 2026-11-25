from typing import Any, Dict
from bs4 import BeautifulSoup

from scraper.models import (
    STOCK_IN_STOCK_ONLINE,
    STOCK_OUT_OF_STOCK,
    STOCK_UNKNOWN,
)
from scraper.normalizer import normalize_currency, normalize_price
from scraper.parsers.base import first_text


def parse_bluestone(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """Domain-specific fallback parser for bluestone.com product pages."""
    title = first_text(soup, ["h1", "[data-testid='product-title']"])

    price_text = first_text(
        soup,
        [
            "[data-testid='product-price']",
            ".final-price",
            ".discounted-price",
            ".price",
        ],
    )
    price = normalize_price(price_text)
    currency = normalize_currency(None, price_text) or ("INR" if price is not None else None)

    stock_status = STOCK_UNKNOWN
    for btn in soup.select("button, a"):
        text = btn.get_text(" ", strip=True).lower()
        if not text:
            continue
        classes = " ".join(btn.get("class") or []).lower()
        is_disabled = btn.has_attr("disabled") or "disabled" in classes or btn.get("aria-disabled") == "true"

        if any(k in text for k in ("buy now", "add to cart", "add to bag")):
            stock_status = STOCK_OUT_OF_STOCK if is_disabled else STOCK_IN_STOCK_ONLINE
            break
        if any(k in text for k in ("out of stock", "sold out", "notify me")):
            stock_status = STOCK_OUT_OF_STOCK
            break

    return {
        "title": title,
        "price": price,
        "currency": currency,
        "stock_status": stock_status,
    }

