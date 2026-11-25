import re
from typing import Any, Optional

from scraper.models import (
    STOCK_IN_STOCK_ONLINE,
    STOCK_IN_STOCK_STORE_ONLY,
    STOCK_OUT_OF_STOCK,
    STOCK_UNKNOWN,
)

_CURRENCY_SYMBOL_MAP = {
    "₹": "INR",
    "rs.": "INR",
    "rs": "INR",
    "inr": "INR",
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "¥": "JPY",
    "jpy": "JPY",
    "aed": "AED",
    "cad": "CAD",
    "aud": "AUD",
    "sgd": "SGD",
}


def normalize_stock_status(raw: Optional[str]) -> str:
    """Normalize raw availability strings or schema.org URLs into canonical stock statuses."""
    if not raw or not isinstance(raw, str):
        return STOCK_UNKNOWN

    text = raw.strip().lower()
    if not text:
        return STOCK_UNKNOWN

    # Handle schema.org URLs or short names
    schema_tail = text.rsplit("/", 1)[-1]
    schema_tail = re.sub(r"[^a-z]", "", schema_tail)

    if schema_tail in {"instock", "onlineonly", "limitedavailability", "preorder", "presale"}:
        return STOCK_IN_STOCK_ONLINE
    if schema_tail in {"instoreonly"}:
        return STOCK_IN_STOCK_STORE_ONLY
    if schema_tail in {"outofstock", "soldout", "discontinued", "backorder"}:
        return STOCK_OUT_OF_STOCK

    # Store-only checks (check before generic "in stock")
    store_only_patterns = [
        "in store only",
        "in-store only",
        "store only",
        "available in store",
        "available in stores",
        "find in store",
        "available at store",
    ]
    if any(p in text for p in store_only_patterns):
        return STOCK_IN_STOCK_STORE_ONLY

    # Out of stock checks (check before "in stock" since "not in stock" contains "in stock")
    out_of_stock_patterns = [
        "out of stock",
        "out-of-stock",
        "not in stock",
        "sold out",
        "currently unavailable",
        "unavailable",
        "notify me",
        "notify when available",
        "coming soon",
        "no longer available",
        "discontinued",
    ]
    if any(p in text for p in out_of_stock_patterns):
        return STOCK_OUT_OF_STOCK

    # In stock online checks
    in_stock_patterns = [
        "in stock",
        "in-stock",
        "available online",
        "add to cart",
        "add to bag",
        "buy now",
        "ships in",
        "ready to ship",
        "available",
    ]
    if any(p in text for p in in_stock_patterns):
        return STOCK_IN_STOCK_ONLINE

    return STOCK_UNKNOWN


def normalize_price(raw: Any) -> Optional[float]:
    """Extract a numeric float price from numbers or formatted currency strings."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if val >= 0 else None

    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    # Remove commas/spaces between digits, e.g. "1,23,456.78" -> "123456.78"
    cleaned = text.replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def normalize_currency(raw: Optional[str], fallback_text: Optional[str] = None) -> Optional[str]:
    """Normalize a currency code or symbol into an ISO-like uppercase code."""
    for candidate in (raw, fallback_text):
        if not candidate or not isinstance(candidate, str):
            continue
        text = candidate.strip()
        if not text:
            continue

        lower = text.lower()
        # Direct match
        if lower in _CURRENCY_SYMBOL_MAP:
            return _CURRENCY_SYMBOL_MAP[lower]

        # Check for 3-letter ISO code
        iso_match = re.search(r"\b([A-Z]{3})\b", text.upper())
        if iso_match and iso_match.group(1).lower() in _CURRENCY_SYMBOL_MAP:
            return iso_match.group(1)

        # Check for symbols inside text
        for symbol, code in _CURRENCY_SYMBOL_MAP.items():
            if symbol in lower:
                return code

        # If raw itself was a 3-letter alpha code, return uppercase
        if candidate is raw and re.fullmatch(r"[A-Za-z]{3}", text):
            return text.upper()

    return None

