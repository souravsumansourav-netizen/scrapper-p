import time
from typing import Any, Dict

from bs4 import BeautifulSoup

from scraper.models import ProductResult, STOCK_UNKNOWN
from scraper.parsers import (
    get_domain_parser,
    get_domain_playwright_stock_checker,
    run_domain_parser,
)
from scraper.tier1_http import fetch_html_tier1, parse_structured_metadata
from scraper.tier3_playwright import fetch_html_tier3

MAX_TOTAL_TIMEOUT_SECONDS = 15.0


def _merge_fields(
    target: ProductResult,
    source: Dict[str, Any],
    override_stock: bool = False,
) -> bool:
    """Merge non-empty fields from `source` into `target`. Return True if any field updated."""
    updated = False
    if target.title is None and source.get("title"):
        target.title = source["title"]
        updated = True
    if target.price is None and source.get("price") is not None:
        target.price = source["price"]
        updated = True
    if target.currency is None and source.get("currency"):
        target.currency = source["currency"]
        updated = True

    src_stock = source.get("stock_status", STOCK_UNKNOWN)
    if src_stock != STOCK_UNKNOWN and (
        target.stock_status == STOCK_UNKNOWN or (override_stock and target.stock_status != src_stock)
    ):
        target.stock_status = src_stock
        updated = True

    return updated


def _merge_result(target: ProductResult, source: ProductResult) -> bool:
    """Merge non-empty fields from another ProductResult into `target`."""
    return _merge_fields(
        target,
        {
            "title": source.title,
            "price": source.price,
            "currency": source.currency,
            "stock_status": source.stock_status,
        },
    )


def extract_product(url: str, timeout_seconds: float = MAX_TOTAL_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Extract standardized product details from an e-commerce URL using a 3-tier waterfall.

    Enforces a strict per-URL timeout (default 15 seconds) across all tiers.
    Tier 1: Fast TLS-spoofed HTTP request (curl_cffi) + JSON-LD & OpenGraph/meta parser.
    Tier 2: Domain-specific HTML/CSS fallback parsers on the fetched HTML.
    Tier 3: Playwright headless browser fallback for JS-rendered pages or missing price/stock.
    """
    deadline = time.monotonic() + timeout_seconds
    result = ProductResult(url=url)
    has_domain_parser = get_domain_parser(url) is not None
    pw_stock_checker = get_domain_playwright_stock_checker(url)

    # --- Tier 1: curl_cffi + JSON-LD / OpenGraph ---
    tier1_timeout = min(8.0, max(1.0, deadline - time.monotonic()))
    html, tier1_error = fetch_html_tier1(url, timeout=tier1_timeout)
    tier1_had_data = False
    tier2_improved = False
    requires_tier3 = False

    if html:
        soup = BeautifulSoup(html, "lxml")
        tier1_result = parse_structured_metadata(soup, url)
        tier1_had_data = _merge_result(result, tier1_result)

        if has_domain_parser:
            domain_data = run_domain_parser(url, soup)
            if domain_data:
                requires_tier3 = bool(domain_data.get("requires_tier3"))
                tier2_improved = _merge_fields(result, domain_data, override_stock=True)

        if result.is_complete() and not requires_tier3:
            result.tier = "tier2" if tier2_improved else "tier1"
            return result.to_dict()

    # --- Tier 3: Playwright headless browser fallback ---
    remaining = deadline - time.monotonic()
    if remaining <= 0.5:
        if tier2_improved:
            result.tier = "tier2"
        elif tier1_had_data:
            result.tier = "tier1"
        else:
            result.tier = None
            result.error = tier1_error or f"Timed out after {timeout_seconds:.0f}s"
        return result.to_dict()

    rendered_html, live_stock, tier3_error = fetch_html_tier3(
        url,
        timeout_ms=int(remaining * 1000),
        stock_checker=pw_stock_checker,
    )
    if rendered_html:
        rendered_soup = BeautifulSoup(rendered_html, "lxml")
        t3_structured = parse_structured_metadata(rendered_soup, url)
        t3_improved = _merge_result(result, t3_structured)

        if has_domain_parser:
            domain_data = run_domain_parser(url, rendered_soup)
            if domain_data:
                t3_improved = _merge_fields(result, domain_data, override_stock=True) or t3_improved

        if live_stock != STOCK_UNKNOWN:
            t3_improved = _merge_fields(result, {"stock_status": live_stock}, override_stock=True) or t3_improved

        if t3_improved or result.is_complete():
            result.tier = "tier3"
            return result.to_dict()

    # Partial or failed fallback reporting
    if tier2_improved:
        result.tier = "tier2"
    elif tier1_had_data:
        result.tier = "tier1"
    else:
        result.tier = None
        result.error = tier3_error or tier1_error or "Failed to extract product details"

    return result.to_dict()
