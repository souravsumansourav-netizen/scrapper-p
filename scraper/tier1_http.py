import json
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from scraper.models import ProductResult, STOCK_UNKNOWN
from scraper.normalizer import (
    normalize_currency,
    normalize_price,
    normalize_stock_status,
)


DEFAULT_TIMEOUT_SECONDS = 8.0


def fetch_html_tier1(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Tuple[Optional[str], Optional[str]]:
    """Fetch HTML using curl_cffi with Chrome TLS impersonation.

    Returns (html, error_message).
    """
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    }
    try:
        resp = curl_requests.get(
            url,
            impersonate="chrome",
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.status_code >= 400:
            return None, f"HTTP {resp.status_code}"
        return resp.text, None
    except Exception as exc:
        return None, str(exc)


def _iter_json_ld_objects(data: Any) -> List[Dict[str, Any]]:
    """Flatten JSON-LD structures (dicts, lists, @graph) into a list of dicts."""
    items: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for elem in data:
            items.extend(_iter_json_ld_objects(elem))
    elif isinstance(data, dict):
        if "@graph" in data:
            items.extend(_iter_json_ld_objects(data["@graph"]))
        items.append(data)
    return items


def _is_product_type(obj: Dict[str, Any]) -> bool:
    obj_type = obj.get("@type")
    if isinstance(obj_type, str):
        return obj_type.lower() == "product" or obj_type.lower().endswith("/product")
    if isinstance(obj_type, list):
        return any(
            isinstance(t, str) and (t.lower() == "product" or t.lower().endswith("/product"))
            for t in obj_type
        )
    return False


def _extract_from_offers(offers: Any) -> Tuple[Optional[float], Optional[str], str]:
    """Extract (price, currency, stock_status) from schema.org offers."""
    if isinstance(offers, list):
        for offer in offers:
            price, currency, stock = _extract_from_offers(offer)
            if price is not None or stock != STOCK_UNKNOWN:
                return price, currency, stock
        return None, None, STOCK_UNKNOWN

    if not isinstance(offers, dict):
        return None, None, STOCK_UNKNOWN

    raw_price = offers.get("price")
    if raw_price is None:
        raw_price = offers.get("lowPrice")
    if raw_price is None and isinstance(offers.get("priceSpecification"), dict):
        raw_price = offers["priceSpecification"].get("price")

    price = normalize_price(raw_price)
    raw_currency = offers.get("priceCurrency")
    if raw_currency is None and isinstance(offers.get("priceSpecification"), dict):
        raw_currency = offers["priceSpecification"].get("priceCurrency")
    currency = normalize_currency(raw_currency, str(raw_price) if raw_price is not None else None)

    stock = normalize_stock_status(offers.get("availability"))

    # Nested offers inside AggregateOffer
    if (price is None or stock == STOCK_UNKNOWN) and "offers" in offers:
        nested_price, nested_currency, nested_stock = _extract_from_offers(offers["offers"])
        if price is None:
            price = nested_price
        if currency is None:
            currency = nested_currency
        if stock == STOCK_UNKNOWN:
            stock = nested_stock

    return price, currency, stock


def _parse_json_ld(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract product fields from schema.org/Product JSON-LD blocks."""
    result: Dict[str, Any] = {
        "title": None,
        "price": None,
        "currency": None,
        "stock_status": STOCK_UNKNOWN,
    }

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw_text = script.string or script.get_text(strip=True)
        if not raw_text:
            continue
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            continue

        for obj in _iter_json_ld_objects(parsed):
            if not _is_product_type(obj):
                continue

            if result["title"] is None and isinstance(obj.get("name"), str):
                result["title"] = obj["name"].strip() or None

            if "offers" in obj:
                price, currency, stock = _extract_from_offers(obj["offers"])
                if result["price"] is None and price is not None:
                    result["price"] = price
                if result["currency"] is None and currency is not None:
                    result["currency"] = currency
                if result["stock_status"] == STOCK_UNKNOWN and stock != STOCK_UNKNOWN:
                    result["stock_status"] = stock

    return result


def _get_meta_content(soup: BeautifulSoup, *selectors: Tuple[str, str]) -> Optional[str]:
    """Return the first non-empty meta content attribute matching (attr_name, attr_value)."""
    for attr_name, attr_value in selectors:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if tag and tag.get("content"):
            val = str(tag["content"]).strip()
            if val:
                return val
    return None


def _parse_opengraph_and_meta(soup: BeautifulSoup) -> Dict[str, Any]:
    """Extract product fields from OpenGraph, product meta tags, and microdata."""
    title = _get_meta_content(
        soup,
        ("property", "og:title"),
        ("name", "twitter:title"),
        ("name", "title"),
    )
    if title is None and soup.title and soup.title.string:
        title = soup.title.string.strip() or None

    raw_price = _get_meta_content(
        soup,
        ("property", "product:price:amount"),
        ("property", "og:price:amount"),
        ("itemprop", "price"),
        ("name", "twitter:data1"),
    )
    if raw_price is None:
        price_el = soup.find(attrs={"itemprop": "price"})
        if price_el:
            raw_price = price_el.get("content") or price_el.get_text(strip=True)

    raw_currency = _get_meta_content(
        soup,
        ("property", "product:price:currency"),
        ("property", "og:price:currency"),
        ("itemprop", "priceCurrency"),
    )
    if raw_currency is None:
        curr_el = soup.find(attrs={"itemprop": "priceCurrency"})
        if curr_el:
            raw_currency = curr_el.get("content") or curr_el.get_text(strip=True)

    raw_avail = _get_meta_content(
        soup,
        ("property", "product:availability"),
        ("property", "og:availability"),
        ("itemprop", "availability"),
    )
    if raw_avail is None:
        avail_el = soup.find(attrs={"itemprop": "availability"})
        if avail_el:
            raw_avail = avail_el.get("href") or avail_el.get("content") or avail_el.get_text(strip=True)

    return {
        "title": title,
        "price": normalize_price(raw_price),
        "currency": normalize_currency(raw_currency, raw_price),
        "stock_status": normalize_stock_status(raw_avail),
    }


def parse_structured_metadata(soup: BeautifulSoup, url: str) -> ProductResult:
    """Parse JSON-LD and OpenGraph/meta tags from a BeautifulSoup document."""
    json_ld = _parse_json_ld(soup)
    meta = _parse_opengraph_and_meta(soup)

    title = json_ld["title"] or meta["title"]
    price = json_ld["price"] if json_ld["price"] is not None else meta["price"]
    currency = json_ld["currency"] or meta["currency"]
    stock_status = (
        json_ld["stock_status"]
        if json_ld["stock_status"] != STOCK_UNKNOWN
        else meta["stock_status"]
    )

    return ProductResult(
        url=url,
        title=title,
        price=price,
        currency=currency,
        stock_status=stock_status,
    )

