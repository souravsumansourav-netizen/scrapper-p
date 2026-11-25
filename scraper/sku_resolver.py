import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

TANISHQ_BASE_URL = "https://www.tanishq.co.in"
SFCC_PRODUCT_SHOW_URL = (
    "https://www.tanishq.co.in/on/demandware.store/Sites-Tanishq-Site/en_IN/Product-Show"
)
SFCC_SUGGESTIONS_URL = (
    "https://www.tanishq.co.in/on/demandware.store/Sites-Tanishq-Site/en_IN/SearchServices-GetSuggestions"
)
SFCC_SEARCH_URL = "https://www.tanishq.co.in/search"


def _ensure_en_in_lang(url: str) -> str:
    """Ensure the resolved Tanishq PDP URL includes `?lang=en_IN`."""
    parsed = urlparse(url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_params["lang"] = "en_IN"
    return urlunparse(parsed._replace(query=urlencode(query_params)))


def _is_valid_pdp_url(url: Optional[str], sku_lower: str) -> bool:
    """Strictly validate that URL is a Tanishq PDP whose path contains `sku_lower`."""
    if not url:
        return False
    parsed = urlparse(url)
    path_lower = parsed.path.lower()
    return (
        "tanishq.co.in" in (parsed.hostname or "").lower()
        and "/product/" in path_lower
        and path_lower.endswith(".html")
        and sku_lower in path_lower
    )


def _extract_pdp_url_from_text(text: str, sku_lower: str) -> Optional[str]:
    """Extract a validated PDP URL containing `sku_lower` from HTML or JSON text."""
    soup = BeautifulSoup(text, "lxml")

    # 1. Check canonical / og:url first
    for selector, attr in [
        ("link[rel='canonical']", "href"),
        ("meta[property='og:url']", "content"),
    ]:
        tag = soup.select_one(selector)
        if tag and tag.get(attr):
            candidate = urljoin(TANISHQ_BASE_URL, str(tag[attr]).strip())
            if _is_valid_pdp_url(candidate, sku_lower):
                return _ensure_en_in_lang(candidate)

    # 2. Check anchor hrefs matching /product/...<sku_lower>...html
    for a in soup.select("a[href*='/product/']"):
        href = a.get("href")
        if not href:
            continue
        candidate = urljoin(TANISHQ_BASE_URL, str(href).strip())
        if _is_valid_pdp_url(candidate, sku_lower):
            return _ensure_en_in_lang(candidate)

    # 3. Regex fallback for JSON/script-embedded relative or absolute product URLs
    abs_pattern = re.compile(
        rf"https?://(?:www\.)?tanishq\.co\.in/product/[^\"'\s<>\\]*{re.escape(sku_lower)}[^\"'\s<>\\]*\.html",
        re.IGNORECASE,
    )
    abs_match = abs_pattern.search(text)
    if abs_match:
        candidate = abs_match.group(0)
        if _is_valid_pdp_url(candidate, sku_lower):
            return _ensure_en_in_lang(candidate)

    rel_pattern = re.compile(
        rf"/product/[^\"'\s<>\\]*{re.escape(sku_lower)}[^\"'\s<>\\]*\.html",
        re.IGNORECASE,
    )
    rel_match = rel_pattern.search(text)
    if rel_match:
        candidate = urljoin(TANISHQ_BASE_URL, rel_match.group(0))
        if _is_valid_pdp_url(candidate, sku_lower):
            return _ensure_en_in_lang(candidate)

    return None


def resolve_tanishq_url(sku_code: str, timeout: float = 10.0) -> Optional[str]:
    """Resolve a Tanishq SKU code (e.g. '500357PCAAAC02') to its canonical PDP URL."""
    sku = (sku_code or "").strip()
    if not sku:
        return None

    sku_lower = sku.lower()
    sku_upper = sku.upper()
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    }

    # a) Try SFCC Product-Show endpoint with lowercase and uppercase pid
    pids_to_try = list(dict.fromkeys([sku_lower, sku_upper]))
    for pid in pids_to_try:
        try:
            resp = curl_requests.get(
                SFCC_PRODUCT_SHOW_URL,
                params={"pid": pid},
                impersonate="chrome",
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
            )
            if resp.status_code < 400:
                if _is_valid_pdp_url(resp.url, sku_lower):
                    return _ensure_en_in_lang(resp.url)
                resolved = _extract_pdp_url_from_text(resp.text, sku_lower)
                if resolved:
                    return resolved
        except Exception:
            pass

    # b) Try SFCC SearchServices-GetSuggestions endpoint
    try:
        resp = curl_requests.get(
            SFCC_SUGGESTIONS_URL,
            params={"q": sku},
            impersonate="chrome",
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.status_code < 400:
            resolved = _extract_pdp_url_from_text(resp.text, sku_lower)
            if resolved:
                return resolved
    except Exception:
        pass

    # c) Fallback to Tanishq site search
    try:
        resp = curl_requests.get(
            SFCC_SEARCH_URL,
            params={"q": sku, "lang": "en_IN"},
            impersonate="chrome",
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.status_code < 400:
            if _is_valid_pdp_url(resp.url, sku_lower):
                return _ensure_en_in_lang(resp.url)
            resolved = _extract_pdp_url_from_text(resp.text, sku_lower)
            if resolved:
                return resolved
    except Exception:
        pass

    return None
