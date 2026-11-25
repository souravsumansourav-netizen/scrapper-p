import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

TANISHQ_BASE_URL = "https://www.tanishq.co.in"
SFCC_PRODUCT_SHOW_URL = (
    "https://www.tanishq.co.in/on/demandware.store/Sites-Tanishq-Site/en_IN/Product-Show"
)
SFCC_SEARCH_URL = "https://www.tanishq.co.in/search"


def _ensure_en_in_lang(url: str) -> str:
    """Ensure the resolved Tanishq PDP URL includes `?lang=en_IN`."""
    parsed = urlparse(url)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_params["lang"] = "en_IN"
    return urlunparse(parsed._replace(query=urlencode(query_params)))


def _is_valid_pdp_url(url: Optional[str]) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return (
        "tanishq.co.in" in (parsed.hostname or "").lower()
        and "/product/" in parsed.path
        and parsed.path.endswith(".html")
    )


def _extract_pdp_url_from_html(html: str, sku_lower: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")

    # 1. Check canonical / og:url first
    for selector, attr in [
        ("link[rel='canonical']", "href"),
        ("meta[property='og:url']", "content"),
    ]:
        tag = soup.select_one(selector)
        if tag and tag.get(attr):
            candidate = urljoin(TANISHQ_BASE_URL, str(tag[attr]).strip())
            if _is_valid_pdp_url(candidate):
                return _ensure_en_in_lang(candidate)

    # 2. Check product links matching SKU code, then any /product/*.html link
    fallback_candidate: Optional[str] = None
    for a in soup.select("a[href*='/product/']"):
        href = a.get("href")
        if not href:
            continue
        candidate = urljoin(TANISHQ_BASE_URL, str(href).strip())
        if not _is_valid_pdp_url(candidate):
            continue
        if sku_lower in candidate.lower():
            return _ensure_en_in_lang(candidate)
        if fallback_candidate is None:
            fallback_candidate = candidate

    if fallback_candidate:
        return _ensure_en_in_lang(fallback_candidate)
    return None


def resolve_tanishq_url(sku_code: str, timeout: float = 10.0) -> Optional[str]:
    """Resolve a Tanishq SKU code (e.g. '501002SFSAAA2A') to its canonical PDP URL."""
    sku = (sku_code or "").strip()
    if not sku:
        return None

    sku_lower = sku.lower()
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    }

    # 1. Try SFCC Product-Show endpoint
    try:
        resp = curl_requests.get(
            SFCC_PRODUCT_SHOW_URL,
            params={"pid": sku},
            impersonate="chrome",
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.status_code < 400:
            if _is_valid_pdp_url(resp.url):
                return _ensure_en_in_lang(resp.url)
            resolved = _extract_pdp_url_from_html(resp.text, sku_lower)
            if resolved:
                return resolved
    except Exception:
        pass

    # 2. Fallback to Tanishq site search
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
            if _is_valid_pdp_url(resp.url) and sku_lower in resp.url.lower():
                return _ensure_en_in_lang(resp.url)
            resolved = _extract_pdp_url_from_html(resp.text, sku_lower)
            if resolved:
                return resolved
            # Regex fallback for JSON/script-embedded product URLs
            pattern = re.compile(
                rf"https?://(?:www\.)?tanishq\.co\.in/product/[^\"'\s<>]*{re.escape(sku_lower)}[^\"'\s<>]*\.html",
                re.IGNORECASE,
            )
            match = pattern.search(resp.text)
            if match:
                return _ensure_en_in_lang(match.group(0))
    except Exception:
        pass

    return None

