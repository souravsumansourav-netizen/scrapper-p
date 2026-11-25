import re
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

TANISHQ_BASE_URL = "https://www.tanishq.co.in"
SFCC_DIRECT_ITEM_URL = "https://www.tanishq.co.in/product/item-{sku_lower}.html"
SFCC_PRODUCT_SHOW_URL = (
    "https://www.tanishq.co.in/on/demandware.store/Sites-Tanishq-Site/en_IN/Product-Show"
)
SFCC_SUGGESTIONS_URL = (
    "https://www.tanishq.co.in/on/demandware.store/Sites-Tanishq-Site/en_IN/SearchServices-GetSuggestions"
)
SFCC_SEARCH_URL = "https://www.tanishq.co.in/search"


class ResolverBlockedError(Exception):
    """Raised when SKU resolution fails due to WAF/rate-limit (403/429), 5xx, or timeout."""


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


def _extract_canonical_href(text: str) -> Optional[str]:
    soup = BeautifulSoup(text, "lxml")
    tag = soup.select_one("link[rel='canonical']")
    if tag and tag.get("href"):
        return urljoin(TANISHQ_BASE_URL, str(tag["href"]).strip())
    return None


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


def _get_or_raise(url: str, params: Optional[dict], headers: dict, timeout: float) -> Any:
    """Perform GET request and raise ResolverBlockedError on WAF, rate-limit, 5xx, or network error."""
    try:
        resp = curl_requests.get(
            url,
            params=params,
            impersonate="chrome",
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
    except Exception as exc:
        raise ResolverBlockedError(f"Request failed ({url}): {exc}") from exc

    if resp.status_code in (403, 429) or resp.status_code >= 500:
        raise ResolverBlockedError(f"HTTP {resp.status_code} from {url}")
    return resp


def _check_pdp_response(resp: Any, sku_lower: str) -> Optional[str]:
    """Validate a PDP response checking final URL path or <link rel='canonical'>."""
    if resp.status_code >= 400:
        return None
    if _is_valid_pdp_url(resp.url, sku_lower):
        return _ensure_en_in_lang(resp.url)
    canonical = _extract_canonical_href(resp.text)
    if _is_valid_pdp_url(canonical, sku_lower):
        return _ensure_en_in_lang(canonical)  # type: ignore[arg-type]
    return None


def resolve_tanishq_url(sku_code: str, timeout: float = 10.0) -> Optional[str]:
    """Resolve a Tanishq SKU code to its canonical PDP URL.

    Returns None only on explicit 404 / valid 200 no-results.
    Raises ResolverBlockedError on 403/429/5xx/timeouts.
    """
    sku = (sku_code or "").strip()
    if not sku:
        return None

    sku_lower = sku.lower()
    sku_upper = sku.upper()
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en-US;q=0.9,en;q=0.8",
    }

    # 1. Direct SFCC URL pattern: /product/item-<sku_lower>.html?lang=en_IN
    direct_url = SFCC_DIRECT_ITEM_URL.format(sku_lower=sku_lower)
    resp = _get_or_raise(direct_url, {"lang": "en_IN"}, headers, timeout)
    resolved = _check_pdp_response(resp, sku_lower)
    if resolved:
        return resolved

    # 2. SFCC Product-Show endpoint (lowercase, then uppercase)
    for pid in list(dict.fromkeys([sku_lower, sku_upper])):
        resp = _get_or_raise(SFCC_PRODUCT_SHOW_URL, {"pid": pid}, headers, timeout)
        resolved = _check_pdp_response(resp, sku_lower)
        if resolved:
            return resolved

    # 3. SFCC SearchServices-GetSuggestions endpoint
    resp = _get_or_raise(SFCC_SUGGESTIONS_URL, {"q": sku}, headers, timeout)
    if resp.status_code < 400:
        resolved = _extract_pdp_url_from_text(resp.text, sku_lower)
        if resolved:
            return resolved

    # 4. Fallback to Tanishq site search
    resp = _get_or_raise(SFCC_SEARCH_URL, {"q": sku, "lang": "en_IN"}, headers, timeout)
    if resp.status_code < 400:
        resolved = _check_pdp_response(resp, sku_lower)
        if resolved:
            return resolved
        resolved = _extract_pdp_url_from_text(resp.text, sku_lower)
        if resolved:
            return resolved

    return None
