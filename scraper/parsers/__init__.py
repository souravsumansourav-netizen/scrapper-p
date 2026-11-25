from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from scraper.parsers.bluestone import parse_bluestone
from scraper.parsers.tanishq import detect_tanishq_stock_playwright, parse_tanishq

ParserFunc = Callable[[BeautifulSoup, str], Dict[str, Any]]
PlaywrightStockFunc = Callable[[Any], str]

# Map domain suffix -> parser function.
# Add new domain-specific fallback parsers here.
DOMAIN_PARSERS: Dict[str, ParserFunc] = {
    "tanishq.co.in": parse_tanishq,
    "bluestone.com": parse_bluestone,
}

# Map domain suffix -> live Playwright stock inspector for JS-controlled visibility.
DOMAIN_PLAYWRIGHT_STOCK: Dict[str, PlaywrightStockFunc] = {
    "tanishq.co.in": detect_tanishq_stock_playwright,
}


def get_domain_parser(url: str) -> Optional[ParserFunc]:
    """Return a matching domain-specific parser for the given URL, if registered."""
    hostname = (urlparse(url).hostname or "").lower()
    for domain, parser in DOMAIN_PARSERS.items():
        if hostname == domain or hostname.endswith(f".{domain}"):
            return parser
    return None


def get_domain_playwright_stock_checker(url: str) -> Optional[PlaywrightStockFunc]:
    """Return a live Playwright stock checker for the given URL, if registered."""
    hostname = (urlparse(url).hostname or "").lower()
    for domain, checker in DOMAIN_PLAYWRIGHT_STOCK.items():
        if hostname == domain or hostname.endswith(f".{domain}"):
            return checker
    return None


def run_domain_parser(url: str, soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """Run the domain-specific parser for `url` if one exists."""
    parser = get_domain_parser(url)
    if parser is None:
        return None
    return parser(soup, url)
