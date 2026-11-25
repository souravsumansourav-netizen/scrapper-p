from typing import Optional
from bs4 import BeautifulSoup


def first_text(soup: BeautifulSoup, selectors: list) -> Optional[str]:
    """Return stripped text of the first matching CSS selector, or None."""
    for selector in selectors:
        el = soup.select_one(selector)
        if el:
            text = el.get_text(" ", strip=True)
            if text:
                return text
    return None

