"""Product attributes read from the product name.

The master enrichment report covers mostly published diamond jewellery, so most
best-selling gold SKUs have no metal / karatage / stone data. Their names usually
say it ("Minimalist 22 Karat Yellow Gold Finger Ring", "Dainty Floral Ruby Nose Pin"),
so we fill the gaps from the name. Master values always win when present.

Values use the master's vocabulary so both sources mean the same thing to the model.
"""
import re
from typing import Any, Dict

_GEMSTONES = r"ruby|rubies|emerald|sapphire|garnet|amethyst|onyx|topaz|tanzanite|citrine|peridot|coral|turquoise|opal|spinel|kundan|polki|gemstone"

_RE = {
    "diamond": re.compile(r"\bdiamonds?\b|\bsolitaire\b", re.I),
    "gemstone": re.compile(rf"\b(?:{_GEMSTONES})\b", re.I),
    "pearl": re.compile(r"\bpearls?\b", re.I),
    "gold": re.compile(r"\bgold(?:en)?\b", re.I),
    "silver": re.compile(r"\bsilver\b", re.I),
    "platinum": re.compile(r"\bplatinum\b", re.I),
    "karat": re.compile(r"\b(24|22|18|14|9)\s*(?:k|kt|karat|carat|ct)\b", re.I),
    "yellow": re.compile(r"\byellow\b", re.I),
    "rose": re.compile(r"\brose\b(?!\s*(?:flower|bud|petal))", re.I),
    "white": re.compile(r"\bwhite\b", re.I),
    "men": re.compile(r"\b(?:for\s+)?men'?s?\b|\bgents?\b", re.I),
    "kids": re.compile(r"\bkids?\b|\bchild(?:ren)?\b|\bbaby\b|\binfant\b", re.I),
    "women": re.compile(r"\bwomen'?s?\b|\bladies\b", re.I),
}

_COLOR_CODES = {"YEL": "Yellow", "ROS": "Rose", "WHI": "White"}
_UNKNOWN = {"", "unknown", "nan", "none", "null"}


def normalize_metal_color(val: Any) -> str:
    """Map master codes like 'YEL-WHI' to 'Yellow and White' so both spellings match."""
    s = str(val or "").strip()
    if s.lower() in _UNKNOWN:
        return ""
    parts = re.split(r"\s*(?:-|&|\band\b)\s*", s, flags=re.I)
    names = [_COLOR_CODES.get(p.upper(), p.strip().capitalize()) for p in parts if p.strip()]
    return " and ".join(names)


def attributes_from_name(name: Any) -> Dict[str, str]:
    """Return metal, gold_karatage, metal_color, jewellery_type, gender found in a product name.

    Only what the name actually says is returned; nothing is guessed.
    """
    text = str(name or "")
    out: Dict[str, str] = {}
    if not text.strip():
        return out

    has_diamond = bool(_RE["diamond"].search(text))
    has_gem = bool(_RE["gemstone"].search(text))
    has_pearl = bool(_RE["pearl"].search(text))
    karat = _RE["karat"].search(text)

    if _RE["platinum"].search(text):
        out["metal"] = "Platinum"
    elif _RE["silver"].search(text):
        out["metal"] = "Silver"
    elif _RE["gold"].search(text) or karat:
        out["metal"] = "Gold"

    if karat:
        out["gold_karatage"] = karat.group(1)

    colors = [c.capitalize() for c in ("yellow", "white", "rose") if _RE[c].search(text)]
    if colors and out.get("metal") == "Gold":
        out["metal_color"] = " and ".join(colors)

    if has_diamond:
        out["jewellery_type"] = "Diamond Jewellery"
    elif has_gem or has_pearl:
        out["jewellery_type"] = "Jewellery with Gemstones"
    elif out.get("metal") == "Gold":
        out["jewellery_type"] = "Gold Jewellery"
    elif out.get("metal") == "Silver":
        out["jewellery_type"] = "Silver Jewellery"

    if _RE["kids"].search(text):
        out["gender"] = "Kids"
    elif _RE["men"].search(text) and not _RE["women"].search(text):
        out["gender"] = "Men"
    elif _RE["women"].search(text):
        out["gender"] = "Women"

    out["stone"] = "diamond" if has_diamond else ("pearl" if has_pearl and not has_gem else ("gemstone" if has_gem else "none"))
    return out
