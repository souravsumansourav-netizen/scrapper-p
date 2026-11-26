"""Canonical jewellery categories shared by ETL, features and tiering.

Sources disagree on labels ("Ring" vs "Finger Ring", "tq-earrings", plurals,
merchandising buckets like "The Festive Edit"). Winners are ranked *within*
a category, so every SKU needs one stable category.
"""
import re
from typing import Any, Optional

_ALIASES = {
    "finger ring": "Finger Ring",
    "finger rings": "Finger Ring",
    "ring": "Finger Ring",
    "rings": "Finger Ring",
    "earring": "Earrings",
    "earrings": "Earrings",
    "jhumka": "Earrings",
    "jhumkas": "Earrings",
    "pendant": "Pendant",
    "pendants": "Pendant",
    "chain": "Chain",
    "chains": "Chain",
    "bangle": "Bangle",
    "bangles": "Bangle",
    "bracelet": "Bracelet",
    "bracelets": "Bracelet",
    "mangalsutra": "Mangalsutra",
    "mangalsutras": "Mangalsutra",
    "nose pin": "Nose Pin",
    "nose pins": "Nose Pin",
    "necklace": "Necklace",
    "necklaces": "Necklace",
    "haram": "Necklace",
    "necklace set": "Necklace Set",
    "necklace sets": "Necklace Set",
    "pendant set": "Pendant Set",
    "pendant and earrings set": "Pendant Set",
    "pendant and earrings seet": "Pendant Set",
    "pendant with earrings set": "Pendant Set",
    "pendant with chain and earrings set": "Pendant Set",
    "pendant with chain": "Pendant with Chain",
    "maang tikka": "Maang Tikka",
    "charm": "Charm",
    "charms": "Charm",
    "brooch": "Brooch",
    "anklet": "Anklet",
    "anklets": "Anklet",
    "kada": "Bangle",
    "idol": "Idol",
    "idols": "Idol",
    "rakhi": "Rakhi",
    "rakhis": "Rakhi",
}

# Order matters: more specific phrases first.
_NAME_KEYWORDS = (
    ("necklace set", "Necklace Set"),
    ("pendant and earring", "Pendant Set"),
    ("pendant set", "Pendant Set"),
    ("pendant with chain", "Pendant with Chain"),
    ("mangalsutra", "Mangalsutra"),
    ("nose pin", "Nose Pin"),
    ("nosepin", "Nose Pin"),
    ("maang tikka", "Maang Tikka"),
    ("idol", "Idol"),
    ("rakhi", "Rakhi"),
    ("earring", "Earrings"),
    ("jhumka", "Earrings"),
    ("stud", "Earrings"),
    ("hoop", "Earrings"),
    ("bangle", "Bangle"),
    ("kada", "Bangle"),
    ("bracelet", "Bracelet"),
    ("pendant", "Pendant"),
    ("necklace", "Necklace"),
    ("chain", "Chain"),
    ("ring", "Finger Ring"),
    ("charm", "Charm"),
)


def _clean(val: Any) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return ""
    s = re.sub(r"^tq-", "", s.lower())
    return re.sub(r"\s+", " ", s.replace("-", " ")).strip()


def canonical_category(*labels: Any, name: Optional[str] = None) -> str:
    """Return the first recognised category among labels, else infer from the product name."""
    for label in labels:
        key = _clean(label)
        if key in _ALIASES:
            return _ALIASES[key]
    text = _clean(name)
    if text:
        padded = f" {text} "
        for kw, cat in _NAME_KEYWORDS:
            if kw == "ring":
                if re.search(r"\bring\b", text):
                    return cat
            elif kw in padded:
                return cat
    return "Other"
