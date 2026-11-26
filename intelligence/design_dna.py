import re
from typing import Dict, List, Mapping, Optional, Tuple

DESIGN_AXES: Dict[str, Tuple[str, ...]] = {
    "motif": (
        "floral",
        "leaf",
        "geometric",
        "lotus",
        "heart",
        "om",
        "deity",
        "peacock",
        "infinity",
        "solitaire",
        "halo",
        "teardrop",
        "butterfly",
        "star",
        "evil eye",
    ),
    "aesthetic": (
        "dainty",
        "minimal",
        "contemporary",
        "modern",
        "traditional",
        "antique",
        "filigree",
        "textured",
        "sleek",
        "delicate",
        "statement",
        "bold",
        "classic",
        "vintage",
    ),
    "wearability": (
        "daily",
        "office",
        "workwear",
        "party",
        "festive",
        "bridal",
        "wedding",
        "gifting",
        "kids",
        "casual",
        "ethnic",
    ),
    "form_factor": (
        "stud",
        "drop",
        "hoop",
        "jhumka",
        "layered",
        "twisted",
        "mesh",
        "open ring",
        "cuff",
        "choker",
        "mangalsutra",
        "tennis",
        "pendant set",
        "bangle",
    ),
    "accent": (
        "enamel",
        "pearl",
        "ruby",
        "emerald",
        "rhodium",
        "rose gold",
        "two-tone",
        "kundan",
        "meenakari",
        "polki",
    ),
}

_PLURALIZABLE_TERMS = {
    "leaf",
    "heart",
    "peacock",
    "solitaire",
    "halo",
    "teardrop",
    "butterfly",
    "star",
    "stud",
    "drop",
    "hoop",
    "jhumka",
    "cuff",
    "choker",
    "mangalsutra",
    "pendant set",
    "bangle",
    "pearl",
    "ruby",
    "emerald",
}


def _compile_term_pattern(term: str) -> re.Pattern:
    parts = [re.escape(p) for p in re.split(r"[\s-]+", term.strip())]
    body = r"[\s-]+".join(parts)
    if term in _PLURALIZABLE_TERMS:
        if term == "leaf":
            body = r"(?:leaf|leaves)"
        elif term == "butterfly":
            body = r"(?:butterfly|butterflies)"
        elif term == "ruby":
            body = r"(?:ruby|rubies)"
        else:
            body = rf"{body}s?"
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


_COMPILED_AXES: Dict[str, List[Tuple[str, re.Pattern]]] = {
    axis: [(term, _compile_term_pattern(term)) for term in terms]
    for axis, terms in DESIGN_AXES.items()
}

DNA_TEXT_FIELDS = (
    "Product Name",
    "product_name",
    "TITLE",
    "title",
    "SHORT DESCRIPTION",
    "short_description",
    "LONG DESCRIPTION",
    "long_description",
    "DESIGN",
    "design",
    "STYLING",
    "styling",
)


def normalize_dna_text(*parts: Optional[str]) -> str:
    """Combine and normalize text fields into collapsed lowercase text."""
    cleaned: List[str] = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            cleaned.append(s)
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip().lower()


def extract_design_dna(text_or_row: object) -> Dict[str, List[str]]:
    """Extract multi-label design tags across all 5 axes from a text string or row mapping."""
    if isinstance(text_or_row, str):
        text = normalize_dna_text(text_or_row)
    elif isinstance(text_or_row, Mapping):
        parts = [text_or_row.get(col) for col in DNA_TEXT_FIELDS]
        text = normalize_dna_text(*parts)
    else:
        text = normalize_dna_text(str(text_or_row))

    result: Dict[str, List[str]] = {}
    for axis, patterns in _COMPILED_AXES.items():
        if not text:
            result[axis] = []
            continue
        result[axis] = [term for term, pat in patterns if pat.search(text)]
    return result

