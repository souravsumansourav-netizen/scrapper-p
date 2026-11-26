import re
from typing import Iterable, Optional, Tuple

import pandas as pd

NON_JEWELLERY_SKU_PREFIXES = ("600", "800", "EGC", "AMOUNTGC")
NON_JEWELLERY_TEXT_RE = re.compile(
    r"\b(?:coin|coins|bar|bars|bullion|gift\s*card|e-gift|egc|voucher)\b|^amount:",
    re.IGNORECASE,
)

DEFAULT_SKU_COLS = (
    "sku_code",
    "Product ID",
    "nproductId",
    "Item ID",
    "BASE SKU",
)
DEFAULT_TEXT_COLS = (
    "Category",
    "category",
    "Product Name",
    "product_name",
    "TITLE",
    "title",
    "Item name",
    "item_name",
)


def _is_non_empty_coin_weight(val: object) -> bool:
    if val is None or pd.isna(val):
        return False
    text = str(val).strip().lower()
    return text not in {"", "nan", "none", "null", "na", "n/a"}


def filter_jewellery(
    df: pd.DataFrame,
    sku_cols: Iterable[str] = DEFAULT_SKU_COLS,
    text_cols: Iterable[str] = DEFAULT_TEXT_COLS,
    coin_weight_col: Optional[str] = "COIN WEIGHT",
) -> Tuple[pd.DataFrame, int]:
    """Exclude coins, bars, bullion, gift cards, e-gift, and vouchers.

    Returns (filtered_df, excluded_count).
    """
    if df.empty:
        return df.copy(), 0

    exclude_mask = pd.Series(False, index=df.index)

    for col in sku_cols:
        if col in df.columns:
            s = df[col].fillna("").astype(str).str.strip().str.upper()
            for prefix in NON_JEWELLERY_SKU_PREFIXES:
                exclude_mask = exclude_mask | s.str.startswith(prefix)

    for col in text_cols:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
            exclude_mask = exclude_mask | s.str.contains(NON_JEWELLERY_TEXT_RE, regex=True)

    if coin_weight_col and coin_weight_col in df.columns:
        exclude_mask = exclude_mask | df[coin_weight_col].map(_is_non_empty_coin_weight)

    excluded_count = int(exclude_mask.sum())
    filtered_df = df.loc[~exclude_mask].copy().reset_index(drop=True)
    return filtered_df, excluded_count

