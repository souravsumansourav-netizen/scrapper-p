"""Leak-free feature frame for the category-winner model (Phase 1: latest period).

Rules that keep the model honest:
- No feature is derived from purchases, revenue or units (those define the label).
- Price comes from the listed price band (web) or the scraped live price, never
  from purchase revenue, so having a price does not reveal a sale.
- The SKU universe is the published catalog plus every SKU seen in the period,
  so unsold, unviewed products are scored too.
- Current stock status is NOT a feature: it is scraped after the period, and an
  item can be out of stock *because* it sold. It is shown on the dashboard only.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intelligence.categories import canonical_category
from intelligence.design_dna import DESIGN_AXES
from intelligence.name_attributes import attributes_from_name, normalize_metal_color

CAT_FEATURES = [
    "category",
    "jewellery_type",
    "stone",
    "metal",
    "metal_color",
    "gold_karatage",
    "gender",
    "is_new_collection",
]
# Not used: collection, occasion and weight_band exist only in the master report, which
# covers ~9% of winners, so they mostly encode "is in the master file" rather than taste.

NUM_FEATURES = [
    "log_views_total",
    "log_views_web",
    "log_views_app",
    "app_view_share",
    "log_clicks_total",
    "click_through",
    "log_atc_total",
    "atc_rate_smoothed",
    "atc_rate_pct_cat",
    "views_pct_cat",
    "price_band_ord",
    "price_pct_cat",
]

PRICE_BAND_ORDER = {"under_25k": 0, "25k_50k": 1, "50k_100k": 2, "over_100k": 3}
PRICE_BAND_MID = {"under_25k": 15_000.0, "25k_50k": 37_500.0, "50k_100k": 75_000.0, "over_100k": 150_000.0}
PRICE_BAND_LABEL = {0: "under ₹25k", 1: "₹25k–50k", 2: "₹50k–1L", 3: "₹1L+"}

ATC_PRIOR_VIEWS = 50.0

# Master columns arrive with different names from the CSV preview and from Supabase.
_MASTER_RENAMES = {
    "master_title": "title",
    "product_type": "jewellery_type",
    "karatage": "gold_karatage",
}

PERIOD_NUM_COLS = (
    "web_views",
    "web_clicks",
    "web_atc",
    "app_views",
    "app_clicks",
    "app_atc",
    "units_sold",
    "oms_units",
    "web_purchases",
    "web_purchased",
    "app_purchases",
)


def _parse_tags(val: Any) -> Dict[str, List[str]]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "y"}


def _price_band_from_price(price: float) -> Optional[str]:
    if price is None or pd.isna(price) or price <= 0:
        return None
    if price < 25_000:
        return "under_25k"
    if price < 50_000:
        return "25k_50k"
    if price < 100_000:
        return "50k_100k"
    return "over_100k"


NAME_FILLED_COLS = ("metal", "gold_karatage", "metal_color", "jewellery_type", "gender")


def fill_attributes_from_names(df: pd.DataFrame) -> None:
    """Fill metal / karatage / colour / type / gender from the product name where the master
    has nothing, and add a `stone` attribute (diamond / gemstone / pearl / none).

    Most best sellers are gold pieces missing from the master report; without this the
    model sees "unknown" for them and learns "unknown = winner" instead of "gold = winner".
    """
    parsed = [attributes_from_name(n) for n in df["product_name"]]
    for col in NAME_FILLED_COLS:
        master_vals = df[col] if col in df.columns else pd.Series(None, index=df.index)
        if col == "metal_color":
            master_vals = master_vals.map(normalize_metal_color)
        cleaned = master_vals.astype(str).str.strip()
        missing = master_vals.isna() | cleaned.str.lower().isin({"", "unknown", "nan", "none", "null"})
        from_name = pd.Series([p.get(col) for p in parsed], index=df.index)
        df[col] = master_vals.where(~missing, from_name)
    stone = pd.Series([p.get("stone", "none") if str(n or "").strip() else None for p, n in zip(parsed, df["product_name"])], index=df.index)
    # A master "Diamond Jewellery" item is a diamond piece even if its name doesn't say so.
    stone = stone.where(~((df["jewellery_type"] == "Diamond Jewellery") & (stone == "none")), "diamond")
    df["stone"] = stone
    # Absent from the master means "not flagged", not a separate category.
    if "is_new_collection" in df.columns:
        df["is_new_collection"] = df["is_new_collection"].map(lambda v: "yes" if _to_bool(v) else "no")


def latest_period(period_df: pd.DataFrame) -> pd.DataFrame:
    """Return rows of the most recent period only."""
    if "period_start" not in period_df.columns:
        return period_df.copy()
    starts = period_df["period_start"].astype(str)
    return period_df[starts == starts.max()].copy()


def prepare_master(master_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if master_df is None or master_df.empty:
        return pd.DataFrame(columns=["sku_code"])
    m = master_df.rename(columns={k: v for k, v in _MASTER_RENAMES.items() if k in master_df.columns and v not in master_df.columns})
    m = m.copy()
    m["sku_code"] = m["sku_code"].astype(str).str.strip().str.upper()
    m = m.drop_duplicates(subset=["sku_code"], keep="first")
    m["published"] = m["published"].map(_to_bool) if "published" in m.columns else False
    return m


def build_feature_frame(
    period_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame] = None,
    market_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build one row per SKU for the latest period. Returns (frame, feature_columns)."""
    p = latest_period(period_df)
    p["sku_code"] = p["sku_code"].astype(str).str.strip().str.upper()
    for col in PERIOD_NUM_COLS:
        p[col] = pd.to_numeric(p[col], errors="coerce").fillna(0.0) if col in p.columns else 0.0
    if p["web_purchases"].sum() == 0:
        p["web_purchases"] = p["web_purchased"]
    # Always recompute: stored units_sold may predate a later App/OMS upload for the period.
    p["units_sold"] = np.maximum(p["oms_units"], p["web_purchases"] + p["app_purchases"])
    agg_spec = {c: "sum" for c in PERIOD_NUM_COLS}
    first = lambda s: next((v for v in s if pd.notna(v) and str(v).strip() not in {"", "nan", "None"}), None)  # noqa: E731
    for c in ("product_name", "category", "price_band", "design_tags"):
        if c in p.columns:
            agg_spec[c] = first
    p = p.groupby("sku_code", as_index=False).agg(agg_spec)

    m = prepare_master(master_df)
    universe = set(p["sku_code"])
    if not m.empty:
        universe |= set(m.loc[m["published"], "sku_code"])
    df = pd.DataFrame({"sku_code": sorted(universe)}).merge(p, on="sku_code", how="left")
    if not m.empty:
        df = df.merge(m, on="sku_code", how="left", suffixes=("", "_m"))

    for col in PERIOD_NUM_COLS:
        df[col] = df[col].fillna(0.0)

    title = df["title"] if "title" in df.columns else pd.Series(None, index=df.index)
    df["product_name"] = df["product_name"].where(df["product_name"].notna(), title)
    master_cat = df["category_m"] if "category_m" in df.columns else pd.Series(None, index=df.index)
    # Canonicalise again: rows stored by older ETL runs may carry raw labels ("Ring", "tq-earrings").
    df["category"] = [
        canonical_category(c, mc, name=n) for c, mc, n in zip(df["category"], master_cat, df["product_name"])
    ]

    fill_attributes_from_names(df)

    # --- Traffic & intent (no purchases) ---
    df["views_total"] = df["web_views"] + df["app_views"]
    df["clicks_total"] = df["web_clicks"] + df["app_clicks"]
    df["atc_total"] = df["web_atc"] + df["app_atc"]
    df["log_views_total"] = np.log1p(df["views_total"])
    df["log_views_web"] = np.log1p(df["web_views"])
    df["log_views_app"] = np.log1p(df["app_views"])
    df["log_clicks_total"] = np.log1p(df["clicks_total"])
    df["log_atc_total"] = np.log1p(df["atc_total"])
    df["app_view_share"] = np.where(df["views_total"] > 0, df["app_views"] / df["views_total"].clip(lower=1e-9), np.nan)
    df["click_through"] = np.where(df["views_total"] > 0, df["clicks_total"] / df["views_total"].clip(lower=1e-9), np.nan)

    # Add-to-cart rate shrunk toward the category average, so 1 cart from 2 views is not 50%.
    cat_atc = df.groupby("category")["atc_total"].transform("sum")
    cat_views = df.groupby("category")["views_total"].transform("sum").clip(lower=1.0)
    prior = cat_atc / cat_views
    df["atc_rate_smoothed"] = (df["atc_total"] + ATC_PRIOR_VIEWS * prior) / (df["views_total"] + ATC_PRIOR_VIEWS)

    viewed = df["views_total"] > 0
    df["views_pct_cat"] = df.groupby("category")["views_total"].rank(pct=True)
    df["atc_rate_pct_cat"] = np.nan
    df.loc[viewed, "atc_rate_pct_cat"] = df[viewed].groupby("category")["atc_rate_smoothed"].rank(pct=True)
    # Used for Hidden Gem rules: traffic rank among SKUs that got any traffic.
    df["views_pct_cat_viewed"] = np.nan
    df.loc[viewed, "views_pct_cat_viewed"] = df[viewed].groupby("category")["views_total"].rank(pct=True)

    # --- Price (listed band or scraped live price; never purchase revenue) ---
    df["live_price"] = np.nan
    df["stock_status"] = None
    if market_df is not None and not market_df.empty:
        mk = market_df.copy()
        mk["sku_code"] = mk["sku_code"].astype(str).str.strip().str.upper()
        mk = mk.drop_duplicates("sku_code").set_index("sku_code")
        df["live_price"] = pd.to_numeric(df["sku_code"].map(mk["price"]) if "price" in mk else np.nan, errors="coerce")
        if "stock_status" in mk:
            df["stock_status"] = df["sku_code"].map(mk["stock_status"])
    band = df["price_band"] if "price_band" in df.columns else pd.Series(None, index=df.index)
    band = band.where(band.notna(), df["live_price"].map(_price_band_from_price))
    df["price_band"] = band
    df["price_band_ord"] = band.map(PRICE_BAND_ORDER).astype(float)
    df["price_estimate"] = df["live_price"].where(df["live_price"] > 0, band.map(PRICE_BAND_MID))
    df["price_pct_cat"] = df.groupby("category")["price_estimate"].rank(pct=True)

    # --- Catalog attributes ---
    for col in CAT_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = (
            df[col].astype(str).str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown", "True": "yes", "False": "no"})
            .fillna("unknown")
        )
        df[col] = pd.Categorical(df[col])

    # --- Design DNA flags ---
    tag_src = []
    for a, b in zip(df.get("design_tags", pd.Series(None, index=df.index)), df.get("design_tags_m", pd.Series(None, index=df.index))):
        tags = _parse_tags(a)
        tag_src.append(tags if tags else _parse_tags(b))
    df["design_tags"] = tag_src
    dna_cols = {}
    for axis, terms in DESIGN_AXES.items():
        for term in terms:
            dna_cols[f"dna_{axis}_{_slug(term)}"] = [1 if term in t.get(axis, []) else 0 for t in tag_src]
    dna_df = pd.DataFrame(dna_cols, index=df.index)
    dna_df = dna_df.loc[:, dna_df.sum() >= 10]  # drop tags too rare to learn from
    df = pd.concat([df, dna_df], axis=1)

    feature_cols = NUM_FEATURES + CAT_FEATURES + list(dna_df.columns)
    return df, feature_cols
