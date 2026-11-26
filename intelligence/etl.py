import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from intelligence.categories import canonical_category
from intelligence.design_dna import extract_design_dna
from intelligence.filters import filter_jewellery

PRICE_BAND_ALIASES = {
    "<25k": "under_25k",
    "25k-50k": "25k_50k",
    "50k-1l": "50k_100k",
    "1l+": "over_100k",
}


def normalize_sku(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip().upper()
    return "" if s in {"NAN", "NONE", "NULL"} else s


def normalize_name(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", s).lower()


def to_base_sku_13(sku: str) -> str:
    """Return 13-char normalized base SKU for 14-char sizeable SKUs."""
    if len(sku) == 14:
        return sku[:13]
    return sku


def _to_num_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.fillna("0")
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("INR", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


def _first_present_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.strip().lower() in lower_map:
            return lower_map[cand.strip().lower()]
    return None


def _first_non_empty(series: pd.Series) -> Optional[str]:
    for val in series:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        s = str(val).strip()
        if s and s.lower() not in {"nan", "none", "null"}:
            return s
    return None


@dataclass
class ETLStats:
    web_total: int = 0
    web_excluded: int = 0
    web_skus: int = 0
    oms_total: int = 0
    oms_excluded: int = 0
    oms_skus: int = 0
    perf_total: int = 0
    perf_excluded: int = 0
    perf_matched_rows: int = 0
    perf_unmatched_rows: int = 0
    perf_purchased_matched_rows: int = 0
    perf_purchased_unmatched_rows: int = 0
    perf_skus: int = 0
    master_total: int = 0
    master_excluded: int = 0
    master_skus: int = 0
    app_total: int = 0
    app_excluded: int = 0
    app_rows_matched: int = 0
    app_purchased_rows: int = 0
    app_purchased_rows_matched: int = 0
    app_skus: int = 0
    period_skus: int = 0
    match_exact: int = 0
    match_size_13: int = 0
    match_title: int = 0
    match_unmatched: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def perf_join_rate(self) -> float:
        denom = self.perf_matched_rows + self.perf_unmatched_rows
        return round((self.perf_matched_rows / denom) * 100.0, 2) if denom > 0 else 0.0

    @property
    def perf_purchased_join_rate(self) -> float:
        denom = self.perf_purchased_matched_rows + self.perf_purchased_unmatched_rows
        return round((self.perf_purchased_matched_rows / denom) * 100.0, 2) if denom > 0 else 0.0

    def master_match_pct(self) -> Dict[str, float]:
        denom = self.period_skus or 1
        return {
            "exact": round((self.match_exact / denom) * 100.0, 2),
            "size_13": round((self.match_size_13 / denom) * 100.0, 2),
            "title": round((self.match_title / denom) * 100.0, 2),
            "unmatched": round((self.match_unmatched / denom) * 100.0, 2),
        }


def parse_web_csv(path: str) -> Tuple[pd.DataFrame, int, int]:
    """Parse and aggregate Web CSV to 1 row per sku_code.

    Returns (df, total_rows, excluded_rows).
    """
    raw = pd.read_csv(path, dtype=str)
    total_rows = len(raw)
    id_col = _first_present_col(raw, ["Product ID", "sku_code", "Item ID"])
    if not id_col:
        raise ValueError(f"Web CSV missing Product ID column: {list(raw.columns)}")

    raw["sku_code"] = raw[id_col].map(normalize_sku)
    raw = raw[raw["sku_code"] != ""].copy()

    filtered, excluded = filter_jewellery(raw)
    if filtered.empty:
        return pd.DataFrame(columns=["sku_code"]), total_rows, excluded

    col_map = {
        "web_clicks": _first_present_col(filtered, ["Items clicked in list"]),
        "web_views": _first_present_col(filtered, ["Items viewed"]),
        "web_atc": _first_present_col(filtered, ["Items added to cart"]),
        "web_purchased": _first_present_col(filtered, ["Items purchased"]),
        "web_revenue": _first_present_col(filtered, ["Gross item revenue"]),
    }
    for metric, src in col_map.items():
        filtered[metric] = _to_num_series(filtered[src]) if src else 0.0

    name_col = _first_present_col(filtered, ["Product Name", "product_name", "Item name"])
    cat_col = _first_present_col(filtered, ["Category", "category"])
    price_col = _first_present_col(filtered, ["Item price", "Price", "price"])
    band_col = _first_present_col(filtered, ["Price Range", "price_range"])

    filtered["product_name"] = filtered[name_col].fillna("").astype(str).str.strip() if name_col else ""
    filtered["web_category"] = filtered[cat_col].fillna("").astype(str).str.strip() if cat_col else ""
    filtered["_unit_price"] = _to_num_series(filtered[price_col]) if price_col else 0.0
    # Listed price band (GA4 item parameter). Present for every viewed item, so unlike
    # purchase-derived price it does not reveal whether the item sold.
    filtered["price_band"] = (
        filtered[band_col].fillna("").astype(str).str.strip().str.lower().map(PRICE_BAND_ALIASES)
        if band_col
        else None
    )

    agg = (
        filtered.groupby("sku_code", as_index=False)
        .agg(
            product_name=("product_name", _first_non_empty),
            web_category=("web_category", _first_non_empty),
            price_band=("price_band", _first_non_empty),
            web_clicks=("web_clicks", "sum"),
            web_views=("web_views", "sum"),
            web_atc=("web_atc", "sum"),
            web_purchased=("web_purchased", "sum"),
            web_revenue=("web_revenue", "sum"),
            _fallback_price=("_unit_price", lambda s: float(s[s > 0].mean()) if (s > 0).any() else np.nan),
        )
    )

    agg["avg_price"] = np.where(
        (agg["web_purchased"] > 0) & (agg["web_revenue"] > 0),
        agg["web_revenue"] / agg["web_purchased"],
        agg["_fallback_price"],
    )
    agg["web_atc_rate"] = np.where(agg["web_views"] > 0, agg["web_atc"] / agg["web_views"], 0.0)
    agg["web_conv_rate"] = np.where(agg["web_views"] > 0, agg["web_purchased"] / agg["web_views"], 0.0)
    agg["web_cart_abandon_rate"] = np.where(
        agg["web_atc"] > 0,
        np.maximum(0.0, (agg["web_atc"] - agg["web_purchased"]) / agg["web_atc"]),
        0.0,
    )
    agg = agg.drop(columns=["_fallback_price"])
    return agg, total_rows, excluded


APP_METRIC_COLS = ("app_clicks", "app_views", "app_atc", "app_purchases", "app_revenue")


def build_name_to_skus(
    web_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame],
) -> Dict[str, List[Tuple[str, float]]]:
    """Map normalised product names to candidate SKUs with split weights.

    Web names come with SKU IDs and view counts, so they are preferred; a name shared
    by several SKUs is split in proportion to their web views. Master titles are the
    fallback, split equally (published SKUs only when any exist for that title).
    """
    out: Dict[str, List[Tuple[str, float]]] = {}
    if web_df is not None and not web_df.empty and "product_name" in web_df.columns:
        tmp = web_df[["sku_code", "product_name", "web_views"]].copy()
        tmp["norm"] = tmp["product_name"].map(normalize_name)
        tmp = tmp[tmp["norm"] != ""]
        for norm, grp in tmp.groupby("norm"):
            views = grp["web_views"].astype(float).clip(lower=0).to_numpy()
            weights = views / views.sum() if views.sum() > 0 else np.full(len(grp), 1.0 / len(grp))
            out[norm] = list(zip(grp["sku_code"].tolist(), weights.tolist()))

    if master_df is not None and not master_df.empty and "norm_title" in master_df.columns:
        tmp = master_df[["sku_code", "norm_title"]].copy()
        tmp["published"] = master_df["published"] if "published" in master_df.columns else False
        tmp = tmp[tmp["norm_title"].fillna("") != ""]
        for norm, grp in tmp.groupby("norm_title"):
            if norm in out:
                continue
            pub = grp[grp["published"] == True]  # noqa: E712
            cands = (pub if not pub.empty else grp)["sku_code"].tolist()
            out[norm] = [(c, 1.0 / len(cands)) for c in cands]
    return out


def parse_app_csv(
    path: str,
    name_to_skus: Dict[str, List[Tuple[str, float]]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse the App (GA4) CSV, which has product names but no SKU IDs.

    Each row is attributed to SKUs via `name_to_skus`; ambiguous names are split by weight.
    Returns (per-SKU df, stats dict).
    """
    raw = pd.read_csv(path, dtype=str)
    stats = {"total": len(raw), "excluded": 0, "matched": 0, "purchased": 0, "purchased_matched": 0}
    name_col = _first_present_col(raw, ["Product Name", "product_name", "Item name"])
    if not name_col:
        raise ValueError(f"App CSV missing Product Name column: {list(raw.columns)}")

    filtered, excluded = filter_jewellery(raw)
    stats["excluded"] = excluded
    if filtered.empty:
        return pd.DataFrame(columns=["sku_code", *APP_METRIC_COLS]), stats

    src = {
        "app_clicks": _first_present_col(filtered, ["Items clicked in list"]),
        "app_views": _first_present_col(filtered, ["Items viewed"]),
        "app_atc": _first_present_col(filtered, ["Items added to cart"]),
        "app_purchases": _first_present_col(filtered, ["Items purchased"]),
        "app_revenue": _first_present_col(filtered, ["Gross item revenue"]),
    }
    metrics = pd.DataFrame(
        {k: (_to_num_series(filtered[c]) if c else 0.0) for k, c in src.items()},
        index=filtered.index,
    )
    norms = filtered[name_col].map(normalize_name)
    cat_col = _first_present_col(filtered, ["Category", "category"])
    app_cats = filtered[cat_col].fillna("").astype(str) if cat_col else pd.Series("", index=filtered.index)

    rows: List[Dict[str, Any]] = []
    for idx in filtered.index:
        cands = name_to_skus.get(norms[idx])
        bought = metrics.at[idx, "app_purchases"] > 0
        stats["purchased"] += int(bought)
        if not cands:
            continue
        stats["matched"] += 1
        stats["purchased_matched"] += int(bought)
        for sku, w in cands:
            rec = {"sku_code": sku, "app_category": app_cats[idx]}
            for col in APP_METRIC_COLS:
                rec[col] = float(metrics.at[idx, col]) * w
            rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=["sku_code", *APP_METRIC_COLS]), stats
    df = pd.DataFrame(rows)
    agg = df.groupby("sku_code", as_index=False).agg(
        app_category=("app_category", _first_non_empty),
        **{col: (col, "sum") for col in APP_METRIC_COLS},
    )
    return agg, stats


def parse_oms_csv(path: str) -> Tuple[pd.DataFrame, int, int]:
    """Parse and aggregate OMS CSV to 1 row per sku_code.

    Returns (df, total_rows, excluded_rows).
    """
    raw = pd.read_csv(path, dtype=str)
    total_rows = len(raw)
    id_col = _first_present_col(raw, ["nproductId", "sku_code", "Product ID"])
    if not id_col:
        raise ValueError(f"OMS CSV missing nproductId column: {list(raw.columns)}")

    raw["sku_code"] = raw[id_col].map(normalize_sku)
    raw = raw[raw["sku_code"] != ""].copy()

    filtered, excluded = filter_jewellery(raw)
    if filtered.empty:
        return pd.DataFrame(columns=["sku_code"]), total_rows, excluded

    units_col = _first_present_col(filtered, ["units", "Units"])
    rev_col = _first_present_col(filtered, ["ste_revenue", "revenue"])
    ratio_col = _first_present_col(filtered, ["ste_ratioOfTotalRevenue", "ratioOfTotalRevenue"])

    filtered["oms_units"] = _to_num_series(filtered[units_col]) if units_col else 0.0
    filtered["oms_revenue"] = _to_num_series(filtered[rev_col]) if rev_col else 0.0
    filtered["oms_revenue_ratio"] = _to_num_series(filtered[ratio_col]) if ratio_col else 0.0
    name_col = _first_present_col(filtered, ["productName", "Product Name"])
    filtered["oms_product_name"] = filtered[name_col].fillna("").astype(str).str.strip() if name_col else ""

    agg = filtered.groupby("sku_code", as_index=False).agg(
        oms_product_name=("oms_product_name", _first_non_empty),
        oms_units=("oms_units", "sum"),
        oms_revenue=("oms_revenue", "sum"),
        oms_revenue_ratio=("oms_revenue_ratio", "sum"),
    )
    return agg, total_rows, excluded


def _find_perf_header_row(path: str) -> Tuple[int, int]:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            lowered = {cell.strip().lower() for cell in row}
            if "item id" in lowered and "item name" in lowered:
                return idx, len(row)
    raise ValueError(f"Could not find GA4 header row ('Item ID' and 'Item name') in {path}")


def parse_perf_csv(
    path: str,
    known_skus: Set[str],
    web_name_to_sku: Dict[str, str],
    master_name_to_sku: Dict[str, str],
) -> Tuple[pd.DataFrame, int, int, int, int, int, int]:
    """Parse GA4 Perf CSV and perform 2-tier join to SKU.

    Returns (agg_df, total_rows, excluded_rows, matched_rows, unmatched_rows,
             purchased_matched_rows, purchased_unmatched_rows).
    """
    header_idx, header_len = _find_perf_header_row(path)
    all_names = [
        "page_path",
        "item_id",
        "item_name",
        "perf_users",
        "perf_purchased",
        "total_users",
        "total_purchased",
    ]
    n_cols = 7 if header_len >= 7 else min(header_len, 5)
    raw = pd.read_csv(
        path,
        skiprows=header_idx + 1,
        names=all_names[:n_cols],
        usecols=range(n_cols),
        index_col=False,
        dtype=str,
    )

    raw["item_id"] = raw["item_id"].fillna("").astype(str).str.strip()
    raw["item_name"] = raw["item_name"].fillna("").astype(str).str.strip()

    # Drop Grand total and empty-key summary rows
    is_grand_or_empty = (
        (raw["item_id"] == "")
        | (raw["item_name"] == "")
        | raw["item_id"].str.lower().isin({"grand total", "nan", "none"})
        | raw["item_name"].str.lower().isin({"grand total", "nan", "none"})
    )
    raw = raw.loc[~is_grand_or_empty].copy()
    total_rows = len(raw)

    work = pd.DataFrame(
        {
            "page_path": raw["page_path"].fillna("").astype(str).str.strip(),
            "item_id": raw["item_id"],
            "item_name": raw["item_name"],
            "perf_users": _to_num_series(raw["perf_users"]).fillna(0.0),
            "perf_purchased": _to_num_series(raw["perf_purchased"]).fillna(0.0),
        }
    )

    filtered, excluded = filter_jewellery(work, sku_cols=("item_id",), text_cols=("item_name",))
    if filtered.empty:
        return (
            pd.DataFrame(columns=["sku_code", "perf_users", "perf_units"]),
            total_rows,
            excluded,
            0,
            0,
            0,
            0,
        )

    def _resolve_row(row: pd.Series) -> str:
        item_id_upper = normalize_sku(row["item_id"])
        # Tier 1: Exact match Item ID against known SKUs
        if item_id_upper and item_id_upper in known_skus:
            return item_id_upper

        # Tier 2: Normalized Item name match against Web Product Name, then Master TITLE
        norm_title = normalize_name(row["item_name"])
        if norm_title:
            if norm_title in web_name_to_sku:
                return web_name_to_sku[norm_title]
            if norm_title in master_name_to_sku:
                return master_name_to_sku[norm_title]
        return ""

    filtered["sku_code"] = filtered.apply(_resolve_row, axis=1)
    is_matched = filtered["sku_code"] != ""
    is_purchased = filtered["perf_purchased"] > 0

    matched_rows = int(is_matched.sum())
    unmatched_rows = int((~is_matched).sum())
    purchased_matched_rows = int((is_matched & is_purchased).sum())
    purchased_unmatched_rows = int((~is_matched & is_purchased).sum())

    matched = filtered.loc[is_matched].copy()
    if matched.empty:
        return (
            pd.DataFrame(columns=["sku_code", "perf_users", "perf_units"]),
            total_rows,
            excluded,
            matched_rows,
            unmatched_rows,
            purchased_matched_rows,
            purchased_unmatched_rows,
        )

    agg = matched.groupby("sku_code", as_index=False).agg(
        perf_users=("perf_users", "sum"),
        perf_units=("perf_purchased", "sum"),
    )
    return (
        agg,
        total_rows,
        excluded,
        matched_rows,
        unmatched_rows,
        purchased_matched_rows,
        purchased_unmatched_rows,
    )


def parse_master_csv(path: str) -> Tuple[pd.DataFrame, int, int]:
    """Parse Master Enrichment CSV robustly and extract design DNA.

    Returns (df, total_rows, excluded_rows).
    """
    raw = pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)
    total_rows = len(raw)

    sku_col = _first_present_col(raw, ["BASE SKU", "sku_code", "Product ID"])
    if not sku_col:
        raise ValueError(f"Master CSV missing BASE SKU column: {list(raw.columns)}")

    raw["sku_code"] = raw[sku_col].map(normalize_sku)
    raw = raw[raw["sku_code"] != ""].copy()

    filtered, excluded = filter_jewellery(raw)
    if filtered.empty:
        return pd.DataFrame(columns=["sku_code"]), total_rows, excluded

    filtered = filtered.drop_duplicates(subset=["sku_code"], keep="first").reset_index(drop=True)
    filtered["base_sku_13"] = filtered["sku_code"].map(to_base_sku_13)

    title_col = _first_present_col(filtered, ["TITLE", "title"])
    filtered["master_title"] = filtered[title_col].fillna("").astype(str).str.strip() if title_col else ""
    filtered["norm_title"] = filtered["master_title"].map(normalize_name)

    def _pick(col_names: List[str]) -> pd.Series:
        c = _first_present_col(filtered, col_names)
        return filtered[c].fillna("").astype(str).str.strip() if c else pd.Series("", index=filtered.index)

    filtered["category"] = [
        canonical_category(c, p, name=t)
        for c, p, t in zip(_pick(["CATEGORY", "Category"]), _pick(["PRODUCT"]), filtered["master_title"])
    ]
    filtered["product_type"] = _pick(["JEWELLERY_TYPE", "PRODUCT TYPE", "PRODUCT_TYPE"])
    filtered["metal"] = _pick(["METAL"])
    filtered["metal_color"] = _pick(["METAL COLOR", "METAL_COLOR"])
    filtered["karatage"] = _pick(["GOLD_KARATAGE", "KARATAGE"])
    filtered["diamond_clarity"] = _pick(["DIAMOND_CLARITY"])
    filtered["weight_band"] = _pick(["WEIGHT BAND", "WEIGHT_BAND"])
    _yes = {"1", "true", "yes", "y"}
    filtered["published"] = _pick(["Published", "PUBLISHED"]).str.lower().isin(_yes)
    filtered["is_new_collection"] = _pick(["NEW COLLECTION"]).str.lower().isin(_yes)
    filtered["is_best_deal"] = _pick(["BEST DEAL"]).str.lower().isin(_yes)
    filtered["gender"] = _pick(["GENDER"])
    filtered["occasion"] = _pick(["OCCASION"])
    filtered["collection"] = _pick(["COLLECTION"])
    filtered["design"] = _pick(["DESIGN"])
    filtered["styling"] = _pick(["STYLING"])
    filtered["short_description"] = _pick(["SHORT DESCRIPTION"])
    filtered["long_description"] = _pick(["LONG DESCRIPTION"])
    img_col = _first_present_col(filtered, ["IMAGES 1", "IMAGE 1", "image_url"])
    if img_col:
        filtered["image_url"] = (
            filtered[img_col]
            .where(pd.notnull(filtered[img_col]), None)
            .map(lambda v: str(v).strip() if v is not None and str(v).strip() not in {"", "nan", "None", "null"} else None)
        )
    else:
        filtered["image_url"] = None

    filtered["design_tags"] = filtered.apply(lambda r: extract_design_dna(r.to_dict()), axis=1)
    return filtered, total_rows, excluded


def run_etl_pipeline(
    web_path: Optional[str] = None,
    oms_path: Optional[str] = None,
    perf_path: Optional[str] = None,
    master_path: Optional[str] = None,
    period_start: str = "",
    period_end: str = "",
    granularity: str = "monthly",
    app_path: Optional[str] = None,
    master_lookup_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, ETLStats]:
    """Execute the full multi-source ETL pipeline.

    `master_lookup_df` (sku_code, title[, category, published]) lets a period upload
    without a master CSV still resolve names and categories against the stored catalog.

    Returns (period_metrics_df, master_attributes_df, stats).
    """
    stats = ETLStats()

    # 1. Master Enrichment
    if master_path:
        master_df, m_tot, m_excl = parse_master_csv(master_path)
    else:
        master_df, m_tot, m_excl = pd.DataFrame(columns=["sku_code"]), 0, 0

    # Catalog used only for name/category lookups (never written back)
    lookup_df = master_df
    if lookup_df.empty and master_lookup_df is not None and not master_lookup_df.empty:
        lookup_df = master_lookup_df.copy()
        lookup_df["sku_code"] = lookup_df["sku_code"].map(normalize_sku)
        title_src = "master_title" if "master_title" in lookup_df.columns else "title"
        lookup_df["master_title"] = lookup_df.get(title_src, pd.Series("", index=lookup_df.index)).fillna("")
        lookup_df["norm_title"] = lookup_df["master_title"].map(normalize_name)
        if "base_sku_13" not in lookup_df.columns:
            lookup_df["base_sku_13"] = lookup_df["sku_code"].map(to_base_sku_13)
    stats.master_total = m_tot
    stats.master_excluded = m_excl
    stats.master_skus = len(master_df)

    # 2. Web CSV
    if web_path:
        web_df, w_tot, w_excl = parse_web_csv(web_path)
    else:
        web_df, w_tot, w_excl = pd.DataFrame(columns=["sku_code"]), 0, 0
    stats.web_total = w_tot
    stats.web_excluded = w_excl
    stats.web_skus = len(web_df)

    # 3. OMS CSV
    if oms_path:
        oms_df, o_tot, o_excl = parse_oms_csv(oms_path)
    else:
        oms_df, o_tot, o_excl = pd.DataFrame(columns=["sku_code"]), 0, 0
    stats.oms_total = o_tot
    stats.oms_excluded = o_excl
    stats.oms_skus = len(oms_df)

    # Build lookup dicts for Perf 2-tier join
    known_skus: Set[str] = set()
    for df in (web_df, oms_df, lookup_df):
        if not df.empty and "sku_code" in df.columns:
            known_skus.update(df["sku_code"].dropna().astype(str).tolist())

    web_name_to_sku: Dict[str, str] = {}
    if not web_df.empty and "product_name" in web_df.columns:
        for _, row in web_df.iterrows():
            nm = normalize_name(row.get("product_name"))
            if nm and nm not in web_name_to_sku:
                web_name_to_sku[nm] = row["sku_code"]

    master_name_to_sku: Dict[str, str] = {}
    if not lookup_df.empty and "norm_title" in lookup_df.columns:
        for _, row in lookup_df.iterrows():
            nm = row.get("norm_title") or ""
            if nm and nm not in master_name_to_sku:
                master_name_to_sku[nm] = row["sku_code"]

    # 4. Perf CSV
    if perf_path:
        perf_df, p_tot, p_excl, p_matched, p_unmatched, p_purch_m, p_purch_u = parse_perf_csv(
            perf_path, known_skus, web_name_to_sku, master_name_to_sku
        )
    else:
        perf_df = pd.DataFrame(columns=["sku_code", "perf_users", "perf_units"])
        p_tot, p_excl, p_matched, p_unmatched, p_purch_m, p_purch_u = 0, 0, 0, 0, 0, 0

    stats.perf_total = p_tot
    stats.perf_excluded = p_excl
    stats.perf_matched_rows = p_matched
    stats.perf_unmatched_rows = p_unmatched
    stats.perf_purchased_matched_rows = p_purch_m
    stats.perf_purchased_unmatched_rows = p_purch_u
    stats.perf_skus = len(perf_df)

    # 4b. App CSV (names only -> attributed to SKUs via web names, then catalog titles)
    if app_path:
        app_df, app_stats = parse_app_csv(app_path, build_name_to_skus(web_df, lookup_df))
    else:
        app_df, app_stats = pd.DataFrame(columns=["sku_code"]), {}
    stats.app_total = app_stats.get("total", 0)
    stats.app_excluded = app_stats.get("excluded", 0)
    stats.app_rows_matched = app_stats.get("matched", 0)
    stats.app_purchased_rows = app_stats.get("purchased", 0)
    stats.app_purchased_rows_matched = app_stats.get("purchased_matched", 0)
    stats.app_skus = len(app_df)

    # 5. Union all period SKUs (Web + App + OMS + resolved Perf) — keep OOS SKUs too
    period_df = pd.DataFrame({"sku_code": sorted(
        set(web_df.get("sku_code", pd.Series(dtype=str)))
        | set(app_df.get("sku_code", pd.Series(dtype=str)))
        | set(oms_df.get("sku_code", pd.Series(dtype=str)))
        | set(perf_df.get("sku_code", pd.Series(dtype=str)))
    )})
    stats.period_skus = len(period_df)

    if not web_df.empty:
        period_df = period_df.merge(web_df, on="sku_code", how="left")
    if not app_df.empty:
        period_df = period_df.merge(app_df, on="sku_code", how="left")
    if not oms_df.empty:
        period_df = period_df.merge(oms_df, on="sku_code", how="left")
    if not perf_df.empty:
        period_df = period_df.merge(perf_df, on="sku_code", how="left")

    # Fill numeric defaults cleanly
    num_defaults = [
        "web_clicks",
        "web_views",
        "web_atc",
        "web_purchased",
        "web_revenue",
        "web_atc_rate",
        "web_conv_rate",
        "web_cart_abandon_rate",
        "oms_units",
        "oms_revenue",
        "oms_revenue_ratio",
        "perf_users",
        "perf_units",
        *APP_METRIC_COLS,
    ]
    for col in num_defaults:
        if col not in period_df.columns:
            period_df[col] = 0.0
        else:
            period_df[col] = period_df[col].fillna(0.0)

    # Units sold: OMS already contains web/app orders, so take the larger figure
    # rather than adding them (adding would double count).
    period_df["units_sold"] = np.maximum(
        period_df["oms_units"], period_df["web_purchased"] + period_df["app_purchases"]
    )
    for col in ("web_category", "app_category", "price_band"):
        if col not in period_df.columns:
            period_df[col] = None

    if "product_name" not in period_df.columns:
        period_df["product_name"] = None
    if "oms_product_name" in period_df.columns:
        # OMS-only SKUs (no web/app traffic) still get a name, and so a category.
        period_df["product_name"] = period_df["product_name"].where(
            period_df["product_name"].notna() & (period_df["product_name"].astype(str).str.strip() != ""),
            period_df["oms_product_name"],
        )
    if "avg_price" not in period_df.columns:
        period_df["avg_price"] = np.nan

    # 6. 3-Tier Master Attribute Matcher
    master_exact_map: Dict[str, Dict[str, Any]] = {}
    master_13_map: Dict[str, Dict[str, Any]] = {}
    master_title_map: Dict[str, Dict[str, Any]] = {}

    if not lookup_df.empty:
        for _, mrow in lookup_df.iterrows():
            rec = mrow.to_dict()
            code = rec["sku_code"]
            b13 = rec.get("base_sku_13") or ""
            ntitle = rec.get("norm_title") or ""
            master_exact_map[code] = rec
            if b13 and b13 not in master_13_map:
                master_13_map[b13] = rec
            if ntitle and ntitle not in master_title_map:
                master_title_map[ntitle] = rec

    match_tiers: List[str] = []
    resolved_titles: List[Optional[str]] = []
    resolved_categories: List[Optional[str]] = []
    resolved_dna: List[Dict[str, List[str]]] = []

    for _, prow in period_df.iterrows():
        code = prow["sku_code"]
        b13 = to_base_sku_13(code)
        pname = prow.get("product_name")
        norm_pname = normalize_name(pname)

        mrec: Optional[Dict[str, Any]] = None
        tier = "unmatched"
        if code in master_exact_map:
            mrec = master_exact_map[code]
            tier = "exact"
            stats.match_exact += 1
        elif b13 in master_13_map:
            mrec = master_13_map[b13]
            tier = "size_13"
            stats.match_size_13 += 1
        elif norm_pname and norm_pname in master_title_map:
            mrec = master_title_map[norm_pname]
            tier = "title"
            stats.match_title += 1
        else:
            stats.match_unmatched += 1

        match_tiers.append(tier)
        if mrec:
            resolved_titles.append(pname or mrec.get("master_title") or None)
            resolved_categories.append(
                canonical_category(
                    mrec.get("category"),
                    prow.get("web_category"),
                    prow.get("app_category"),
                    name=pname or mrec.get("master_title"),
                )
            )
            combined_row = dict(mrec)
            if pname:
                combined_row["Product Name"] = pname
            resolved_dna.append(extract_design_dna(combined_row))
        else:
            resolved_titles.append(pname or None)
            resolved_categories.append(
                canonical_category(prow.get("web_category"), prow.get("app_category"), name=pname)
            )
            resolved_dna.append(extract_design_dna({"Product Name": pname or ""}))

    period_df["master_match_tier"] = match_tiers
    period_df["product_name"] = resolved_titles
    period_df["category"] = resolved_categories
    period_df["design_tags"] = resolved_dna
    period_df["period_start"] = period_start
    period_df["period_end"] = period_end
    period_df["granularity"] = granularity

    # Prepare clean master_attributes_df
    if not master_df.empty:
        master_out_cols = [
            "sku_code",
            "base_sku_13",
            "master_title",
            "category",
            "product_type",
            "metal",
            "metal_color",
            "karatage",
            "diamond_clarity",
            "weight_band",
            "gender",
            "occasion",
            "collection",
            "design",
            "styling",
            "published",
            "is_new_collection",
            "is_best_deal",
            "short_description",
            "long_description",
            "image_url",
            "design_tags",
        ]
        master_attrs_df = master_df[[c for c in master_out_cols if c in master_df.columns]].copy()
    else:
        master_attrs_df = pd.DataFrame(columns=["sku_code", "design_tags"])

    return period_df, master_attrs_df, stats


SKU_PERIOD_METRICS_COLS = (
    "sku_code",
    "period_start",
    "period_end",
    "granularity",
    "product_name",
    "category",
    "price_range",
    "avg_price",
    "web_clicks",
    "web_views",
    "web_atc",
    "web_purchases",
    "web_revenue",
    "web_atc_rate",
    "web_conv_rate",
    "web_cart_abandon_rate",
    "oms_units",
    "oms_revenue_inr",
    "oms_rev_share",
    "perf_users",
    "perf_units",
    "match_tier",
    "price_band",
    "app_clicks",
    "app_views",
    "app_atc",
    "app_purchases",
    "app_revenue",
    "units_sold",
)

SKU_MASTER_ATTRIBUTES_COLS = (
    "sku_code",
    "base_sku_13",
    "title",
    "category",
    "collection",
    "metal",
    "gold_karatage",
    "metal_color",
    "jewellery_type",
    "occasion",
    "diamond_clarity",
    "weight_band",
    "design",
    "styling",
    "is_new_collection",
    "is_best_deal",
    "short_description",
    "long_description",
    "design_tags",
    "image_url",
    "gender",
    "published",
)


_PERIOD_INT_COLS = {
    "web_clicks",
    "web_views",
    "web_atc",
    "web_purchases",
    "oms_units",
    "perf_users",
    "perf_units",
}
_PERIOD_FLOAT_DECIMALS = {
    "avg_price": 2,
    "web_revenue": 2,
    "oms_revenue_inr": 2,
    "web_atc_rate": 6,
    "web_conv_rate": 6,
    "web_cart_abandon_rate": 6,
    "oms_rev_share": 8,
    # App rows shared by several SKUs are split by weight, so these can be fractional.
    "app_clicks": 2,
    "app_views": 2,
    "app_atc": 2,
    "app_purchases": 2,
    "app_revenue": 2,
    "units_sold": 2,
}


def _clean_record_nans(rec: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in list(rec.items()):
        if isinstance(v, float) and pd.isna(v):
            rec[k] = None
    return rec


def _dedupe_records(records: List[Dict[str, Any]], key_cols: Tuple[str, ...]) -> List[Dict[str, Any]]:
    """Deduplicate records on conflict key (last write wins) to avoid batch duplicate-key errors."""
    deduped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for rec in records:
        key = tuple(rec.get(k) for k in key_cols)
        deduped[key] = rec
    return list(deduped.values())


_SOURCE_PERIOD_COLS = {
    "web": {"web_clicks", "web_views", "web_atc", "web_purchases", "web_revenue", "web_atc_rate",
            "web_conv_rate", "web_cart_abandon_rate", "avg_price", "price_band", "product_name", "category", "match_tier"},
    "app": {"app_clicks", "app_views", "app_atc", "app_purchases", "app_revenue"},
    "oms": {"oms_units", "oms_revenue_inr", "oms_rev_share"},
    "perf": {"perf_users", "perf_units"},
}
_PERIOD_KEY_COLS = {"sku_code", "period_start", "period_end", "granularity"}


def prepare_period_metrics_records(
    period_df: pd.DataFrame,
    sources: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Map, cast, deduplicate, and filter period_df rows to exact sku_period_metrics SQL columns.

    With `sources`, only columns owned by those sources are written, so a partial upload
    for a period does not overwrite other sources' stored values with zeros. `units_sold`
    is only written when Web, App and OMS are all present (it needs all three); the model
    recomputes it from stored columns anyway.
    """
    if period_df.empty:
        return []
    df = period_df.rename(
        columns={
            "web_purchased": "web_purchases",
            "oms_revenue": "oms_revenue_inr",
            "oms_revenue_ratio": "oms_rev_share",
            "master_match_tier": "match_tier",
        }
    )
    allowed = set(SKU_PERIOD_METRICS_COLS)
    if sources is not None:
        owned = set(_PERIOD_KEY_COLS)
        for src in sources:
            owned |= _SOURCE_PERIOD_COLS.get(src, set())
        if {"web", "app", "oms"} <= set(sources):
            owned.add("units_sold")
        allowed &= owned
    cols = [c for c in SKU_PERIOD_METRICS_COLS if c in df.columns and c in allowed]
    records = df[cols].where(pd.notnull(df[cols]), None).to_dict(orient="records")
    cleaned: List[Dict[str, Any]] = []
    for r in records:
        rec = _clean_record_nans({k: v for k, v in r.items() if k in allowed})
        for col in _PERIOD_INT_COLS:
            if col in rec and rec[col] is not None:
                rec[col] = int(round(float(rec[col])))
        for col, dec in _PERIOD_FLOAT_DECIMALS.items():
            if col in rec and rec[col] is not None:
                rec[col] = round(float(rec[col]), dec)
        cleaned.append(rec)
    return _dedupe_records(cleaned, ("sku_code", "period_start", "period_end"))


def prepare_master_attribute_records(master_attrs_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Map, deduplicate, and filter master_attrs_df rows to exact sku_master_attributes SQL columns."""
    if master_attrs_df.empty:
        return []
    df = master_attrs_df.rename(
        columns={
            "master_title": "title",
            "karatage": "gold_karatage",
            "product_type": "jewellery_type",
        }
    )
    allowed = set(SKU_MASTER_ATTRIBUTES_COLS)
    cols = [c for c in SKU_MASTER_ATTRIBUTES_COLS if c in df.columns]
    records = df[cols].where(pd.notnull(df[cols]), None).to_dict(orient="records")
    cleaned = [_clean_record_nans({k: v for k, v in r.items() if k in allowed}) for r in records]
    return _dedupe_records(cleaned, ("sku_code",))


