import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from intelligence.design_dna import extract_design_dna
from intelligence.filters import filter_jewellery


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

    filtered["product_name"] = filtered[name_col].fillna("").astype(str).str.strip() if name_col else ""
    filtered["web_category"] = filtered[cat_col].fillna("").astype(str).str.strip() if cat_col else ""
    filtered["_unit_price"] = _to_num_series(filtered[price_col]) if price_col else 0.0

    agg = (
        filtered.groupby("sku_code", as_index=False)
        .agg(
            product_name=("product_name", _first_non_empty),
            web_category=("web_category", _first_non_empty),
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

    agg = filtered.groupby("sku_code", as_index=False).agg(
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

    filtered["category"] = _pick(["CATEGORY", "Category"])
    filtered["product_type"] = _pick(["PRODUCT TYPE", "PRODUCT_TYPE"])
    filtered["metal_color"] = _pick(["METAL COLOR", "METAL_COLOR"])
    filtered["karatage"] = _pick(["KARATAGE"])
    filtered["gender"] = _pick(["GENDER"])
    filtered["occasion"] = _pick(["OCCASION"])
    filtered["collection"] = _pick(["COLLECTION"])
    filtered["design"] = _pick(["DESIGN"])
    filtered["styling"] = _pick(["STYLING"])
    filtered["short_description"] = _pick(["SHORT DESCRIPTION"])
    filtered["long_description"] = _pick(["LONG DESCRIPTION"])

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
) -> Tuple[pd.DataFrame, pd.DataFrame, ETLStats]:
    """Execute the full multi-source ETL pipeline.

    Returns (period_metrics_df, master_attributes_df, stats).
    """
    stats = ETLStats()

    # 1. Master Enrichment
    if master_path:
        master_df, m_tot, m_excl = parse_master_csv(master_path)
    else:
        master_df, m_tot, m_excl = pd.DataFrame(columns=["sku_code"]), 0, 0
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
    for df in (web_df, oms_df, master_df):
        if not df.empty and "sku_code" in df.columns:
            known_skus.update(df["sku_code"].dropna().astype(str).tolist())

    web_name_to_sku: Dict[str, str] = {}
    if not web_df.empty and "product_name" in web_df.columns:
        for _, row in web_df.iterrows():
            nm = normalize_name(row.get("product_name"))
            if nm and nm not in web_name_to_sku:
                web_name_to_sku[nm] = row["sku_code"]

    master_name_to_sku: Dict[str, str] = {}
    if not master_df.empty and "norm_title" in master_df.columns:
        for _, row in master_df.iterrows():
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

    # 5. Union all period SKUs (Web + OMS + resolved Perf) — keep OOS SKUs too
    period_df = pd.DataFrame({"sku_code": sorted(
        set(web_df.get("sku_code", pd.Series(dtype=str)))
        | set(oms_df.get("sku_code", pd.Series(dtype=str)))
        | set(perf_df.get("sku_code", pd.Series(dtype=str)))
    )})
    stats.period_skus = len(period_df)

    if not web_df.empty:
        period_df = period_df.merge(web_df, on="sku_code", how="left")
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
    ]
    for col in num_defaults:
        if col not in period_df.columns:
            period_df[col] = 0.0
        else:
            period_df[col] = period_df[col].fillna(0.0)

    if "product_name" not in period_df.columns:
        period_df["product_name"] = None
    if "avg_price" not in period_df.columns:
        period_df["avg_price"] = np.nan

    # 6. 3-Tier Master Attribute Matcher
    master_exact_map: Dict[str, Dict[str, Any]] = {}
    master_13_map: Dict[str, Dict[str, Any]] = {}
    master_title_map: Dict[str, Dict[str, Any]] = {}

    if not master_df.empty:
        for _, mrow in master_df.iterrows():
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
            resolved_categories.append(mrec.get("category") or prow.get("web_category") or None)
            combined_row = dict(mrec)
            if pname:
                combined_row["Product Name"] = pname
            resolved_dna.append(extract_design_dna(combined_row))
        else:
            resolved_titles.append(pname or None)
            resolved_categories.append(prow.get("web_category") or None)
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
            "metal_color",
            "karatage",
            "gender",
            "occasion",
            "collection",
            "design",
            "styling",
            "design_tags",
        ]
        master_attrs_df = master_df[[c for c in master_out_cols if c in master_df.columns]].copy()
    else:
        master_attrs_df = pd.DataFrame(columns=["sku_code", "design_tags"])

    return period_df, master_attrs_df, stats

