import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from intelligence.design_dna import DESIGN_AXES

STRUCTURED_CAT_COLS = [
    "category",
    "price_range",
    "collection",
    "metal",
    "gold_karatage",
    "metal_color",
    "jewellery_type",
    "occasion",
    "diamond_clarity",
    "weight_band",
    "is_new_collection",
    "is_best_deal",
]


def _parse_design_tags(val: Any) -> Dict[str, List[str]]:
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val.strip():
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def _derive_price_range(price: float) -> str:
    if pd.isna(price) or price <= 0:
        return "unknown"
    if price < 15000:
        return "under_15k"
    if price < 35000:
        return "15k_35k"
    if price < 75000:
        return "35k_75k"
    if price < 150000:
        return "75k_150k"
    return "150k_plus"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def build_feature_matrix(
    period_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame] = None,
    text_pipeline: Optional[Tuple[TfidfVectorizer, TruncatedSVD]] = None,
    fit_text: bool = True,
) -> Tuple[pd.DataFrame, Optional[Tuple[TfidfVectorizer, TruncatedSVD]], Dict[str, Any]]:
    """Build multi-period + structured + 2-layer Description DNA feature DataFrame.

    Does NOT filter out Out-of-Stock SKUs.
    Returns (features_df, fitted_text_pipeline, metadata).
    """
    if period_df.empty:
        return pd.DataFrame(), text_pipeline, {"n_periods": 0}

    work = period_df.copy()
    work["sku_code"] = work["sku_code"].astype(str).str.strip().str.upper()

    for col in ("web_views", "web_clicks", "web_atc", "web_purchased", "web_revenue", "oms_units", "perf_users", "perf_units"):
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    if "avg_price" not in work.columns:
        work["avg_price"] = np.nan
    work["avg_price"] = pd.to_numeric(work["avg_price"], errors="coerce")

    if "period_start" not in work.columns:
        work["period_start"] = "single"
    periods_sorted = sorted(work["period_start"].dropna().astype(str).unique())
    n_periods = len(periods_sorted)

    work["_sold_any"] = ((work["perf_units"] >= 1) | (work["oms_units"] >= 1)).astype(int)

    def _first_valid(s: pd.Series) -> Optional[str]:
        for v in s:
            if pd.notna(v) and str(v).strip() not in {"", "nan", "None"}:
                return str(v).strip()
        return None

    agg = work.groupby("sku_code", as_index=False).agg(
        product_name=("product_name", _first_valid) if "product_name" in work.columns else ("sku_code", lambda _: None),
        period_category=("category", _first_valid) if "category" in work.columns else ("sku_code", lambda _: None),
        total_views=("web_views", "sum"),
        total_clicks=("web_clicks", "sum"),
        total_atc=("web_atc", "sum"),
        total_web_purchased=("web_purchased", "sum"),
        total_perf_users=("perf_users", "sum"),
        perf_units=("perf_units", "sum"),
        oms_units=("oms_units", "sum"),
        avg_price=("avg_price", "mean"),
        repeat_seller_periods=("_sold_any", "sum"),
        period_design_tags=("design_tags", _first_valid) if "design_tags" in work.columns else ("sku_code", lambda _: None),
    )

    agg["log_views"] = np.log1p(agg["total_views"])
    global_atc_mean = float(agg["total_atc"].sum() / max(1.0, agg["total_views"].sum()))
    m_prior = 50.0
    agg["smoothed_atc_rate"] = (agg["total_atc"] + m_prior * global_atc_mean) / (agg["total_views"] + m_prior)
    agg["cart_abandon_rate"] = np.where(
        agg["total_atc"] > 0,
        np.maximum(0.0, (agg["total_atc"] - agg["total_web_purchased"]) / agg["total_atc"]),
        0.0,
    )
    agg["click_to_view_ratio"] = np.where(
        agg["total_views"] > 0,
        agg["total_clicks"] / agg["total_views"],
        0.0,
    )

    # Multi-period trend_3m
    if n_periods >= 2:
        latest_period = periods_sorted[-1]
        latest_views = (
            work[work["period_start"].astype(str) == latest_period]
            .groupby("sku_code")["web_views"]
            .sum()
        )
        prior_views = (
            work[work["period_start"].astype(str) != latest_period]
            .groupby("sku_code")["web_views"]
            .mean()
        )
        agg["_latest_v"] = agg["sku_code"].map(latest_views).fillna(0.0)
        agg["_prior_v"] = agg["sku_code"].map(prior_views).fillna(0.0)
        agg["trend_3m"] = np.where(
            agg["_prior_v"] > 0,
            np.clip((agg["_latest_v"] / agg["_prior_v"]) - 1.0, -1.0, 5.0),
            0.0,
        )
        agg = agg.drop(columns=["_latest_v", "_prior_v"])
    else:
        agg["trend_3m"] = 0.0

    # Merge Structured Master Attributes
    if master_df is not None and not master_df.empty:
        m = master_df.copy()
        m["sku_code"] = m["sku_code"].astype(str).str.strip().str.upper()
        m = m.drop_duplicates(subset=["sku_code"], keep="first")
        agg = agg.merge(m, on="sku_code", how="left", suffixes=("", "_master"))
    else:
        agg["master_title"] = None

    # Resolve unified category & text fields
    if "category" not in agg.columns:
        agg["category"] = agg["period_category"]
    else:
        agg["category"] = agg["category"].fillna(agg["period_category"])
    agg["category"] = agg["category"].fillna("unknown").astype(str).str.strip().replace("", "unknown")

    # Price percentile within category
    agg["avg_price"] = agg["avg_price"].fillna(agg.groupby("category")["avg_price"].transform("median"))
    agg["avg_price"] = agg["avg_price"].fillna(agg["avg_price"].median()).fillna(0.0)
    agg["price_pct_in_category"] = agg.groupby("category")["avg_price"].rank(pct=True).fillna(0.5)

    # Map/fill structured categorical features
    if "gold_karatage" not in agg.columns and "karatage" in agg.columns:
        agg["gold_karatage"] = agg["karatage"]
    if "jewellery_type" not in agg.columns and "product_type" in agg.columns:
        agg["jewellery_type"] = agg["product_type"]
    if "metal" not in agg.columns and "metal_color" in agg.columns:
        agg["metal"] = agg["metal_color"]
    if "price_range" not in agg.columns:
        agg["price_range"] = agg["avg_price"].map(_derive_price_range)

    for col in STRUCTURED_CAT_COLS:
        if col not in agg.columns:
            agg[col] = "unknown"
        agg[col] = (
            agg[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
        )
        agg[col] = pd.Categorical(agg[col])

    # Layer 1: Multi-hot binary flags for all 5 axes in design_tags
    tag_source_col = "design_tags" if "design_tags" in agg.columns else "period_design_tags"
    parsed_tags = [
        _parse_design_tags(v if pd.notna(v) else p)
        for v, p in zip(agg[tag_source_col], agg["period_design_tags"])
    ]

    dna_cols: Dict[str, List[int]] = {}
    for axis, terms in DESIGN_AXES.items():
        for term in terms:
            col_name = f"dna_{axis}_{_slug(term)}"
            dna_cols[col_name] = [1 if term in tags.get(axis, []) else 0 for tags in parsed_tags]
    dna_df = pd.DataFrame(dna_cols, index=agg.index)
    agg = pd.concat([agg, dna_df], axis=1)

    # Layer 2: TF-IDF + TruncatedSVD on combined text
    combined_text = (
        agg["product_name"].fillna("").astype(str)
        + " "
        + agg.get("master_title", pd.Series("", index=agg.index)).fillna("").astype(str)
        + " "
        + agg.get("short_description", pd.Series("", index=agg.index)).fillna("").astype(str)
        + " "
        + agg.get("long_description", pd.Series("", index=agg.index)).fillna("").astype(str)
    ).str.strip()

    if fit_text or text_pipeline is None:
        min_df = 5 if len(agg) >= 50 else 1
        vec = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 2),
            min_df=min_df,
            stop_words="english",
        )
        tfidf_mat = vec.fit_transform(combined_text)
        n_comp = min(15, max(1, tfidf_mat.shape[1] - 1))
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd_mat = svd.fit_transform(tfidf_mat) if tfidf_mat.shape[1] > 1 else np.zeros((len(agg), 1))
        text_pipeline = (vec, svd)
    else:
        vec, svd = text_pipeline
        tfidf_mat = vec.transform(combined_text)
        svd_mat = svd.transform(tfidf_mat)

    svd_cols = {f"svd_{i}": svd_mat[:, i] for i in range(svd_mat.shape[1])}
    for i in range(svd_mat.shape[1], 15):
        svd_cols[f"svd_{i}"] = 0.0
    svd_df = pd.DataFrame(svd_cols, index=agg.index)
    agg = pd.concat([agg, svd_df], axis=1)

    meta = {
        "n_periods": n_periods,
        "global_mean_atc": global_atc_mean,
        "dna_columns": list(dna_cols.keys()),
        "svd_columns": [f"svd_{i}" for i in range(15)],
    }
    return agg, text_pipeline, meta

