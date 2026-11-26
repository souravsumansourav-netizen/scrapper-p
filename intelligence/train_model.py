import re
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from intelligence.features import STRUCTURED_CAT_COLS, build_feature_matrix

BASE_BEHAVIORAL_FEATURES = [
    "total_views",
    "log_views",
    "total_clicks",
    "total_atc",
    "smoothed_atc_rate",
    "click_to_view_ratio",
    "avg_price",
    "price_pct_in_category",
    "trend_3m",
]

MULTI_PERIOD_EXTRA_FEATURES = [
    "cart_abandon_rate",
    "repeat_seller_periods",
]


def get_feature_columns(feat_df: pd.DataFrame, is_multi_period: bool) -> List[str]:
    """Return leakage-free feature column list."""
    cols = list(BASE_BEHAVIORAL_FEATURES)
    if is_multi_period:
        cols.extend(MULTI_PERIOD_EXTRA_FEATURES)
    cols.extend([c for c in STRUCTURED_CAT_COLS if c in feat_df.columns])
    cols.extend([c for c in feat_df.columns if c.startswith("dna_")])
    cols.extend([c for c in feat_df.columns if c.startswith("svd_")])
    return cols


def _fit_lgbm_binary(X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        subsample_for_bin=50000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def _cv_oof_and_calibrated(
    X: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
) -> Tuple[np.ndarray, float, lgb.LGBMClassifier]:
    """Run 5-fold Stratified CV, calibrate OOF probabilities, and fit full model for explainability."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    actual_splits = min(n_splits, n_pos, n_neg)

    if actual_splits < 2:
        full_model = _fit_lgbm_binary(X, y)
        probs = full_model.predict_proba(X)[:, 1]
        return probs, 0.5, full_model

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    oof_raw = np.zeros(len(y), dtype=float)

    for train_idx, val_idx in skf.split(X, y):
        clf = _fit_lgbm_binary(X.iloc[train_idx], y[train_idx])
        oof_raw[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]

    auc = float(roc_auc_score(y, oof_raw))
    iso = IsotonicRegression(out_of_bounds="clip")
    oof_cal = iso.fit_transform(oof_raw, y)

    full_model = _fit_lgbm_binary(X, y)
    return oof_cal, round(auc, 4), full_model


def _format_driver(feat_name: str, contrib: float, row: pd.Series) -> str:
    sign = "+" if contrib >= 0 else "-"

    if feat_name == "smoothed_atc_rate":
        pct = float(row.get("smoothed_atc_rate", 0.0)) * 100.0
        qual = "High" if contrib >= 0 else "Low"
        return f"{sign} {qual} Smoothed ATC ({pct:.1f}%)"
    if feat_name in ("total_views", "log_views"):
        views = int(row.get("total_views", 0))
        qual = "Strong" if contrib >= 0 else "Low"
        return f"{sign} {qual} Views ({views})"
    if feat_name == "total_atc":
        atc = int(row.get("total_atc", 0))
        qual = "High" if contrib >= 0 else "Low"
        return f"{sign} {qual} ATC ({atc})"
    if feat_name == "price_pct_in_category":
        pct = int(round(float(row.get("price_pct_in_category", 0.5)) * 100))
        qual = "High" if contrib >= 0 else "Lower"
        return f"{sign} {qual} Price Percentile ({pct}%)"
    if feat_name == "avg_price":
        price = float(row.get("avg_price", 0.0))
        return f"{sign} Price (INR {price:,.0f})"
    if feat_name == "click_to_view_ratio":
        ctr = float(row.get("click_to_view_ratio", 0.0)) * 100.0
        return f"{sign} Click/View ({ctr:.1f}%)"
    if feat_name == "category":
        return f"{sign} Category: {row.get('category', 'unknown')}"
    if feat_name.startswith("dna_"):
        parts = feat_name.split("_", 2)
        axis = parts[1].capitalize() if len(parts) > 1 else "DNA"
        term = parts[2].replace("_", " ").title() if len(parts) > 2 else feat_name
        return f"{sign} {axis}: {term}"
    if feat_name.startswith("svd_"):
        return f"{sign} Description Theme ({feat_name.upper()})"
    if feat_name in STRUCTURED_CAT_COLS:
        label = feat_name.replace("_", " ").title()
        return f"{sign} {label}: {row.get(feat_name, 'unknown')}"

    pretty = re.sub(r"_+", " ", feat_name).title()
    return f"{sign} {pretty}"


def _extract_top_drivers(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    feat_df: pd.DataFrame,
    top_k: int = 4,
) -> List[List[str]]:
    """Use LightGBM native pred_contrib=True to extract top K drivers per SKU."""
    contribs = model.predict(X, pred_contrib=True)
    feat_names = list(X.columns)
    all_drivers: List[List[str]] = []

    for idx in range(len(X)):
        row_contrib = contribs[idx, :-1]  # exclude trailing bias column
        top_indices = np.argsort(np.abs(row_contrib))[::-1][:top_k]
        row_series = feat_df.iloc[idx]
        drivers = [
            _format_driver(feat_names[j], float(row_contrib[j]), row_series)
            for j in top_indices
        ]
        all_drivers.append(drivers)

    return all_drivers


def _assign_tier(row: pd.Series) -> str:
    perf_u = float(row["perf_units"])
    oms_u = float(row["oms_units"])
    p_perf = float(row["perf_sell_prob"])
    b_score = float(row["breakout_score"])

    if perf_u >= 1 or (p_perf >= 0.65 and oms_u >= 2):
        return "Proven Winner"
    if perf_u == 0 and b_score >= 97.0:
        return "Hidden Gem"
    if (perf_u == 0 and b_score >= 85.0) or p_perf >= 0.35:
        return "Watchlist"
    return "Low Potential"


def train_and_score(
    period_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Train dual LightGBM models, compute breakout scores, tiers, and top_drivers."""
    periods = sorted(period_df["period_start"].dropna().astype(str).unique()) if "period_start" in period_df.columns else ["single"]
    is_multi_period = len(periods) >= 2

    if is_multi_period:
        latest_period = periods[-1]
        hist_period_df = period_df[period_df["period_start"].astype(str) != latest_period].copy()
        latest_targets = (
            period_df[period_df["period_start"].astype(str) == latest_period]
            .groupby("sku_code", as_index=False)[["perf_units", "oms_units"]]
            .sum()
        )
        oot_feat, text_pipe, _ = build_feature_matrix(hist_period_df, master_df, fit_text=True)
        oot_merged = oot_feat.merge(
            latest_targets.rename(columns={"perf_units": "_t_perf", "oms_units": "_t_oms"}),
            on="sku_code",
            how="left",
        )
        oot_merged["_t_perf"] = oot_merged["_t_perf"].fillna(0.0)
        oot_merged["_t_oms"] = oot_merged["_t_oms"].fillna(0.0)

        oot_cols = get_feature_columns(oot_merged, is_multi_period=True)
        X_oot = oot_merged[oot_cols]
        y_oot_perf = (oot_merged["_t_perf"] >= 1).astype(int).to_numpy()
        y_oot_oms = (oot_merged["_t_oms"] >= 1).astype(int).to_numpy()

        _, perf_auc, _ = _cv_oof_and_calibrated(X_oot, y_oot_perf)
        _, oms_auc, _ = _cv_oof_and_calibrated(X_oot, y_oot_oms)

        # Refit on all periods to score forward
        feat_df, _, _ = build_feature_matrix(period_df, master_df, text_pipeline=text_pipe, fit_text=False)
        feat_cols = get_feature_columns(feat_df, is_multi_period=True)
        X_all = feat_df[feat_cols]
        y_perf = (feat_df["perf_units"] >= 1).astype(int).to_numpy()
        y_oms = (feat_df["oms_units"] >= 1).astype(int).to_numpy()
        perf_probs, _, perf_model = _cv_oof_and_calibrated(X_all, y_perf)
        oms_probs, _, _ = _cv_oof_and_calibrated(X_all, y_oms)
    else:
        feat_df, _, _ = build_feature_matrix(period_df, master_df, fit_text=True)
        feat_cols = get_feature_columns(feat_df, is_multi_period=False)
        X_all = feat_df[feat_cols]
        y_perf = (feat_df["perf_units"] >= 1).astype(int).to_numpy()
        y_oms = (feat_df["oms_units"] >= 1).astype(int).to_numpy()

        perf_probs, perf_auc, perf_model = _cv_oof_and_calibrated(X_all, y_perf)
        oms_probs, oms_auc, _ = _cv_oof_and_calibrated(X_all, y_oms)

    # Breakout / Potential Top Seller Score (0-100)
    views_pct = feat_df["total_views"].rank(pct=True).to_numpy()
    traffic_headroom = np.clip(1.0 - views_pct, 0.15, 1.0)
    atc_pct = feat_df["smoothed_atc_rate"].rank(pct=True).to_numpy()

    intent_signal = 0.5 * perf_probs + 0.3 * oms_probs + 0.2 * atc_pct
    raw_breakout = 100.0 * intent_signal * (0.4 + 0.6 * traffic_headroom)
    unscaled_mask = feat_df["perf_units"].to_numpy() == 0
    raw_breakout = raw_breakout + np.where(unscaled_mask, 5.0, 0.0)

    # Percentile-rank scale cleanly to 0.00-100.00 across unscaled SKUs (perf_units == 0),
    # and map scaled SKUs onto the exact same 0-100 distribution.
    breakout_score = np.zeros(len(feat_df), dtype=float)
    if unscaled_mask.any():
        unscaled_raw = raw_breakout[unscaled_mask]
        unscaled_pct = pd.Series(unscaled_raw).rank(pct=True).to_numpy() * 100.0
        breakout_score[unscaled_mask] = unscaled_pct

        if (~unscaled_mask).any():
            sorted_u = np.sort(unscaled_raw)
            grid = np.linspace(0.0, 100.0, len(sorted_u))
            breakout_score[~unscaled_mask] = np.interp(raw_breakout[~unscaled_mask], sorted_u, grid)
    else:
        breakout_score = pd.Series(raw_breakout).rank(pct=True).to_numpy() * 100.0

    breakout_score = np.round(np.clip(breakout_score, 0.0, 100.0), 2)

    top_drivers = _extract_top_drivers(perf_model, X_all, feat_df, top_k=4)

    design_tags_col = feat_df.get("design_tags", pd.Series([{} for _ in range(len(feat_df))], index=feat_df.index))
    parsed_design_tags = [
        v if isinstance(v, dict) else {}
        for v in design_tags_col
    ]

    feature_snapshots = [
        {
            "product_name": None if pd.isna(r.get("product_name")) else r.get("product_name"),
            "category": str(r.get("category", "unknown")),
            "avg_price": round(float(r.get("avg_price", 0.0)), 2),
            "total_views": int(r.get("total_views", 0)),
            "total_atc": int(r.get("total_atc", 0)),
            "smoothed_atc_rate": round(float(r.get("smoothed_atc_rate", 0.0)), 4),
            "perf_units": int(r.get("perf_units", 0)),
            "oms_units": int(r.get("oms_units", 0)),
        }
        for _, r in feat_df.iterrows()
    ]

    scored = pd.DataFrame(
        {
            "sku_code": feat_df["sku_code"],
            "product_name": feat_df["product_name"],
            "category": feat_df["category"].astype(str),
            "avg_price": np.round(feat_df["avg_price"], 2),
            "total_views": feat_df["total_views"].astype(int),
            "total_atc": feat_df["total_atc"].astype(int),
            "smoothed_atc_rate": np.round(feat_df["smoothed_atc_rate"], 4),
            "perf_units": feat_df["perf_units"].astype(int),
            "oms_units": feat_df["oms_units"].astype(int),
            "perf_sell_prob": np.round(perf_probs, 4),
            "oms_sell_prob": np.round(oms_probs, 4),
            "breakout_score": breakout_score,
            "trend_3m": np.round(feat_df["trend_3m"], 4),
            "top_drivers": top_drivers,
            "design_tags": parsed_design_tags,
            "feature_snapshot": feature_snapshots,
            "model_version": "lgbm_v1",
        }
    )
    scored["tier"] = scored.apply(_assign_tier, axis=1)

    metrics = {
        "n_skus": len(scored),
        "n_periods": len(periods),
        "validation_mode": "out_of_time" if is_multi_period else "5fold_stratified_cv",
        "perf_roc_auc": perf_auc,
        "oms_roc_auc": oms_auc,
        "tier_counts": scored["tier"].value_counts().to_dict(),
    }
    return scored, metrics

