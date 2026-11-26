"""Category-winner model (Phase 1).

Label: a SKU is a winner when it is in the top 20% of *selling* SKUs in its
category by units sold, and sold at least 2 units. Units sold =
max(OMS units, web + app purchases).

Phase 1 uses one period, so the model learns "which SKUs look like this month's
winners" from traffic, intent, price and design — not purchases. Every SKU is
scored by a model that never saw it (5-fold out-of-fold), then calibrated.
Phase 2 will switch to forecasting next month once several months are loaded.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from intelligence.features import CAT_FEATURES, PRICE_BAND_LABEL, build_feature_frame

MODEL_VERSION = "lgbm_v2_category_winner"

WINNER_TOP_SHARE = 0.20
WINNER_MIN_UNITS = 2.0

GEM_MIN_VIEWS = 50.0  # enough traffic that the add-to-cart rate is not luck
GEM_MIN_ATC = 5.0
GEM_MAX_TRAFFIC_PCT = 0.50  # at or below the category median traffic (among viewed SKUs)
GEM_MIN_ATC_RATE_PCT = 0.90  # top 10% add-to-cart rate in its category
GEM_MIN_CATEGORY_VIEWED = 30  # "below median traffic" means little in a tiny category

WATCHLIST_MIN_PROB = 0.30


def label_winners(df: pd.DataFrame) -> pd.Series:
    """Top 20% of selling SKUs in each category by units, with at least 2 units."""
    units = df["units_sold"].astype(float)
    sellers = units >= 1
    thresholds = df[sellers].groupby("category", observed=True)["units_sold"].quantile(1.0 - WINNER_TOP_SHARE)
    cutoff = df["category"].astype(str).map(thresholds.rename(index=str)).fillna(np.inf)
    return (units >= np.maximum(cutoff, WINNER_MIN_UNITS)).astype(int)


def _fit_lgbm(X: pd.DataFrame, y: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y, categorical_feature=[c for c in CAT_FEATURES if c in X.columns])
    return model


def _oof_calibrated(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray, lgb.LGBMClassifier]:
    """Out-of-fold raw + isotonic-calibrated probabilities, and a full-data model for explanations."""
    splits = min(n_splits, int(y.sum()), int(len(y) - y.sum()))
    full_model = _fit_lgbm(X, y)
    if splits < 2:
        raw = full_model.predict_proba(X)[:, 1]
        return raw, raw, full_model
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits=splits, shuffle=True, random_state=42).split(X, y):
        oof[va] = _fit_lgbm(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    calibrated = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit_transform(oof, y)
    return oof, calibrated, full_model


# ---------------------------------------------------------------- drivers
_DRIVER_LABELS = {
    "log_views_total": "Views",
    "log_views_web": "Web views",
    "log_views_app": "App views",
    "log_clicks_total": "Listing clicks",
    "log_atc_total": "Add-to-carts",
}


def _pct(v: Any) -> str:
    return "—" if v is None or pd.isna(v) else f"{float(v) * 100:.1f}%"


def _top_share(v: Any) -> str:
    if v is None or pd.isna(v):
        return "no data"
    return f"top {max(1, round((1 - float(v)) * 100))}%"


def _driver_text(feat: str, row: pd.Series) -> str:
    if feat in _DRIVER_LABELS:
        raw = {"log_views_total": "views_total", "log_views_web": "web_views", "log_views_app": "app_views",
               "log_clicks_total": "clicks_total", "log_atc_total": "atc_total"}[feat]
        return f"{_DRIVER_LABELS[feat]} {int(round(float(row.get(raw, 0) or 0))):,}"
    if feat == "atc_rate_smoothed":
        return f"Add-to-cart rate {_pct(row.get(feat))}"
    if feat == "atc_rate_pct_cat":
        return f"Add-to-cart rate {_top_share(row.get(feat))} of category"
    if feat == "views_pct_cat":
        return f"Traffic {_top_share(row.get(feat))} of category"
    if feat == "app_view_share":
        return f"App share of views {_pct(row.get(feat))}"
    if feat == "click_through":
        return f"Click-through {_pct(row.get(feat))}"
    if feat == "price_band_ord":
        v = row.get(feat)
        return f"Price band {PRICE_BAND_LABEL.get(int(v), '?') if pd.notna(v) else 'unknown'}"
    if feat == "price_pct_cat":
        v = row.get(feat)
        if pd.isna(v):
            return "Price unknown"
        return "Lower-priced for its category" if float(v) < 0.5 else "Higher-priced for its category"
    if feat.startswith("dna_"):
        _, axis, term = feat.split("_", 2)
        return f"{axis.capitalize()}: {term.replace('_', ' ')}"
    if feat in CAT_FEATURES:
        return f"{feat.replace('_', ' ').capitalize()}: {row.get(feat)}"
    return re.sub(r"_+", " ", feat).capitalize()


def _top_drivers(model: lgb.LGBMClassifier, X: pd.DataFrame, frame: pd.DataFrame, k: int = 4) -> List[List[str]]:
    contribs = model.predict(X, pred_contrib=True)[:, :-1]
    names = list(X.columns)
    out = []
    for i in range(len(X)):
        row = frame.iloc[i]
        picked: List[str] = []
        for j in np.argsort(np.abs(contribs[i]))[::-1]:
            if len(picked) == k or abs(contribs[i, j]) < 1e-6:
                break
            if names[j] in CAT_FEATURES and str(row.get(names[j])) in {"unknown", "no"}:
                continue  # "Occasion: unknown" explains nothing to a merchandiser
            picked.append(f"{'↑' if contribs[i, j] >= 0 else '↓'} {_driver_text(names[j], row)}")
        out.append(picked)
    return out


# ---------------------------------------------------------------- tiers
def assign_tiers(df: pd.DataFrame) -> pd.Series:
    viewed_in_cat = df.assign(_v=df["views_total"] > 0).groupby("category", observed=True)["_v"].transform("sum")
    gem = (
        (df["is_winner"] == 0)
        & (viewed_in_cat >= GEM_MIN_CATEGORY_VIEWED)
        & (df["views_total"] >= GEM_MIN_VIEWS)
        & (df["atc_total"] >= GEM_MIN_ATC)
        & (df["views_pct_cat_viewed"] <= GEM_MAX_TRAFFIC_PCT)
        & (df["atc_rate_pct_cat"] >= GEM_MIN_ATC_RATE_PCT)
    )
    tier = np.where(
        df["is_winner"] == 1, "Proven Winner",
        np.where(gem, "Hidden Gem",
                 np.where(df["winner_prob"] >= WATCHLIST_MIN_PROB, "Watchlist", "Low Potential")),
    )
    return pd.Series(tier, index=df.index)


def _precision_at_k(df: pd.DataFrame, k: int = 10) -> Dict[str, Dict[str, Any]]:
    out = {}
    for cat, grp in df.groupby("category", observed=True):
        n_win = int(grp["is_winner"].sum())
        if n_win == 0:
            continue
        top = grp.nlargest(min(k, len(grp)), "winner_prob")
        out[str(cat)] = {
            "skus": int(len(grp)),
            "winners": n_win,
            "hits_in_top_k": int(top["is_winner"].sum()),
            "k": int(len(top)),
        }
    return out


def train_and_score(
    period_df: pd.DataFrame,
    master_df: Optional[pd.DataFrame] = None,
    market_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frame, feat_cols = build_feature_frame(period_df, master_df, market_df)
    frame = frame.reset_index(drop=True)
    frame["is_winner"] = label_winners(frame)
    y = frame["is_winner"].to_numpy()
    X = frame[feat_cols]

    oof_raw, prob, model = _oof_calibrated(X, y)
    frame["winner_prob"] = prob
    frame["_rank_score"] = oof_raw  # finer-grained than isotonic steps, for ranking

    frame["category_size"] = frame.groupby("category", observed=True)["sku_code"].transform("count")
    frame["category_rank"] = frame.groupby("category", observed=True)["_rank_score"].rank(ascending=False, method="first")
    frame["category_pct"] = frame.groupby("category", observed=True)["_rank_score"].rank(pct=True) * 100.0
    gem_score = frame["atc_rate_pct_cat"] * 100.0
    frame["gem_score"] = gem_score.where(frame["views_total"] > 0)
    frame["tier"] = assign_tiers(frame)
    drivers = _top_drivers(model, X, frame)
    # Hidden Gems are chosen by a rule, not the winner model, so lead with the rule's reason.
    for i in np.flatnonzero(frame["tier"].to_numpy() == "Hidden Gem"):
        r = frame.iloc[i]
        reason = (
            f"★ Add-to-cart rate {_top_share(r['atc_rate_pct_cat'])} of category "
            f"on below-median traffic ({int(r['views_total']):,} views)"
        )
        drivers[i] = [reason] + drivers[i][:3]

    snap_cols = ["product_name", "views_total", "web_views", "app_views", "atc_total", "atc_rate_smoothed",
                 "units_sold", "oms_units", "price_band", "live_price"]
    snapshots = []
    for _, r in frame.iterrows():
        snap = {}
        for c in snap_cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            snap[c] = round(float(v), 4) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
        snapshots.append(snap)

    scored = pd.DataFrame({
        "sku_code": frame["sku_code"],
        "product_name": frame["product_name"],
        "category": frame["category"].astype(str),
        "winner_prob": np.round(frame["winner_prob"], 4),
        "category_rank": frame["category_rank"].astype(int),
        "category_size": frame["category_size"].astype(int),
        "category_pct": np.round(frame["category_pct"], 2),
        "breakout_score": np.round(frame["category_pct"], 2),  # kept for older dashboard builds
        "units_sold": np.round(frame["units_sold"], 2),
        "gem_score": np.round(frame["gem_score"], 2),
        "tier": frame["tier"],
        "is_winner": frame["is_winner"],
        "views_total": np.round(frame["views_total"], 0),
        "atc_total": np.round(frame["atc_total"], 0),
        "atc_rate_smoothed": np.round(frame["atc_rate_smoothed"], 4),
        "stock_status": frame["stock_status"],
        "top_drivers": drivers,
        "design_tags": frame["design_tags"],
        "feature_snapshot": snapshots,
        "model_version": MODEL_VERSION,
    })

    metrics = {
        "n_skus": int(len(scored)),
        "n_features": len(feat_cols),
        "n_winners": int(y.sum()),
        "base_rate": round(float(y.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y, oof_raw)), 4) if 0 < y.sum() < len(y) else None,
        "avg_precision": round(float(average_precision_score(y, oof_raw)), 4) if y.sum() else None,
        "tier_counts": scored["tier"].value_counts().to_dict(),
        "precision_at_10": _precision_at_k(frame.assign(winner_prob=oof_raw)),
        "validation_mode": "5fold_out_of_fold_same_period",
    }
    return scored, metrics
