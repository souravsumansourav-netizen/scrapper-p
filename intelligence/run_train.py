import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from intelligence.train_model import train_and_score

TMP_DIR = Path(__file__).parent / "tmp"


def _load_from_local(local_dir: str, market_csv: Optional[str]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    base = Path(local_dir)
    period_files = sorted(base.glob("sku_period_metrics*.parquet")) or sorted(base.glob("sku_period_metrics*.csv"))
    if not period_files:
        raise FileNotFoundError(f"No sku_period_metrics parquet/csv files found in {base}")
    period_df = pd.concat(
        [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in period_files],
        ignore_index=True,
    )
    master_files = sorted(base.glob("sku_master_attributes*.csv"))
    master_df = pd.concat([pd.read_csv(f, dtype=str) for f in master_files], ignore_index=True) if master_files else None
    market_df = pd.read_csv(market_csv) if market_csv else None
    return period_df, master_df, market_df


def _client() -> Any:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
    return create_client(url, key)


def _fetch_all_rows(client: Any, table_name: str, columns: str = "*", page_size: int = 1000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        batch = client.table(table_name).select(columns).range(offset, offset + page_size - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page_size:
            return rows
        offset += page_size


def _load_from_supabase() -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    client = _client()
    period_rows = _fetch_all_rows(client, "sku_period_metrics")
    if not period_rows:
        raise RuntimeError("sku_period_metrics table is empty in Supabase.")
    master_rows = _fetch_all_rows(client, "sku_master_attributes")
    market_rows = _fetch_all_rows(client, "sku_market_data", "sku_code, price, stock_status")
    return (
        pd.DataFrame(period_rows),
        pd.DataFrame(master_rows) if master_rows else None,
        pd.DataFrame(market_rows) if market_rows else None,
    )


def _print_report(scored: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    print("\n=== SKU Intelligence: Category Winner Model ===")
    print(f"SKUs scored   : {metrics['n_skus']}  (features={metrics['n_features']}, validation={metrics['validation_mode']})")
    print(f"Winners       : {metrics['n_winners']}  (base rate {metrics['base_rate'] * 100:.2f}%)")
    print(f"ROC-AUC       : {metrics['roc_auc']}   Avg precision: {metrics['avg_precision']}")
    print(f"Tier counts   : {metrics['tier_counts']}")

    print("\nPer category: of the 10 SKUs the model ranks highest, how many were real winners")
    print(f"  {'Category':<20}{'SKUs':>7}{'Winners':>9}{'Hits@10':>9}")
    p10 = sorted(metrics["precision_at_10"].items(), key=lambda kv: -kv[1]["winners"])
    for cat, m in p10:
        print(f"  {cat:<20}{m['skus']:>7}{m['winners']:>9}{m['hits_in_top_k']:>6}/{m['k']}")

    tiers = scored.groupby(["category", "tier"]).size().unstack(fill_value=0)
    print("\nTiers by category:")
    print(tiers.sort_values(by=list(tiers.columns)[0], ascending=False).to_string())

    for tier, sort_col, n in (("Proven Winner", "units_sold", 5), ("Hidden Gem", "gem_score", 8), ("Watchlist", "winner_prob", 8)):
        print(f"\nTop {tier}s:")
        sub = scored[scored["tier"] == tier].sort_values(sort_col, ascending=False).head(n)
        for _, r in sub.iterrows():
            print(
                f"  {r['sku_code']:<16}{str(r['category'])[:14]:<15}{str(r['product_name'])[:30]:<31}"
                f"units={r['units_sold']:<5g} views={int(r['views_total']):<6} carts={int(r['atc_total']):<4} "
                f"p={r['winner_prob']:.2f} rank={r['category_rank']}/{r['category_size']} | {r['top_drivers'][:2]}"
            )


SCORE_COLS = (
    "sku_code",
    "category",
    "winner_prob",
    "category_rank",
    "category_size",
    "category_pct",
    "units_sold",
    "gem_score",
    "breakout_score",
    "tier",
    "top_drivers",
    "design_tags",
    "feature_snapshot",
    "model_version",
    "scored_at",
)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _push_scores_to_supabase(scored: pd.DataFrame) -> None:
    client = _client()
    now_iso = datetime.now(timezone.utc).isoformat()
    out = scored.copy()
    out["scored_at"] = now_iso
    out = out.drop_duplicates("sku_code", keep="last")
    records = [
        {c: _jsonable(r[c]) for c in SCORE_COLS}
        for r in out[list(SCORE_COLS)].to_dict(orient="records")
    ]
    # Old-model fields are cleared so the dashboard never mixes model versions.
    for rec in records:
        rec.update({"perf_sell_prob": None, "oms_sell_prob": None, "trend_3m": None})

    for i in range(0, len(records), 500):
        client.table("sku_intelligence_scores").upsert(records[i : i + 500], on_conflict="sku_code").execute()
    # Remove SKUs that are no longer in the scored universe (stale rows from earlier runs).
    client.table("sku_intelligence_scores").delete().lt("scored_at", now_iso).execute()
    print(f"\nUpserted {len(records)} scored SKUs into sku_intelligence_scores and removed stale rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the category-winner model and score SKUs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-local", type=str, help="Directory with ETL dry-run previews")
    source.add_argument("--from-supabase", action="store_true", help="Read training data from Supabase")
    parser.add_argument("--market-csv", type=str, default=None, help="Optional local sku_market_data export (sku_code, price, stock_status)")
    parser.add_argument("--dry-run", action="store_true", help="Print report and save intelligence/tmp/scored_skus.csv")
    parser.add_argument("--push-supabase", action="store_true", help="Upsert scores into sku_intelligence_scores")
    args = parser.parse_args()

    if args.from_local:
        period_df, master_df, market_df = _load_from_local(args.from_local, args.market_csv)
    else:
        period_df, master_df, market_df = _load_from_supabase()

    scored, metrics = train_and_score(period_df, master_df, market_df)
    _print_report(scored, metrics)

    if args.dry_run:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        save_df = scored.copy()
        for col in ("top_drivers", "design_tags", "feature_snapshot"):
            save_df[col] = save_df[col].map(lambda v: json.dumps(v, ensure_ascii=False))
        save_df.to_csv(TMP_DIR / "scored_skus.csv", index=False)
        (TMP_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        print(f"\nSaved {TMP_DIR / 'scored_skus.csv'} and model_metrics.json")

    if args.push_supabase:
        _push_scores_to_supabase(scored)


if __name__ == "__main__":
    main()
