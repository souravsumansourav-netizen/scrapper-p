import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from intelligence.train_model import train_and_score

TMP_DIR = Path(__file__).parent / "tmp"


def _load_from_local(local_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    base = Path(local_dir)
    period_files = sorted(base.glob("sku_period_metrics*.parquet"))
    if not period_files:
        period_files = sorted(base.glob("sku_period_metrics*.csv"))
    if not period_files:
        raise FileNotFoundError(f"No sku_period_metrics parquet/csv files found in {base}")

    period_dfs = [
        pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        for f in period_files
    ]
    period_df = pd.concat(period_dfs, ignore_index=True)

    master_files = sorted(base.glob("sku_master_attributes*.csv"))
    master_df = pd.concat([pd.read_csv(f, dtype=str) for f in master_files], ignore_index=True) if master_files else None
    return period_df, master_df


def _fetch_all_rows(client: Any, table_name: str, page_size: int = 1000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    while True:
        resp = client.table(table_name).select("*").range(offset, offset + page_size - 1).execute()
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _load_from_supabase() -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for --from-supabase.")

    client = create_client(url, key)
    period_rows = _fetch_all_rows(client, "sku_period_metrics")
    master_rows = _fetch_all_rows(client, "sku_master_attributes")
    if not period_rows:
        raise RuntimeError("sku_period_metrics table is empty in Supabase.")
    return pd.DataFrame(period_rows), (pd.DataFrame(master_rows) if master_rows else None)


def _print_report(scored: pd.DataFrame, metrics: Dict[str, Any]) -> None:
    print("\n=== SKU Intelligence Training & Scoring Report ===")
    print(f"SKUs Scored     : {metrics['n_skus']} (periods={metrics['n_periods']}, val={metrics['validation_mode']})")
    print(f"Model A (Perf)  : ROC-AUC = {metrics['perf_roc_auc']:.4f}")
    print(f"Model B (OMS)   : ROC-AUC = {metrics['oms_roc_auc']:.4f}")
    print(f"Tier Counts     : {metrics['tier_counts']}")

    print("\nTop 10 Proven Winners:")
    winners = (
        scored[scored["tier"] == "Proven Winner"]
        .sort_values(["perf_units", "perf_sell_prob"], ascending=False)
        .head(10)
    )
    if winners.empty:
        print("  (none)")
    else:
        for idx, (_, r) in enumerate(winners.iterrows(), start=1):
            print(
                f"  {idx:2d}. {r['sku_code']} | {str(r['product_name'])[:32]:<32} | "
                f"perf_u={r['perf_units']} oms_u={r['oms_units']} | "
                f"p_perf={r['perf_sell_prob']:.2f} score={r['breakout_score']:.1f}"
            )

    print("\nTop 15 Hidden Gems:")
    gems = (
        scored[scored["tier"] == "Hidden Gem"]
        .sort_values("breakout_score", ascending=False)
        .head(15)
    )
    if gems.empty:
        print("  (none)")
    else:
        for idx, (_, r) in enumerate(gems.iterrows(), start=1):
            print(
                f"  {idx:2d}. {r['sku_code']} | {str(r['product_name'])[:32]:<32} | "
                f"score={r['breakout_score']:5.1f} p_perf={r['perf_sell_prob']:.2f} "
                f"views={r['total_views']} atc%={r['smoothed_atc_rate']*100:.1f}% | "
                f"drivers={r['top_drivers']}"
            )


SKU_INTELLIGENCE_SCORES_COLS = (
    "sku_code",
    "perf_sell_prob",
    "oms_sell_prob",
    "breakout_score",
    "tier",
    "trend_3m",
    "top_drivers",
    "design_tags",
    "feature_snapshot",
    "model_version",
    "scored_at",
)


def _push_scores_to_supabase(scored: pd.DataFrame) -> None:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for --push-supabase.")

    client = create_client(url, key)
    now_iso = datetime.now(timezone.utc).isoformat()
    out = scored.copy()
    out["scored_at"] = now_iso
    allowed = set(SKU_INTELLIGENCE_SCORES_COLS)
    cols = [c for c in SKU_INTELLIGENCE_SCORES_COLS if c in out.columns]
    raw_records = out[cols].where(pd.notnull(out[cols]), None).to_dict(orient="records")

    deduped: Dict[str, Dict[str, Any]] = {}
    for rec in raw_records:
        cleaned = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in rec.items() if k in allowed}
        code = cleaned.get("sku_code")
        if code:
            deduped[str(code)] = cleaned
    records = list(deduped.values())

    batch_size = 500
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        client.table("sku_intelligence_scores").upsert(chunk, on_conflict="sku_code").execute()
    print(f"\nUpserted {len(records)} scored SKUs into sku_intelligence_scores.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual LightGBM models and score SKUs.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--from-local", type=str, help="Directory with local parquet/csv previews")
    source_group.add_argument("--from-supabase", action="store_true", help="Read training data from Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Print report and save intelligence/tmp/scored_skus.csv")
    parser.add_argument("--push-supabase", action="store_true", help="Upsert scores into sku_intelligence_scores")
    args = parser.parse_args()

    if args.from_local:
        period_df, master_df = _load_from_local(args.from_local)
    else:
        period_df, master_df = _load_from_supabase()

    scored, metrics = train_and_score(period_df, master_df)
    _print_report(scored, metrics)

    if args.dry_run:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        out_path = TMP_DIR / "scored_skus.csv"
        save_df = scored.copy()
        save_df["top_drivers"] = save_df["top_drivers"].map(json.dumps)
        save_df.to_csv(out_path, index=False)
        print(f"\nSaved scored SKUs preview to {out_path}")

    if args.push_supabase:
        _push_scores_to_supabase(scored)


if __name__ == "__main__":
    main()

