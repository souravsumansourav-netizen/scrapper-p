import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from intelligence.etl import (
    ETLStats,
    prepare_master_attribute_records,
    prepare_period_metrics_records,
    run_etl_pipeline,
)

TMP_DIR = Path(__file__).parent / "tmp"


def _compute_top_tags_by_weighted_atc(period_df: pd.DataFrame, top_n: int = 10) -> List[Dict[str, Any]]:
    """Compute top N design/aesthetic/wearability tags by weighted ATC%."""
    if period_df.empty:
        return []

    tag_atc: Dict[str, float] = defaultdict(float)
    tag_views: Dict[str, float] = defaultdict(float)
    tag_skus: Dict[str, int] = defaultdict(int)

    for _, row in period_df.iterrows():
        views = float(row.get("web_views") or 0.0)
        atc = float(row.get("web_atc") or 0.0)
        dna = row.get("design_tags") or {}
        if isinstance(dna, str):
            dna = json.loads(dna)
        for axis in ("motif", "aesthetic", "wearability"):
            for tag in dna.get(axis, []):
                key = f"{axis}:{tag}"
                tag_atc[key] += atc
                tag_views[key] += views
                tag_skus[key] += 1

    rows = []
    for key, views in tag_views.items():
        if views <= 0:
            continue
        atc = tag_atc[key]
        rows.append(
            {
                "tag": key,
                "skus": tag_skus[key],
                "views": int(views),
                "atc": int(atc),
                "weighted_atc_pct": round((atc / views) * 100.0, 2),
            }
        )
    rows.sort(key=lambda r: (r["weighted_atc_pct"], r["views"]), reverse=True)
    return rows[:top_n]


def _print_audit_summary(period_df: pd.DataFrame, stats: ETLStats) -> None:
    print("\n=== SKU Intelligence ETL Audit Summary ===")
    print(f"Web CSV    : {stats.web_total} rows | {stats.web_excluded} excluded | {stats.web_skus} jewellery SKUs")
    print(f"OMS CSV    : {stats.oms_total} rows | {stats.oms_excluded} excluded | {stats.oms_skus} jewellery SKUs")
    print(
        f"Perf CSV   : {stats.perf_total} rows | {stats.perf_excluded} excluded | "
        f"overall join {stats.perf_join_rate:.2f}% ({stats.perf_matched_rows}/{stats.perf_matched_rows + stats.perf_unmatched_rows}) | "
        f"purchased-row join {stats.perf_purchased_join_rate:.2f}% "
        f"({stats.perf_purchased_matched_rows}/{stats.perf_purchased_matched_rows + stats.perf_purchased_unmatched_rows})"
    )
    print(f"Master CSV : {stats.master_total} rows | {stats.master_excluded} excluded | {stats.master_skus} jewellery SKUs")
    pcts = stats.master_match_pct()
    print(
        f"Period SKUs: {stats.period_skus} | Master Match -> "
        f"exact: {pcts['exact']}% ({stats.match_exact}), "
        f"size_13: {pcts['size_13']}% ({stats.match_size_13}), "
        f"title: {pcts['title']}% ({stats.match_title}), "
        f"unmatched: {pcts['unmatched']}% ({stats.match_unmatched})"
    )

    top_tags = _compute_top_tags_by_weighted_atc(period_df, top_n=10)
    print("\nTop 10 Design / Aesthetic / Wearability Tags by Weighted ATC%:")
    if not top_tags:
        print("  (no web views available)")
    else:
        for idx, item in enumerate(top_tags, start=1):
            print(
                f"  {idx:2d}. {item['tag']:<26} "
                f"ATC%={item['weighted_atc_pct']:5.2f}% "
                f"(skus={item['skus']}, atc={item['atc']}, views={item['views']})"
            )


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    for rec in records:
        for k, v in list(rec.items()):
            if isinstance(v, float) and pd.isna(v):
                rec[k] = None
    return records


def _push_to_supabase(period_df: pd.DataFrame, master_attrs_df: pd.DataFrame) -> None:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for --push-supabase.")

    client = create_client(url, key)
    batch_size = 500

    master_records = prepare_master_attribute_records(master_attrs_df)
    for i in range(0, len(master_records), batch_size):
        chunk = master_records[i : i + batch_size]
        client.table("sku_master_attributes").upsert(chunk, on_conflict="sku_code").execute()
    print(f"Upserted {len(master_records)} rows into sku_master_attributes.")

    period_records = prepare_period_metrics_records(period_df)
    for i in range(0, len(period_records), batch_size):
        chunk = period_records[i : i + batch_size]
        client.table("sku_period_metrics").upsert(
            chunk,
            on_conflict="sku_code,period_start,period_end",
        ).execute()
    print(f"Upserted {len(period_records)} rows into sku_period_metrics.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SKU Intelligence multi-source ETL.")
    parser.add_argument("--web", type=str, default=None, help="Path to Web CSV (web-data-*.csv)")
    parser.add_argument("--oms", type=str, default=None, help="Path to OMS CSV")
    parser.add_argument("--perf", type=str, default=None, help="Path to GA4 Perf CSV")
    parser.add_argument("--master", type=str, default=None, help="Path to Master Enrichment CSV")
    parser.add_argument("--period-start", required=True, help="Period start date (YYYY-MM-DD)")
    parser.add_argument("--period-end", required=True, help="Period end date (YYYY-MM-DD)")
    parser.add_argument("--granularity", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--dry-run", action="store_true", help="Print audit summary and save preview to intelligence/tmp/")
    parser.add_argument("--push-supabase", action="store_true", help="Batch-upsert outputs to Supabase.")
    args = parser.parse_args()

    if not any([args.web, args.oms, args.perf, args.master]):
        print("At least one input CSV (--web, --oms, --perf, --master) is required.", file=sys.stderr)
        sys.exit(1)

    period_df, master_attrs_df, stats = run_etl_pipeline(
        web_path=args.web,
        oms_path=args.oms,
        perf_path=args.perf,
        master_path=args.master,
        period_start=args.period_start,
        period_end=args.period_end,
        granularity=args.granularity,
    )

    _print_audit_summary(period_df, stats)

    if args.dry_run:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        period_csv = TMP_DIR / f"sku_period_metrics_{args.period_start}_{args.granularity}.csv"
        master_csv = TMP_DIR / f"sku_master_attributes_{args.period_start}_{args.granularity}.csv"

        period_save = period_df.copy()
        period_save["design_tags"] = period_save["design_tags"].map(json.dumps)
        period_save.to_csv(period_csv, index=False)

        master_save = master_attrs_df.copy()
        if "design_tags" in master_save.columns:
            master_save["design_tags"] = master_save["design_tags"].map(json.dumps)
        master_save.to_csv(master_csv, index=False)

        try:
            period_parquet = TMP_DIR / f"sku_period_metrics_{args.period_start}_{args.granularity}.parquet"
            period_save.to_parquet(period_parquet, index=False)
            print(f"\nSaved dry-run previews to {TMP_DIR} (CSV + Parquet).")
        except Exception:
            print(f"\nSaved dry-run previews to {TMP_DIR} (CSV).")

    if args.push_supabase:
        _push_to_supabase(period_df, master_attrs_df)


if __name__ == "__main__":
    main()

