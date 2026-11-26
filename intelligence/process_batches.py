import argparse
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError

from intelligence.etl import run_etl_pipeline


def _get_supabase_client() -> Any:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
    return create_client(url, key)


def _normalize_r2_endpoint(endpoint: str, bucket: str) -> str:
    cleaned = endpoint.strip().rstrip("/")
    suffix = f"/{bucket.strip()}"
    if cleaned.endswith(suffix):
        cleaned = cleaned[: -len(suffix)]
    return cleaned


def _get_r2_client() -> Tuple[Any, str]:
    endpoint = os.environ.get("R2_ENDPOINT")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME")
    if not all([endpoint, access_key, secret_key, bucket]):
        raise RuntimeError(
            "R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_BUCKET_NAME must be set."
        )
    bucket_name = str(bucket).strip()
    normalized_endpoint = _normalize_r2_endpoint(str(endpoint), bucket_name)
    s3 = boto3.client(
        "s3",
        endpoint_url=normalized_endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    return s3, bucket_name


def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    for rec in records:
        for k, v in list(rec.items()):
            if isinstance(v, float) and pd.isna(v):
                rec[k] = None
    return records


def _batch_source_type(batch: Dict[str, Any]) -> str:
    raw = (
        batch.get("source_type")
        or batch.get("file_type")
        or batch.get("source")
        or ""
    )
    return str(raw).strip().lower()


def _fetch_pending_batches(client: Any, batch_id: Optional[str]) -> List[Dict[str, Any]]:
    retryable_statuses = ["uploaded", "error"]
    if batch_id:
        target_resp = (
            client.table("intel_upload_batches")
            .select("*")
            .eq("id", batch_id)
            .limit(1)
            .execute()
        )
        rows = target_resp.data or []
        if not rows:
            return []
        target = rows[0]
        target_allowed = target.get("status") in ("uploaded", "error", "processing")
        p_start = target.get("period_start")
        p_end = target.get("period_end")
        if p_start and p_end:
            group_resp = (
                client.table("intel_upload_batches")
                .select("*")
                .in_("status", retryable_statuses)
                .eq("period_start", p_start)
                .eq("period_end", p_end)
                .order("created_at", desc=False)
                .execute()
            )
            group_rows = list(group_resp.data or [])
            if target_allowed and all(r.get("id") != target.get("id") for r in group_rows):
                group_rows.append(target)
            return group_rows
        return [target] if target_allowed else []

    resp = (
        client.table("intel_upload_batches")
        .select("*")
        .in_("status", retryable_statuses)
        .order("created_at", desc=False)
        .execute()
    )
    return resp.data or []


def _process_group(
    client: Any,
    s3: Any,
    bucket: str,
    group_key: Tuple[str, str, str],
    batches: List[Dict[str, Any]],
) -> None:
    period_start, period_end, granularity = group_key
    batch_ids = [b["id"] for b in batches]

    for bid in batch_ids:
        client.table("intel_upload_batches").update({"status": "processing"}).eq("id", bid).execute()

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            paths_by_type: Dict[str, str] = {}

            for b in batches:
                src_type = _batch_source_type(b)
                r2_key = b["r2_key"]
                local_file = tmp_path / f"{b['id']}_{Path(r2_key).name}"
                try:
                    s3.download_file(bucket, r2_key, str(local_file))
                except ClientError as ce:
                    endpoint_url = getattr(getattr(s3, "meta", None), "endpoint_url", None)
                    print(
                        f"R2 ClientError downloading bucket={bucket!r} r2_key={r2_key!r} "
                        f"endpoint_url={endpoint_url!r}: {ce}"
                    )
                    raise
                if src_type in {"web", "oms", "perf", "master"}:
                    paths_by_type[src_type] = str(local_file)

            period_df, master_attrs_df, stats = run_etl_pipeline(
                web_path=paths_by_type.get("web"),
                oms_path=paths_by_type.get("oms"),
                perf_path=paths_by_type.get("perf"),
                master_path=paths_by_type.get("master"),
                period_start=period_start,
                period_end=period_end,
                granularity=granularity,
            )

            batch_size = 500
            if not master_attrs_df.empty:
                master_records = _df_to_records(master_attrs_df)
                for i in range(0, len(master_records), batch_size):
                    client.table("sku_master_attributes").upsert(
                        master_records[i : i + batch_size],
                        on_conflict="sku_code",
                    ).execute()

            if not period_df.empty and period_start and period_end:
                period_records = _df_to_records(period_df)
                for i in range(0, len(period_records), batch_size):
                    client.table("sku_period_metrics").upsert(
                        period_records[i : i + batch_size],
                        on_conflict="sku_code,period_start,period_end,granularity",
                    ).execute()

            match_pct = round(100.0 - stats.master_match_pct()["unmatched"], 2) if stats.period_skus > 0 else None
            ingested_map = {
                "web": stats.web_skus,
                "oms": stats.oms_skus,
                "perf": stats.perf_skus,
                "master": stats.master_skus,
            }
            excluded_map = {
                "web": stats.web_excluded,
                "oms": stats.oms_excluded,
                "perf": stats.perf_excluded,
                "master": stats.master_excluded,
            }

            for b in batches:
                src_type = _batch_source_type(b)
                client.table("intel_upload_batches").update(
                    {
                        "status": "done",
                        "rows_ingested": ingested_map.get(src_type, stats.period_skus),
                        "rows_excluded_non_jewellery": excluded_map.get(src_type, 0),
                        "master_match_pct": match_pct,
                        "error_message": None,
                    }
                ).eq("id", b["id"]).execute()

            print(
                f"Processed group ({period_start}..{period_end}, {granularity}): "
                f"{len(batches)} batch(es), period_skus={stats.period_skus}, match_pct={match_pct}%"
            )
    except Exception as exc:
        for bid in batch_ids:
            client.table("intel_upload_batches").update(
                {
                    "status": "error",
                    "error_message": str(exc)[:1000],
                }
            ).eq("id", bid).execute()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Process uploaded R2 CSV batches for SKU Intelligence.")
    parser.add_argument("--batch-id", type=str, default="", help="Optional specific batch ID to process.")
    args = parser.parse_args()

    batch_id = args.batch_id.strip() or None
    client = _get_supabase_client()
    pending = _fetch_pending_batches(client, batch_id)
    if not pending:
        print("No pending uploaded batches found. Exiting cleanly.")
        return

    s3, bucket = _get_r2_client()
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for b in pending:
        key = (
            str(b.get("period_start") or ""),
            str(b.get("period_end") or ""),
            str(b.get("granularity") or "monthly"),
        )
        grouped[key].append(b)

    for group_key, group_batches in grouped.items():
        _process_group(client, s3, bucket, group_key, group_batches)


if __name__ == "__main__":
    main()

