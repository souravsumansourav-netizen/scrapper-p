import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from scraper import extract_product
from scraper.sku_resolver import ResolverBlockedError, resolve_tanishq_url

IST = timezone(timedelta(hours=5, minutes=30))
STALE_RUNNING_MINUTES = 30
RETRY_DELAY_SECONDS = 3.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_noon_ist_utc() -> datetime:
    now_ist = datetime.now(IST)
    noon_ist = now_ist.replace(hour=12, minute=0, second=0, microsecond=0)
    return noon_ist.astimezone(timezone.utc)


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        cleaned = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _get_supabase_client() -> Any:
    from supabase import create_client

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in environment.")
    return create_client(url, key)


def _reset_stale_running(client: Any) -> int:
    """Reset any `running` rows older than 30 minutes back to `queued`."""
    cutoff_utc = datetime.now(timezone.utc) - timedelta(minutes=STALE_RUNNING_MINUTES)
    running_resp = (
        client.table("sku_market_data")
        .select("*")
        .eq("scrape_status", "running")
        .execute()
    )
    reset_count = 0
    for row in running_resp.data or []:
        code = (row.get("sku_code") or "").strip().upper()
        if not code:
            continue
        ts_dt = (
            _parse_iso(row.get("updated_at"))
            or _parse_iso(row.get("scraped_at"))
            or _parse_iso(row.get("created_at"))
        )
        if ts_dt is not None and ts_dt < cutoff_utc:
            client.table("sku_market_data").update({"scrape_status": "queued"}).eq("sku_code", code).execute()
            reset_count += 1
    return reset_count


def _collect_queued_skus(client: Any) -> List[Dict[str, Optional[str]]]:
    """Return deduplicated list of {sku_code, product_url} where scrape_status = 'queued'."""
    targets: Dict[str, Optional[str]] = {}

    queued_resp = (
        client.table("sku_market_data")
        .select("sku_code, product_url")
        .eq("scrape_status", "queued")
        .execute()
    )
    for row in queued_resp.data or []:
        code = (row.get("sku_code") or "").strip().upper()
        if code and code not in targets:
            targets[code] = row.get("product_url")

    return [{"sku_code": code, "product_url": url} for code, url in targets.items()]


def _collect_daily_skus(client: Any) -> List[Dict[str, Optional[str]]]:
    """Return deduplicated untriaged SKUs whose scraped_at is null or older than today 12:00 PM IST."""
    cutoff_utc = _today_noon_ist_utc()

    untriaged_resp = client.table("skus").select("sku_code").eq("triaged", False).execute()
    untriaged_codes: List[str] = []
    seen = set()
    for row in untriaged_resp.data or []:
        code = (row.get("sku_code") or "").strip().upper()
        if code and code not in seen:
            seen.add(code)
            untriaged_codes.append(code)

    if not untriaged_codes:
        return []

    market_resp = (
        client.table("sku_market_data")
        .select("sku_code, product_url, scraped_at")
        .in_("sku_code", untriaged_codes)
        .execute()
    )
    market_by_code = {
        (r.get("sku_code") or "").strip().upper(): r
        for r in (market_resp.data or [])
        if r.get("sku_code")
    }

    targets: List[Dict[str, Optional[str]]] = []
    for code in untriaged_codes:
        market_row = market_by_code.get(code)
        if market_row is None:
            targets.append({"sku_code": code, "product_url": None})
            continue
        scraped_dt = _parse_iso(market_row.get("scraped_at"))
        if scraped_dt is None or scraped_dt < cutoff_utc:
            targets.append({"sku_code": code, "product_url": market_row.get("product_url")})

    return targets


def _format_scrape_error(extracted: Dict[str, Any]) -> str:
    """Return a specific, diagnostic error reason for failed extractions."""
    raw_err = (extracted.get("error") or "").strip()
    if raw_err:
        lower = raw_err.lower()
        if "http 403" in lower:
            return f"HTTP 403 blocked ({raw_err})"
        if "http 429" in lower:
            return f"HTTP 429 rate-limited ({raw_err})"
        if "timeout" in lower or "timed out" in lower:
            return f"Playwright timeout ({raw_err})"
        return raw_err

    if extracted.get("price") is None:
        return "No price found on page"
    return "No stock or price data extracted"


def _attempt_scrape(sku_code: str, cached_url: Optional[str]) -> Dict[str, Any]:
    """Run one attempt of SKU URL resolution + product extraction."""
    try:
        product_url = cached_url or resolve_tanishq_url(sku_code)
    except ResolverBlockedError as exc:
        return {
            "sku_code": sku_code,
            "product_url": None,
            "price": None,
            "currency": None,
            "stock_status": "unknown",
            "scrape_status": "error",
            "scrape_error": str(exc),
            "scraped_at": _utc_now_iso(),
        }

    if not product_url:
        return {
            "sku_code": sku_code,
            "product_url": None,
            "price": None,
            "currency": None,
            "stock_status": "not_found",
            "scrape_status": "done",
            "scrape_error": None,
            "scraped_at": _utc_now_iso(),
        }

    extracted = extract_product(product_url)
    err = extracted.get("error")
    has_core = extracted.get("price") is not None or extracted.get("stock_status") != "unknown"
    is_done = has_core and not err
    return {
        "sku_code": sku_code,
        "product_url": product_url,
        "price": extracted.get("price"),
        "currency": extracted.get("currency"),
        "stock_status": extracted.get("stock_status", "unknown"),
        "scrape_status": "done" if is_done else "error",
        "scrape_error": None if is_done else _format_scrape_error(extracted),
        "scraped_at": _utc_now_iso(),
    }


def _has_existing_good_data(client: Any, sku_code: str) -> bool:
    """Return True if sku_market_data already has a non-null price or resolved stock_status."""
    resp = (
        client.table("sku_market_data")
        .select("price, stock_status")
        .eq("sku_code", sku_code)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return False
    row = rows[0]
    return row.get("price") is not None or row.get("stock_status") not in (None, "unknown")


def _process_sku(
    sku_code: str,
    cached_url: Optional[str],
    client: Optional[Any],
    dry_run: bool,
) -> Dict[str, Any]:
    if not dry_run and client is not None:
        client.table("sku_market_data").upsert(
            {"sku_code": sku_code, "scrape_status": "running"},
            on_conflict="sku_code",
        ).execute()

    payload = _attempt_scrape(sku_code, cached_url)

    # Automatic 1x retry after 3s on transient error
    if payload["scrape_status"] == "error":
        time.sleep(RETRY_DELAY_SECONDS)
        payload = _attempt_scrape(sku_code, cached_url)

    if not dry_run and client is not None:
        if payload["scrape_status"] == "error" and _has_existing_good_data(client, sku_code):
            # Preserve existing price, currency, stock_status, product_url, and scraped_at.
            client.table("sku_market_data").update(
                {
                    "scrape_status": "error",
                    "scrape_error": payload["scrape_error"],
                }
            ).eq("sku_code", sku_code).execute()
        else:
            client.table("sku_market_data").upsert(payload, on_conflict="sku_code").execute()
            if payload["scrape_status"] == "done":
                _record_snapshot(client, payload)

    return payload


def _record_snapshot(client: Any, payload: Dict[str, Any]) -> None:
    """Append price/stock to sku_market_snapshots so history builds up for the intelligence model.

    Never fails the scrape: if the table is missing (migration 0020 not applied) we just log.
    """
    try:
        client.table("sku_market_snapshots").insert(
            {
                "sku_code": payload["sku_code"],
                "scraped_at": payload["scraped_at"],
                "price": payload.get("price"),
                "currency": payload.get("currency"),
                "stock_status": payload.get("stock_status"),
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001
        print(f"  (snapshot not recorded for {payload['sku_code']}: {exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Tanishq product market data with Supabase.")
    parser.add_argument(
        "--mode",
        choices=["queued", "daily"],
        default="queued",
        help="Sync mode: 'queued' (only scrape_status='queued') or 'daily' (stale untriaged SKUs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run SKU -> URL -> extract_product locally without reading/writing Supabase.",
    )
    parser.add_argument(
        "--skus",
        type=str,
        default="",
        help="Comma-separated SKU codes (used with --dry-run).",
    )
    args = parser.parse_args()

    if args.dry_run:
        raw_skus = [s.strip().upper() for s in args.skus.split(",") if s.strip()]
        if not raw_skus:
            print("Provide --skus when using --dry-run (e.g. --skus 501002SFSAAA2A,50D3PTSBGAAA09)", file=sys.stderr)
            sys.exit(1)
        deduped = list(dict.fromkeys(raw_skus))
        results = []
        for idx, code in enumerate(deduped):
            if idx > 0:
                time.sleep(random.uniform(1.5, 3.0))
            results.append(_process_sku(code, None, None, dry_run=True))
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    client = _get_supabase_client()
    reset_count = _reset_stale_running(client)
    if reset_count:
        print(f"Reset {reset_count} stale 'running' SKU(s) back to 'queued'.")

    targets = _collect_queued_skus(client) if args.mode == "queued" else _collect_daily_skus(client)
    print(f"[mode={args.mode}] Found {len(targets)} unique SKU(s) to process.")

    for idx, item in enumerate(targets, start=1):
        if idx > 1:
            time.sleep(random.uniform(1.5, 3.0))
        code = item["sku_code"]
        print(f"[{idx}/{len(targets)}] Processing {code}...")
        res = _process_sku(code, item.get("product_url"), client, dry_run=False)
        print(
            f"  -> status={res['scrape_status']} price={res['price']} "
            f"stock={res['stock_status']} url={res['product_url']}"
        )


if __name__ == "__main__":
    main()
