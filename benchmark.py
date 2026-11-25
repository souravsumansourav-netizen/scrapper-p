import json
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from scraper import extract_product

URLS_FILE = Path(__file__).parent / "urls.txt"
RESULTS_FILE = Path(__file__).parent / "benchmark_results.json"


def _truncate(val: Any, max_len: int) -> str:
    text = "-" if val is None else str(val)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        ("Domain", 18),
        ("Title", 34),
        ("Price", 11),
        ("Curr", 5),
        ("Stock Status", 20),
        ("Tier", 7),
        ("Latency", 8),
        ("Error", 22),
    ]
    header_line = " | ".join(h.ljust(w) for h, w in headers)
    sep_line = "-+-".join("-" * w for _, w in headers)
    print(header_line)
    print(sep_line)

    for row in rows:
        domain = (urlparse(row["url"]).hostname or "").replace("www.", "")
        price_str = f"{row['price']:.2f}" if isinstance(row.get("price"), (int, float)) else "-"
        latency_str = f"{row['latency_s']:.2f}s"
        cols = [
            _truncate(domain, 18).ljust(18),
            _truncate(row.get("title"), 34).ljust(34),
            _truncate(price_str, 11).rjust(11),
            _truncate(row.get("currency"), 5).ljust(5),
            _truncate(row.get("stock_status"), 20).ljust(20),
            _truncate(row.get("tier") or "none", 7).ljust(7),
            _truncate(latency_str, 8).rjust(8),
            _truncate(row.get("error"), 22).ljust(22),
        ]
        print(" | ".join(cols))


def main() -> None:
    urls = [
        line.strip()
        for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    results: List[Dict[str, Any]] = []
    for idx, url in enumerate(urls, start=1):
        print(f"[{idx}/{len(urls)}] Extracting: {url}")
        start = time.perf_counter()
        data = extract_product(url)
        latency = round(time.perf_counter() - start, 3)
        data["latency_s"] = latency
        results.append(data)

    print()
    _print_table(results)

    RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(results)} results to {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()

