import json
import sys

from scraper import extract_product


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python test_url.py <product_url>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    result = extract_product(url)
    tier = result.get("tier") or "none"
    print(f"Succeeded tier: {tier}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

