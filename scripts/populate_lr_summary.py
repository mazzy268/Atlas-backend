"""
Populate data/land_registry_summary.csv from HM Land Registry Price Paid bulk CSVs.

Downloads annual files for YEARS (2020-2024), streams each line, aggregates
average_price and transaction_count by (postcode_district, year), then writes
data/land_registry_summary.csv.

Usage:
    python scripts/populate_lr_summary.py

No external dependencies beyond stdlib + requests.
"""

import collections
import csv
import io
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

BASE_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-{year}.csv"
YEARS = [2020, 2021, 2022, 2023, 2024]
CHUNK = 1 << 20  # 1 MB read chunks

OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "land_registry_summary.csv")


def _district(postcode: str) -> str:
    """Extract outward code from a postcode, e.g. 'SW1A 2AA' -> 'SW1A'."""
    pc = postcode.strip().upper()
    space = pc.find(" ")
    return pc[:space] if space != -1 else pc[:-3]


def fetch_year(year: int) -> dict:
    """Stream one annual file and return {(district, year): [prices]}."""
    url = BASE_URL.format(year=year)
    print(f"  Downloading {year} from {url} …", flush=True)
    totals: dict = collections.defaultdict(lambda: [0, 0])  # [price_sum, count]

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total_bytes = 0
        buf = ""
        for chunk in r.iter_content(chunk_size=CHUNK, decode_unicode=True):
            total_bytes += len(chunk.encode("utf-8", errors="replace"))
            buf += chunk
            lines = buf.split("\n")
            buf = lines[-1]  # keep incomplete last line
            for line in lines[:-1]:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Land Registry PP data: no header, quoted fields
                    parts = next(csv.reader([line]))
                except Exception:
                    continue
                if len(parts) < 4:
                    continue
                try:
                    price = int(parts[1].replace(",", ""))
                except ValueError:
                    continue
                postcode = parts[3].strip()
                if not postcode or " " not in postcode:
                    continue
                d = _district(postcode)
                if not d:
                    continue
                totals[(d, year)][0] += price
                totals[(d, year)][1] += 1

        # flush remaining buffer
        if buf.strip():
            try:
                parts = next(csv.reader([buf.strip()]))
                if len(parts) >= 4:
                    price = int(parts[1].replace(",", ""))
                    postcode = parts[3].strip()
                    if postcode and " " in postcode:
                        d = _district(postcode)
                        if d:
                            totals[(d, year)][0] += price
                            totals[(d, year)][1] += 1
            except Exception:
                pass

        mb = total_bytes / 1_048_576
        print(f"    -> {mb:.1f} MB processed, {len(totals):,} district-year pairs", flush=True)

    return totals


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    all_totals: dict = collections.defaultdict(lambda: [0, 0])

    for year in YEARS:
        year_totals = fetch_year(year)
        for key, (s, c) in year_totals.items():
            all_totals[key][0] += s
            all_totals[key][1] += c

    rows = []
    for (district, year), (price_sum, count) in all_totals.items():
        if count == 0:
            continue
        rows.append({
            "postcode_district": district,
            "year": year,
            "average_price": round(price_sum / count),
            "transaction_count": count,
        })

    rows.sort(key=lambda r: (r["postcode_district"], r["year"]))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["postcode_district", "year", "average_price", "transaction_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows):,} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
