"""
Populate data/schools_ratings.csv from Ofsted state-funded school inspection outcomes.

Source: DfE / Ofsted — State-funded schools inspections and outcomes as at 31 August 2025
URL: https://assets.publishing.service.gov.uk/media/691ee0612a687551bd8153da/
     State-funded_schools_inspections_and_outcomes_as_at_31_August_2025.csv

Output columns:
    urn, name, postcode, postcode_district, phase, ofsted_rating

ofsted_rating values: Outstanding | Good | Requires improvement | Inadequate | null

Usage:
    python scripts/populate_schools_ratings.py
"""

import csv
import io
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("Install requests first:  pip install requests")

SOURCE_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "691ee0612a687551bd8153da/"
    "State-funded_schools_inspections_and_outcomes_as_at_31_August_2025.csv"
)

RATING_MAP = {"1": "Outstanding", "2": "Good", "3": "Requires improvement", "4": "Inadequate"}

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "schools_ratings.csv",
)


def _district(postcode: str) -> str:
    pc = postcode.strip().upper()
    space = pc.find(" ")
    return pc[:space] if space != -1 else pc[:-3] if len(pc) > 3 else pc


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    print(f"Downloading from {SOURCE_URL} ...", flush=True)
    r = requests.get(SOURCE_URL, timeout=120)
    r.raise_for_status()

    # The file uses UTF-8 with BOM
    text = r.content.decode("latin-1")
    print(f"  Downloaded {len(r.content) / 1_048_576:.1f} MB", flush=True)

    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        urn = row.get("URN", "").strip()
        name = row.get("School name", "").strip()
        postcode = row.get("Postcode", "").strip()
        phase = row.get("Ofsted phase", "").strip()
        rating_raw = row.get("Overall effectiveness", "").strip()

        if not urn or not postcode:
            continue

        rating = RATING_MAP.get(rating_raw)  # None if not yet inspected / N/A
        district = _district(postcode)

        rows.append({
            "urn": urn,
            "name": name,
            "postcode": postcode,
            "postcode_district": district,
            "phase": phase,
            "ofsted_rating": rating or "",
        })

    rows.sort(key=lambda r: (r["postcode_district"], r["name"]))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["urn", "name", "postcode", "postcode_district", "phase", "ofsted_rating"]
        )
        writer.writeheader()
        writer.writerows(rows)

    rated = sum(1 for r in rows if r["ofsted_rating"])
    print(f"Wrote {len(rows):,} schools ({rated:,} with Ofsted ratings) to {OUT_PATH}")


if __name__ == "__main__":
    main()
