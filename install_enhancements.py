#!/usr/bin/env python3
"""
Atlas Data Enhancement Installer
Run from inside your Atlas folder: python install_enhancements.py
Replaces 3 service files with improved versions.
"""
import os, sys, shutil

FILES = {
    "true_value.py":    "app/services/advanced/true_value.py",
    "deal_scanner.py":  "app/services/advanced/deal_scanner.py",
    "market_heatmap.py":"app/services/advanced/market_heatmap.py",
}

def main():
    if not os.path.exists("app"):
        print("ERROR: Run this from inside your Atlas folder")
        sys.exit(1)

    src_dir = os.path.dirname(os.path.abspath(__file__))
    replaced = 0

    for fname, dst_rel in FILES.items():
        src = os.path.join(src_dir, fname)
        dst = os.path.join(".", dst_rel)
        if not os.path.exists(src):
            print(f"  SKIP {fname} — not found next to this script")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Replaced {dst_rel}")
        replaced += 1

    print()
    print(f"Done — {replaced} files upgraded.")
    print()
    print("What improved:")
    print("  true_value.py    — VOA 2024 rents, recency weighting, type-matched comps,")
    print("                     sector fallback, confidence + data_freshness fields")
    print("  deal_scanner.py  — 24-month window only, recency median benchmark,")
    print("                     stronger BMV detection, data_freshness field")
    print("  market_heatmap.py— ONS HPI regional growth rates, genuine 12m vs prior")
    print("                     12m momentum, data_freshness field")
    print()
    print("Restart:")
    print("  uvicorn dashboard_main:app --reload --port 8000")

if __name__ == "__main__":
    main()
