#!/usr/bin/env python3
"""
Install Dashboard API Layer
Run from inside your Atlas folder: python install_dashboard.py
"""
import os, sys, shutil

def main():
    if not os.path.exists("app"):
        print("ERROR: Run this from inside your Atlas folder")
        sys.exit(1)

    src_dir = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(src_dir, "dashboard_main.py")
    dst = os.path.join(".", "dashboard_main.py")

    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)
        print("Copied dashboard_main.py to Atlas folder")
    elif os.path.abspath(src) == os.path.abspath(dst):
        print("dashboard_main.py already in Atlas folder")
    else:
        print("ERROR: dashboard_main.py not found in extension folder")
        sys.exit(1)

    print()
    print("=" * 50)
    print("DASHBOARD API LAYER INSTALLED")
    print("=" * 50)
    print()
    print("Run the dashboard API with:")
    print("  uvicorn dashboard_main:app --reload --port 8000")
    print()
    print("Endpoints available:")
    print("  POST /analyse-property  <- main Lovable endpoint")
    print("  GET  /market-heatmap")
    print("  GET  /deal-scanner")
    print("  GET  /risk-analysis")
    print("  GET  /true-value")
    print("  GET  /liquidity-score")
    print("  GET  /development-potential")
    print("  GET  /health")
    print()
    print("Connect Lovable to: http://localhost:8000")

if __name__ == "__main__":
    main()
