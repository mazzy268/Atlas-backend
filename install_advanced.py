#!/usr/bin/env python3
"""
Atlas Advanced Features - Installation Script
Run from inside your Atlas folder: python install_advanced.py
Copies all new files into the correct locations and updates main.py
"""
import os
import sys
import shutil

def main():
    if not os.path.exists("app"):
        print("ERROR: Run this from inside your Atlas folder")
        sys.exit(1)

    print("Installing Atlas Advanced Features...")
    print()

    # Create directories
    dirs = [
        "app/services/advanced",
        "app/api/endpoints",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()
    print("Directories created")

    # Copy service files
    src_dir = os.path.dirname(os.path.abspath(__file__))
    
    service_files = [
        "market_heatmap.py",
        "liquidity_engine.py",
        "true_value.py",
        "development_potential.py",
        "infrastructure_impact.py",
        "street_intelligence.py",
        "market_risk.py",
        "deal_scanner.py",
    ]

    for fname in service_files:
        src = os.path.join(src_dir, "app", "services", "advanced", fname)
        dst = os.path.join("app", "services", "advanced", fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied services/advanced/{fname}")
        else:
            print(f"  WARNING: {src} not found - skipping")

    # Copy advanced init
    adv_init_src = os.path.join(src_dir, "app", "services", "advanced", "__init__.py")
    adv_init_dst = "app/services/advanced/__init__.py"
    if os.path.exists(adv_init_src):
        shutil.copy2(adv_init_src, adv_init_dst)

    # Copy router
    router_src = os.path.join(src_dir, "app", "api", "endpoints", "advanced.py")
    router_dst = "app/api/endpoints/advanced.py"
    if os.path.exists(router_src):
        shutil.copy2(router_src, router_dst)
        print("  Copied api/endpoints/advanced.py")

    # Update main.py to include the new router
    print()
    print("Updating main.py...")
    main_path = "app/main.py"
    if os.path.exists(main_path):
        with open(main_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "advanced" not in content:
            # Add import
            old_import = "from app.api.endpoints import property, sales, crime, demographics, flood, portfolio"
            new_import = (
                "from app.api.endpoints import property, sales, crime, demographics, flood, portfolio\n"
                "from app.api.endpoints.advanced import router as advanced_router"
            )
            content = content.replace(old_import, new_import)

            # Add router registration
            old_router = "app.include_router(portfolio.router, tags=[\"Portfolio\"])"
            new_router = (
                "app.include_router(portfolio.router, tags=[\"Portfolio\"])\n"
                "app.include_router(advanced_router, tags=[\"Advanced Intelligence\"])"
            )
            content = content.replace(old_router, new_router)

            with open(main_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("  main.py updated with advanced router")
        else:
            print("  main.py already has advanced router - skipping")
    else:
        print("  WARNING: main.py not found")
        print("  Manually add to main.py:")
        print("    from app.api.endpoints.advanced import router as advanced_router")
        print("    app.include_router(advanced_router, tags=['Advanced Intelligence'])")

    # Install pandas and scikit-learn
    print()
    print("Installing additional packages...")
    os.system(f"{sys.executable} -m pip install pandas scikit-learn --quiet")
    print("  pandas and scikit-learn installed")

    print()
    print("=" * 55)
    print("INSTALLATION COMPLETE")
    print("=" * 55)
    print()
    print("8 new endpoints added:")
    print("  GET  /market-heatmap")
    print("  GET  /liquidity-score")
    print("  GET  /true-value")
    print("  POST /development-potential")
    print("  GET  /infrastructure-impact")
    print("  GET  /street-intelligence")
    print("  GET  /risk-analysis")
    print("  GET  /deal-scanner")
    print()
    print("Restart your API:")
    print("  uvicorn app.main:app --port 8001")
    print()
    print("Then open:")
    print("  http://127.0.0.1:8001/docs")
    print()
    print("Scroll down to 'Advanced Intelligence' section")

if __name__ == "__main__":
    main()
