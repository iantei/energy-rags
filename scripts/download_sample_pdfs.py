"""
Download a small set of free NREL technical reports to seed data/pdfs/.
Run once before ingesting: python scripts/download_sample_pdfs.py
"""

import urllib.request
from pathlib import Path

# Free public NREL/DOE technical report PDFs
# Replace or extend this list with any reports relevant to your research focus
REPORTS = [
    {
        "name": "nrel_evi_pro_lite.pdf",
        "url": "https://www.nrel.gov/docs/fy21osti/79093.pdf",
        "desc": "EVI-Pro Lite: Estimating Electric Vehicle Charging Infrastructure",
    },
    {
        "name": "nrel_ev_grid_impacts.pdf",
        "url": "https://www.nrel.gov/docs/fy20osti/74064.pdf",
        "desc": "Grid Impacts of Electric Vehicle and Natural Gas Vehicle Deployment",
    },
    {
        "name": "nrel_ev_fleet_electrification.pdf",
        "url": "https://www.nrel.gov/docs/fy21osti/77481.pdf",
        "desc": "Fleet DNA: Commercial Fleet Vehicle Operating Data",
    },
]

OUT_DIR = Path(__file__).parent.parent / "data" / "pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def download(name: str, url: str, desc: str):
    dest = OUT_DIR / name
    if dest.exists():
        print(f"  Already exists, skipping: {name}")
        return
    print(f"  Downloading: {desc}")
    print(f"    {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"    Saved {name} ({size_kb} KB)")
    except Exception as e:
        print(f"    ❌ Failed: {e}")
        print(f"    Download manually from: {url}")


if __name__ == "__main__":
    print(f"\nDownloading NREL sample reports to {OUT_DIR}\n")
    for r in REPORTS:
        download(**r)
    print(f"\nDone. Files in {OUT_DIR}:")
    for f in OUT_DIR.glob("*.pdf"):
        print(f"  {f.name}")
    print("\nNext step: python app.py  (then use the Setup tab to ingest)")
