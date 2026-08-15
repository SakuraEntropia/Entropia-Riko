"""Brand asset helper — prints the asset paths to replace.

The welcome hero is a JPEG bitmap:

    public/brand/hero.jpg   (960×220 header image)

The app shows a blue gradient placeholder until you drop a real JPEG there.
The logo mark is hand-editable vector art:

    public/brand/logo.svg   (64×64 brand mark)

Run from the project root:

    .venv/bin/python scripts/make_brand_assets.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "public" / "brand"


def main() -> None:
    BRAND.mkdir(parents=True, exist_ok=True)
    print("Brand assets (replace in place):")
    print(f"  header image (JPEG, 960×220): {BRAND / 'hero.jpg'}")
    print(f"  logo mark (SVG, 64×64):       {BRAND / 'logo.svg'}")
    print()
    print("Drop a JPEG at public/brand/hero.jpg to replace the gradient placeholder.")


if __name__ == "__main__":
    main()
