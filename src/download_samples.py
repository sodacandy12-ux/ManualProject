# -*- coding: utf-8 -*-
"""Wikimedia Commons에서 라이선스 있는 샘플 사진 5장을 받는다."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "assets" / "samples"

# Special:FilePath 는 해시 경로를 몰라도 된다. width로 용량을 줄인다.
FILES = [
    ("fire_smoke_indoor.jpg", "KingsXfire.jpg"),
    ("road_flood_vehicles.jpg", "Hurricane_Ian_flooding_at_UTC_1.jpg"),
    ("underground_flood_basement.jpg", "Flood_Basement_ACG_(5075030165).jpg"),
    ("building_crack_collapse.jpg", "Berkman_Plaza_II_Collapse.jpg"),
    ("landslide_road.jpg", "Oso_Mudslide_22_March_2014_Aerial_view.jpg"),
]


def fetch(filename: str, dest: Path) -> None:
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}?width=1280"
    req = Request(url, headers={"User-Agent": "DisasterVision-RAG/1.0 (local demo)"})
    print(f"GET {filename}")
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    if len(data) < 2000:
        raise RuntimeError(f"too small: {filename} ({len(data)} bytes)")
    dest.write_bytes(data)
    print(f"  -> {dest.name} ({len(data)} bytes)")


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, wiki_name in FILES:
        dest = SAMPLE_DIR / local_name
        try:
            fetch(wiki_name, dest)
        except Exception as exc:
            print(f"FAIL {local_name}: {exc}")


if __name__ == "__main__":
    main()
