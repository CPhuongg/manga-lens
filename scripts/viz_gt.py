"""Vẽ GT của Manga109 kèm số thứ tự trong XML để KIỂM TRA xem thứ tự annotation
có trùng thứ tự đọc không (đừng giả định trước khi dùng làm GT cho reading order).

    python scripts/viz_gt.py --root /path/Manga109 --title ARMS --page 10 --out gt.png
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from convert_manga109 import parse_book  # noqa: E402

COL = {"frame": "#e63946", "text": "#1d7af3", "face": "#2a9d8f", "body": "#f4a261"}

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--title", required=True)
ap.add_argument("--page", type=int, required=True)
ap.add_argument("--out", default="gt.png")
a = ap.parse_args()

pages = parse_book(a.root / "annotations" / f"{a.title}.xml")
pg = next(p for p in pages if p["index"] == a.page)
img = Image.open(a.root / "images" / a.title / f"{a.page:03d}.jpg").convert("RGB")
d = ImageDraw.Draw(img)
cnt = {}
for cls, x1, y1, x2, y2, txt in pg["boxes"]:
    cnt[cls] = cnt.get(cls, 0) + 1
    d.rectangle((x1, y1, x2, y2), outline=COL[cls], width=3)
    d.text((x1 + 4, y1 + 4), f"{cls[0]}{cnt[cls]}", fill=COL[cls])
img.save(a.out)
print("Đã lưu", a.out)
