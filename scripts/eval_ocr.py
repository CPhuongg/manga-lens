"""CER của OCR trên text box GT (tách riêng khỏi lỗi detection).

    python scripts/eval_ocr.py --root /path/Manga109 --titles data/yolo/test_titles.txt --max-pages 200
"""
import argparse
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_manga109 import parse_book  # noqa: E402
from src.ocr import cer, crop, ocr_japanese  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--titles", required=True, type=Path)
ap.add_argument("--max-pages", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

items = []
for t in a.titles.read_text().split():
    for pg in parse_book(a.root / "annotations" / f"{t}.xml"):
        for b in pg["boxes"]:
            if b[0] == "text" and b[5]:
                items.append((t, pg["index"], b))
pages = sorted({(t, i) for t, i, _ in items})
random.Random(a.seed).shuffle(pages)
keep = set(pages[:a.max_pages])

scores = []
for t, i, (_, x1, y1, x2, y2, gt) in items:
    if (t, i) not in keep:
        continue
    img = Image.open(a.root / "images" / t / f"{i:03d}.jpg").convert("RGB")
    scores.append(cer(ocr_japanese(crop(img, (x1, y1, x2, y2))), gt))
print(f"{len(scores)} text box, CER trung bình = {sum(scores) / len(scores):.3f}")
