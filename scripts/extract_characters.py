"""Trích nhãn nhân vật (face/body) từ XML Manga109 -> data/characters.json.

Converter YOLO bỏ qua thuộc tính `character`; script này giữ lại, kèm split theo bộ
(đọc `data/yolo/{split}_titles.txt`). Id nhân vật chỉ có nghĩa TRONG từng bộ.

Dùng:
    python scripts/extract_characters.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21"
"""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# nhân vật nền gộp chung nhiều người -> không phải 1 danh tính
OTHER = re.compile(r"^(others?|その他|モブ.*|群衆.*|通行人.*)$", re.I)


def parse_book(xml_path):
    root = ET.parse(xml_path).getroot()
    chars = {c.get("id"): c.get("name") or "" for c in root.iter("character")}
    items = []
    for pg in root.iter("page"):
        for el in pg:
            if el.tag not in ("face", "body"):
                continue
            items.append({
                "page": int(pg.get("index")), "w": int(pg.get("width")), "h": int(pg.get("height")),
                "cls": el.tag, "id": el.get("id"), "char": el.get("character"),
                "box": [int(el.get(k)) for k in ("xmin", "ymin", "xmax", "ymax")],
            })
    return chars, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--yolo", default=Path("data/yolo"), type=Path, help="chứa {split}_titles.txt")
    ap.add_argument("--out", default=Path("data/characters.json"), type=Path)
    args = ap.parse_args()

    split_of = {t: s for s in ("train", "val", "test")
                for t in (args.yolo / f"{s}_titles.txt").read_text().split()}
    books = {}
    for p in sorted((args.root / "annotations").glob("*.xml")):
        chars, items = parse_book(p)
        books[p.stem] = {
            "split": split_of[p.stem],
            "characters": {i: {"name": n, "other": bool(OTHER.match(n))} for i, n in chars.items()},
            "items": items,
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(books, ensure_ascii=False))
    n = sum(len(b["items"]) for b in books.values())
    print(f"{len(books)} bộ, {n} box face/body -> {args.out}")


if __name__ == "__main__":
    main()
