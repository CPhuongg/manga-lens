"""Chuyển Manga109 (XML) sang định dạng YOLO, chia split THEO BỘ TRUYỆN.

Dùng:
    python scripts/convert_manga109.py --root /path/Manga109 --out data/yolo
    (thêm --copy nếu dùng Windows hoặc không muốn symlink)

Cấu trúc Manga109 mong đợi:
    <root>/images/<Title>/000.jpg ...
    <root>/annotations/<Title>.xml
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["frame", "text", "face", "body"]  # id = vị trí trong list
CLS_ID = {c: i for i, c in enumerate(CLASSES)}


def split_titles(titles, seed=0, val=0.1, test=0.1):
    """Chia theo bộ truyện để tránh leak (cùng nhân vật/style giữa train và test)."""
    titles = sorted(titles)
    rng = random.Random(seed)
    rng.shuffle(titles)
    n = len(titles)
    n_val, n_test = max(1, round(n * val)), max(1, round(n * test))
    return {
        "test": titles[:n_test],
        "val": titles[n_test:n_test + n_val],
        "train": titles[n_test + n_val:],
    }


def parse_book(xml_path):
    """Trả về list page: {index, w, h, boxes:[(cls, xmin,ymin,xmax,ymax, text)]}."""
    root = ET.parse(xml_path).getroot()
    pages = []
    for page in root.iter("page"):
        w, h = int(page.get("width")), int(page.get("height"))
        boxes = []
        for el in page:
            if el.tag not in CLS_ID:
                continue
            x1, y1, x2, y2 = (int(el.get(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append((el.tag, x1, y1, x2, y2, (el.text or "").strip()))
        pages.append({"index": int(page.get("index")), "w": w, "h": h, "boxes": boxes})
    return pages


def to_yolo_line(cls, x1, y1, x2, y2, w, h):
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{CLS_ID[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("data/yolo"), type=Path)
    ap.add_argument("--copy", action="store_true", help="copy ảnh thay vì symlink")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    xmls = {p.stem: p for p in (args.root / "annotations").glob("*.xml")}
    if not xmls:
        raise SystemExit(f"Không thấy XML trong {args.root / 'annotations'}")
    splits = split_titles(list(xmls), seed=args.seed)

    for split, titles in splits.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_img = 0
        for title in titles:
            for pg in parse_book(xmls[title]):
                name = f"{title}_{pg['index']:03d}"
                src = args.root / "images" / title / f"{pg['index']:03d}.jpg"
                if not src.exists():
                    continue
                link_or_copy(src, args.out / "images" / split / f"{name}.jpg", args.copy)
                lines = [to_yolo_line(c, x1, y1, x2, y2, pg["w"], pg["h"])
                         for c, x1, y1, x2, y2, _ in pg["boxes"]]
                (args.out / "labels" / split / f"{name}.txt").write_text(
                    "\n".join(l for l in lines if l))
                n_img += 1
        print(f"{split}: {len(titles)} bộ, {n_img} trang")
        (args.out / f"{split}_titles.txt").write_text("\n".join(sorted(titles)))

    yaml = (f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n"
            f"test: images/test\nnames:\n" +
            "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (args.out / "data.yaml").write_text(yaml)
    print("Đã ghi", args.out / "data.yaml")


if __name__ == "__main__":
    main()
