"""Dựng nhãn người nói (speaker attribution) từ Manga109Dialog + annotations 2026 -> data/speaker_labels.json.

Manga109Dialog (CC BY 4.0, github.com/manga109/public-annotations/Manga109Dialog) gồm `speaker_to_text`
(text_id, speaker_id) theo annotations BẢN 2021.12.30. Đã kiểm tra: `speaker_id` luôn là id <body> cùng trang,
và id face/body/frame giữ nguyên ở bản 2026; còn id <text> bị đánh số lại hoàn toàn -> ghép text 2021 với
text 2026 theo IoU trong từng trang (khớp 1-1, tham lam từ IoU lớn).

    python scripts/build_speaker_labels.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21"

Dữ liệu nguồn Manga109 (không chuyển cho bên thứ ba): file đầu ra nằm trong data/ (gitignored).
Mỗi trang: size, frames/texts/faces/bodies (box 2026 kèm id, character), `links` = {id text 2026: [id body người nói]}.
Text KHÔNG có trong links = Dialog không gán nhãn (lời dẫn, tiêu đề, người nói ngoài trang...), chưa chắc là "không người nói".
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_characters import OTHER  # nhân vật nền gộp nhiều người (cùng quy tắc với re-ID)

BOX = ("xmin", "ymin", "xmax", "ymax")


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def read_pages(path):
    root = ET.parse(path).getroot()
    chars = {c.get("id"): c.get("name") or "" for c in root.iter("character")}
    pages = {}
    for pg in root.iter("page"):
        d = {"w": int(pg.get("width")), "h": int(pg.get("height")),
             "frame": [], "text": [], "face": [], "body": []}
        for el in pg:
            if el.tag in d and el.tag not in ("w", "h"):
                d[el.tag].append({"id": el.get("id"), "box": [int(el.get(k)) for k in BOX],
                                  "char": el.get("character"), "s": (el.text or "").strip()})
        pages[int(pg.get("index"))] = d
    return chars, pages


def match_text(old, new, thr):
    """Khớp 1-1 text cũ -> text mới theo IoU >= thr. Trả {id cũ: id mới}."""
    cand = sorted(((iou(o["box"], n["box"]), i, j) for i, o in enumerate(old) for j, n in enumerate(new)),
                  reverse=True)
    used_o, used_n, out = set(), set(), {}
    for v, i, j in cand:
        if v < thr:
            break
        if i in used_o or j in used_n:
            continue
        used_o.add(i), used_n.add(j)
        out[old[i]["id"]] = new[j]["id"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dialog", default=Path("data/dialog"), type=Path)
    ap.add_argument("--yolo", default=Path("data/yolo"), type=Path, help="chứa {split}_titles.txt")
    ap.add_argument("--out", default=Path("data/speaker_labels.json"), type=Path)
    ap.add_argument("--iou", type=float, default=0.5)
    a = ap.parse_args()

    split_of = {t: s for s in ("train", "val", "test")
                for t in (a.yolo / f"{s}_titles.txt").read_text().split()}
    books, stats = {}, Counter()
    for dp in sorted(a.dialog.glob("*.xml")):
        t = dp.stem
        chars, new = read_pages(a.root / "annotations" / f"{t}.xml")
        _, old = read_pages(a.root / "annotations.v2021.12.30" / f"{t}.xml")
        pages = {}
        for pg in ET.parse(dp).getroot().iter("page"):
            idx = int(pg.get("index"))
            pairs = [(s.get("text_id"), s.get("speaker_id")) for s in pg.iter("speaker_to_text")]
            if not pairs:
                continue
            m = match_text(old[idx]["text"], new[idx]["text"], a.iou)
            links = {}
            body_ids = {b["id"] for b in new[idx]["body"]}
            for tid, sid in pairs:
                stats[(split_of[t], "pairs")] += 1
                if tid not in m:
                    stats[(split_of[t], "pairs_unmatched_text")] += 1
                    continue
                if sid not in body_ids:
                    stats[(split_of[t], "pairs_no_body")] += 1
                    continue
                links.setdefault(m[tid], []).append(sid)
            stats[(split_of[t], "texts_linked")] += len(links)
            pages[idx] = {"links": links}
        for idx, pl in pages.items():
            pl.update({k: new[idx][k] for k in ("w", "h", "frame", "text", "face", "body")})
        books[t] = {"split": split_of[t], "characters": {i: {"name": n, "other": bool(OTHER.match(n))} for i, n in chars.items()}, "pages": {str(i): p for i, p in pages.items()}}

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"iou_text_match": a.iou, "books": books}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(books)} bộ -> {a.out}")
    for s in ("train", "val", "test"):
        n = stats[(s, "pairs")]
        print(f"{s:5s}: {n} cặp Dialog | text 2021 không ghép được sang 2026: {stats[(s, 'pairs_unmatched_text')]} "
              f"({stats[(s, 'pairs_unmatched_text')] / max(1, n):.1%}) | speaker không phải body 2026: "
              f"{stats[(s, 'pairs_no_body')]} | text 2026 có nhãn: {stats[(s, 'texts_linked')]}")


if __name__ == "__main__":
    main()
