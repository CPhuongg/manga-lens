"""CER của OCR: trên text box GT (tách riêng lỗi detection) và, nếu có --weights, trên box do
detector dự đoán (đo lỗi tích lũy detection + OCR).

    python scripts/eval_ocr.py --root /path/Manga109 --titles data/yolo/test_titles.txt --max-pages 200
    python scripts/eval_ocr.py --root ... --titles ... --weights weights/best.pt
    python scripts/eval_ocr.py --root ... --titles ... --worst 30 --dump-dir ocr_worst
        (in 30 box GT sai nhiều nhất; --dump-dir lưu ảnh cắt + worst.tsv, CHỈ để xem local,
         không đưa lên repo/README vì là trang truyện có bản quyền)
"""
import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_manga109 import parse_book  # noqa: E402
from src.ocr import cer, crop, levenshtein, match_boxes, normalize, ocr_japanese  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--titles", required=True, type=Path)
ap.add_argument("--max-pages", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--weights", default=None, help="best.pt của detector -> đo thêm CER trên box dự đoán")
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--iou", type=float, default=0.5, help="ngưỡng IoU để ghép box dự đoán với GT")
ap.add_argument("--worst", type=int, default=0, help="in N box GT có CER cao nhất")
ap.add_argument("--dump-dir", type=Path, default=None, help="lưu ảnh cắt của các box --worst + worst.tsv")
a = ap.parse_args()

by_page = defaultdict(list)
for t in a.titles.read_text().split():
    for pg in parse_book(a.root / "annotations" / f"{t}.xml"):
        for b in pg["boxes"]:
            if b[0] == "text" and b[5]:
                by_page[(t, pg["index"])].append(b[1:])  # (x1, y1, x2, y2, gt_text)
pages = sorted(by_page)
random.Random(a.seed).shuffle(pages)
pages = pages[:a.max_pages]

det = None
if a.weights:
    from ultralytics import YOLO
    det = YOLO(a.weights)

gt_scores, matched_scores, e2e_scores, n_fp, n_pred = [], [], [], 0, 0
gt_edits, gt_chars, records = 0, 0, []  # CER theo tổng ký tự + chi tiết từng box để xem lỗi
for t, i in pages:
    img = Image.open(a.root / "images" / t / f"{i:03d}.jpg").convert("RGB")
    gts = by_page[(t, i)]
    for x1, y1, x2, y2, gt in gts:
        c = crop(img, (x1, y1, x2, y2))
        pred = ocr_japanese(c)
        gt_scores.append(cer(pred, gt))
        gt_edits += levenshtein(normalize(pred), normalize(gt))
        gt_chars += len(normalize(gt))
        records.append((gt_scores[-1], t, i, (x1, y1, x2, y2), gt, pred, c))
    if det is None:
        continue
    res = det.predict(img, imgsz=a.imgsz, conf=a.conf, classes=[1], verbose=False)[0]
    preds = res.boxes.xyxy.cpu().tolist()
    pairs = match_boxes(preds, [g[:4] for g in gts], a.iou)
    n_pred += len(preds)
    n_fp += len(preds) - len(pairs)
    score = {j: cer(ocr_japanese(crop(img, preds[p])), gts[j][4]) for p, j in pairs}
    matched_scores += score.values()
    e2e_scores += [score.get(j, 1.0) for j in range(len(gts))]  # GT bị bỏ sót tính CER = 1


def fmt(name, s):
    exact = sum(v == 0 for v in s) / len(s)
    print(f"{name}: {len(s)} box | CER mean {sum(s) / len(s):.3f} | median "
          f"{statistics.median(s):.3f} | đúng hoàn toàn {exact:.1%}")


print(f"{len(pages)} trang")
fmt("GT box        ", gt_scores)
print(f"  CER theo tổng ký tự (GT box): {gt_edits / max(1, gt_chars):.3f} ({gt_chars} ký tự GT)")
if det is not None:
    fmt("box dự đoán  ", matched_scores)
    fmt("end-to-end   ", e2e_scores)
    n_gt = len(e2e_scores)
    print(f"recall box text (IoU>={a.iou}, conf>={a.conf}): {len(matched_scores) / n_gt:.1%} | "
          f"box dự đoán thừa (FP): {n_fp}/{n_pred}")
    print("  (box dự đoán = chỉ các box ghép được với GT; end-to-end = GT bị bỏ sót tính CER 1.0)")

if a.worst:
    records.sort(key=lambda r: r[0], reverse=True)
    top = records[:a.worst]
    if a.dump_dir:
        a.dump_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"\n{a.worst} box GT sai nhiều nhất (CER, trang, kích thước WxH, GT -> dự đoán):")
    for n, (c, t, i, (x1, y1, x2, y2), gt, pred, im) in enumerate(top):
        w, h = x2 - x1, y2 - y1
        print(f"{c:.2f} {t}/{i:03d} {w}x{h} | {gt!r} -> {pred!r}")
        name = f"{n:02d}_{t}_{i:03d}.png"
        if a.dump_dir:
            im.save(a.dump_dir / name)
        rows.append(f"{name}\t{c:.3f}\t{t}\t{i}\t{w}x{h}\t{gt}\t{pred}")
    if a.dump_dir:
        (a.dump_dir / "worst.tsv").write_text("\n".join(rows), encoding="utf-8")
        print(f"đã lưu ảnh + worst.tsv vào {a.dump_dir}")
