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
from src.ocr import (MISS_CATS, add_low_conf, cer, crop, diagnose_miss, levenshtein,  # noqa: E402
                     match_boxes, normalize, ocr_japanese)

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--titles", required=True, type=Path)
ap.add_argument("--max-pages", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--weights", default=None, help="best.pt của detector -> đo thêm CER trên box dự đoán")
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--iou", type=float, default=0.5, help="ngưỡng IoU để ghép box dự đoán với GT")
ap.add_argument("--strict", action="store_true",
                help="so khớp từng ký tự (chỉ bỏ khoảng trắng); mặc định bỏ khác biệt dấu câu "
                     "(．/…/・, toàn/nửa chiều rộng)")
ap.add_argument("--conf-low", type=float, default=None,
                help="(cần --weights) lấp chỗ trống: nhận thêm box conf trong [conf-low, conf) nếu "
                     "không chồng box đã nhận; so sánh recall/FP trước-sau")
ap.add_argument("--low-overlap", type=float, default=0.3,
                help="box conf thấp bị loại nếu giao/diện tích-box-nhỏ >= ngưỡng này với box đã nhận")
ap.add_argument("--diag", action="store_true",
                help="(cần --weights) phân loại vì sao box GT bị sót: gộp/hạ conf/xé mảnh/lệch/không thấy")
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
base_match, base_pred, n_extra, n_extra_ok = 0, 0, 0, 0
misses = []  # (loại, IoU tốt nhất, conf, trang, WxH, text) cho --diag
gt_edits, gt_chars, records = 0, 0, []  # CER theo tổng ký tự + chi tiết từng box để xem lỗi
for t, i in pages:
    img = Image.open(a.root / "images" / t / f"{i:03d}.jpg").convert("RGB")
    gts = by_page[(t, i)]
    for x1, y1, x2, y2, gt in gts:
        c = crop(img, (x1, y1, x2, y2))
        pred = ocr_japanese(c)
        gt_scores.append(cer(pred, gt, a.strict))
        gt_edits += levenshtein(normalize(pred, a.strict), normalize(gt, a.strict))
        gt_chars += len(normalize(gt, a.strict))
        records.append((gt_scores[-1], t, i, (x1, y1, x2, y2), gt, pred, c))
    if det is None:
        continue
    # chạy ở conf thấp nhất cần (--diag / --conf-low) rồi lọc; kết quả >= a.conf giống hệt chạy
    # trực tiếp ở a.conf (NMS xét theo điểm)
    run_conf = min(a.conf, 0.01 if a.diag else a.conf, a.conf_low if a.conf_low else a.conf)
    res = det.predict(img, imgsz=a.imgsz, conf=run_conf, classes=[1], verbose=False)[0]
    all_p = list(zip(res.boxes.xyxy.cpu().tolist(), res.boxes.conf.cpu().tolist()))
    hi = [b for b, c in all_p if c >= a.conf]
    gt_boxes = [g[:4] for g in gts]
    base_pairs = match_boxes(hi, gt_boxes, a.iou)  # baseline: chỉ conf >= a.conf (không tốn OCR)
    base_match += len(base_pairs)
    base_pred += len(hi)
    preds = hi
    if a.conf_low:
        cands = [(b, c) for b, c in all_p if a.conf_low <= c < a.conf]
        preds = hi + add_low_conf(hi, cands, a.low_overlap)
    pairs = match_boxes(preds, gt_boxes, a.iou)
    n_extra += len(preds) - len(hi)
    n_extra_ok += sum(p >= len(hi) for p, _ in pairs)
    n_pred += len(preds)
    n_fp += len(preds) - len(pairs)
    score = {j: cer(ocr_japanese(crop(img, preds[p])), gts[j][4], a.strict) for p, j in pairs}
    for j in range(len(gts)):
        if a.diag and j not in score:
            g = gts[j]
            cat, bi, bc = diagnose_miss(g[:4], all_p, a.conf, a.iou)
            misses.append((cat, bi, bc, f"{t}/{i:03d}", f"{g[2] - g[0]}x{g[3] - g[1]}", g[4]))
    matched_scores += score.values()
    e2e_scores += [score.get(j, 1.0) for j in range(len(gts))]  # GT bị bỏ sót tính CER = 1


def fmt(name, s):
    if not s:
        print(f"{name}: 0 box")
        return
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
    if a.conf_low:
        print(f"lấp chỗ trống (conf-low={a.conf_low}, overlap<{a.low_overlap}): "
              f"thêm {n_extra} box, đúng (ghép được GT) {n_extra_ok}, thừa (FP mới) {n_extra - n_extra_ok}")
        print(f"  baseline conf>={a.conf}: recall {base_match / n_gt:.1%} | FP {base_pred - base_match}/{base_pred}")
        print(f"  sau lấp chỗ trống     : recall {len(matched_scores) / n_gt:.1%} | FP {n_fp}/{n_pred}")

if a.diag and det is not None:
    print(f"\nChẩn đoán {len(misses)} box GT bị sót (IoU>={a.iou}, conf>={a.conf}):")
    for cat in MISS_CATS:
        n = sum(m[0] == cat for m in misses)
        print(f"  {cat:9s}: {n:4d} ({n / max(1, len(misses)):.0%})")
    print("  (loại, IoU tốt nhất, conf của box đó, trang, WxH, GT; tối đa 15 box mỗi loại):")
    for cat in MISS_CATS:
        for m in sorted((m for m in misses if m[0] == cat), key=lambda m: -m[1])[:15]:
            print(f"  {m[0]:8s} iou={m[1]:.2f} conf={m[2]:.2f} {m[3]} {m[4]} {m[5]!r}")

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
