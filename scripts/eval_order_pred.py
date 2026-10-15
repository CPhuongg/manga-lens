"""Đo thứ tự THOẠI với box do DETECTOR dự đoán (số thật của pipeline), trên các trang đã gán nhãn
bằng scripts/label_order.py --mode text.

    python scripts/eval_order_pred.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21" \
        --weights weights/best.pt

Tách nguồn lỗi bằng 4 tổ hợp (panel | text): GT|GT (tham chiếu), dự đoán|GT, GT|dự đoán, dự đoán|dự đoán.
Khi text là box dự đoán: ghép với box text GT bằng IoU (như eval_ocr); tau/đúng hoàn toàn tính trên
các text GHÉP ĐƯỢC (thứ tự tương đối của chúng), còn text GT bị sót và box thừa (FP) báo riêng.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ocr import match_boxes  # noqa: E402
from src.reading_order import PANEL_TOL, kendall_tau, order_page, reading_order  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path, help="thư mục Manga109 (để đọc ảnh)")
ap.add_argument("--weights", default="weights/best.pt")
ap.add_argument("--labels", type=Path, default=Path("data/reading_order_text_labels.json"))
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--iou", type=float, default=0.5, help="ngưỡng IoU ghép box dự đoán với GT")
ap.add_argument("--panel-tol", type=float, default=PANEL_TOL)
ap.add_argument("--device", default=None)
ap.add_argument("--worst", type=int, default=6)
a = ap.parse_args()

labels = json.loads(a.labels.read_text(encoding="utf-8"))
pages = {k: v for k, v in labels.items() if not v["skip"] and len(v["order"]) == len(v["boxes"])}
print(f"{len(pages)} trang nhãn thoại; detector {a.weights} conf>={a.conf}")

from ultralytics import YOLO  # noqa: E402
det = YOLO(a.weights)


def detect(img):
    r = det.predict(img, conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
    fr, tx = [], []
    for xyxy, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()):
        (fr if int(c) == 0 else tx if int(c) == 1 else []).append(xyxy)
    return fr, tx


def flat(layout):
    return [t for item in layout for t in item["texts"]]


def subset_score(gt_order, pred_order, gt_of):
    """gt_of: {chỉ số box dự đoán -> chỉ số GT} (các cặp ghép được). So thứ tự tương đối của chúng."""
    seq = [gt_of[p] for p in pred_order if p in gt_of]
    keep = set(seq)
    ref = [g for g in gt_order if g in keep]
    return kendall_tau(ref, seq), seq == ref


rows = {name: [] for name in ("GT|GT", "pred|GT", "GT|pred", "pred|pred", "pred|pred (thẳng, không panel)")}
n_gt = n_match = n_pred = n_fp = 0
fr_gt = fr_pred = fr_match = 0
per_page = []
for k, v in pages.items():
    img = Image.open(a.root / "images" / v["title"] / f"{v['page']:03d}.jpg").convert("RGB")
    size = tuple(v["size"])
    gt_t, gt_f, gt_order = v["boxes"], v["frames"], v["order"]
    pf, pt = detect(img)

    # panel dự đoán: ghép với panel GT chỉ để thống kê
    fr_gt += len(gt_f); fr_pred += len(pf); fr_match += len(match_boxes(pf, gt_f, a.iou))

    ident = {i: i for i in range(len(gt_t))}
    # 1) GT|GT  2) pred|GT : text GT đầy đủ nên tính trên cả trang
    for name, frames in (("GT|GT", gt_f), ("pred|GT", pf)):
        pred_order = flat(order_page(frames, gt_t, size, panel_tol=a.panel_tol))
        rows[name].append((kendall_tau(gt_order, pred_order), pred_order == gt_order, len(gt_t)))

    # 3) GT|pred  4) pred|pred : text dự đoán -> chỉ tính trên text ghép được
    pairs = match_boxes(pt, gt_t, a.iou)
    gt_of = {p: g for p, g in pairs}
    n_gt += len(gt_t); n_match += len(pairs); n_pred += len(pt); n_fp += len(pt) - len(pairs)
    if len(pairs) >= 2:
        for name, frames in (("GT|pred", gt_f), ("pred|pred", pf)):
            order = flat(order_page(frames, pt, size, panel_tol=a.panel_tol))
            tau, ok = subset_score(gt_order, order, gt_of)
            rows[name].append((tau, ok, len(pairs)))
        tau, ok = subset_score(gt_order, reading_order(pt, size), gt_of)
        rows["pred|pred (thẳng, không panel)"].append((tau, ok, len(pairs)))
        per_page.append((k, rows["pred|pred"][-1][0], len(pairs), len(gt_t), len(pt) - len(pairs),
                         len(pf), len(gt_f)))

print(f"\npanel: GT {fr_gt}, dự đoán {fr_pred}, ghép được (IoU>={a.iou}) {fr_match} "
      f"-> recall {fr_match / fr_gt:.1%}, precision {fr_match / max(1, fr_pred):.1%}")
print(f"text : GT {n_gt}, dự đoán {n_pred}, ghép được {n_match} -> recall {n_match / n_gt:.1%}, "
      f"box thừa (FP) {n_fp} ({n_fp / max(1, n_pred):.1%})")
print(f"\n{'panel | text':34s} {'trang':>5s} {'tau mean':>9s} {'median':>7s} {'đúng hoàn toàn':>15s}")
for name, rs in rows.items():
    if rs:
        print(f"{name:34s} {len(rs):5d} {statistics.mean(r[0] for r in rs):9.3f} "
              f"{statistics.median(r[0] for r in rs):7.3f} {sum(r[1] for r in rs):8d}/{len(rs)} "
              f"({sum(r[1] for r in rs) / len(rs):.0%})")

print(f"\n{a.worst} trang tau thấp nhất của pred|pred "
      "(trang, tau, text ghép/GT, FP, panel dự đoán/GT):")
for k, tau, m, g, fp, npf, ngf in sorted(per_page, key=lambda r: r[1])[:a.worst]:
    print(f"  {k:24s} tau={tau:5.2f}  text {m}/{g}  FP {fp}  panel {npf}/{ngf}")
