"""Đo thứ tự đọc trên nhãn tự gán (scripts/label_order.py).

    python scripts/eval_order.py --mode text    # THOẠI (chuẩn đo chính)
    python scripts/eval_order.py --mode panel   # panel (chẩn đoán)
    python scripts/eval_order.py --mode text --worst 10

Chỉ số: Kendall tau (1 = đúng hoàn toàn) và % trang đúng hoàn toàn. Input là box GT (không lẫn
lỗi detector). Chế độ text so 3 cách:
  - qua panel : order_page(frame GT, text GT) = cách pipeline đang dùng
  - thẳng     : XY-cut trực tiếp trên box text, KHÔNG dùng panel
  - baseline  : trên->dưới rồi phải->trái theo tâm box
=> trả lời "có cần panel để sắp thoại không".
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reading_order import kendall_tau, order_page, reading_order  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["text", "panel"], default="text")
ap.add_argument("--labels", type=Path, default=None)
ap.add_argument("--worst", type=int, default=0, help="in N trang có tau thấp nhất")
a = ap.parse_args()

path = a.labels or Path("data/reading_order_text_labels.json" if a.mode == "text"
                        else "data/reading_order_labels.json")
labels = json.loads(path.read_text(encoding="utf-8"))
pages = {k: v for k, v in labels.items() if not v["skip"] and len(v["order"]) == len(v["boxes"])}
print(f"[{a.mode}] {len(pages)} trang có nhãn ({len(labels) - len(pages)} trang bỏ qua) từ {path}")


def naive(boxes, size):
    """Baseline: hàng từ trên xuống (tâm y), trong hàng từ phải sang trái. Không chia trang đôi."""
    return sorted(range(len(boxes)),
                  key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, -(boxes[i][0] + boxes[i][2]) / 2))


def evaluate(fn):
    rows = []
    for k, v in pages.items():
        pred = fn(v)
        rows.append((k, len(v["boxes"]), kendall_tau(v["order"], pred), pred == v["order"]))
    return rows


def report(name, rows):
    taus = [r[2] for r in rows]
    print(f"{name:36s} tau mean {statistics.mean(taus):.3f} | median {statistics.median(taus):.3f} "
          f"| đúng hoàn toàn {sum(r[3] for r in rows) / len(rows):.1%}")


def by(fn_b):
    return lambda v: fn_b(v["boxes"], tuple(v["size"]))


methods = {}
if a.mode == "text":
    def via_panels(**kw):
        return lambda v: [t for item in order_page(v.get("frames", []), v["boxes"], tuple(v["size"]),
                                                   rtl=True, **kw) for t in item["texts"]]

    methods["qua panel (pipeline, panel_tol=0.05)"] = via_panels()
    methods["qua panel (dung sai cũ 0.01)"] = via_panels(panel_tol=0.01)
    methods["thẳng trên box text (không panel)"] = by(lambda b, s: reading_order(b, s, rtl=True))
    methods["thẳng, tắt tách trang đôi"] = by(lambda b, s: reading_order(b, s, rtl=True, spread=False))
else:
    methods["XY-cut panel, tol 0.05 (pipeline)"] = by(lambda b, s: reading_order(b, s, rtl=True, tol_ratio=0.05))
    methods["XY-cut panel, tol 0.01 (cũ)"] = by(lambda b, s: reading_order(b, s, rtl=True))
    methods["tol 0.05, tắt tách trang đôi"] = by(
        lambda b, s: reading_order(b, s, rtl=True, tol_ratio=0.05, spread=False))
methods["baseline (y rồi phải->trái)"] = by(naive)

results = {name: evaluate(fn) for name, fn in methods.items()}
for name, rows in results.items():
    report(name, rows)

main = next(iter(results.values()))
print(f"\ntheo số box ({next(iter(results))}):")
for lo, hi in ((2, 5), (6, 12), (13, 999)):
    sub = [r for r in main if lo <= r[1] <= hi]
    if sub:
        print(f"  {lo}-{hi} box: {len(sub)} trang | tau {statistics.mean(r[2] for r in sub):.3f} "
              f"| đúng hoàn toàn {sum(r[3] for r in sub) / len(sub):.1%}")

if a.worst:
    print(f"\n{a.worst} trang tau thấp nhất ({next(iter(results))}):")
    for k, n, tau, ok in sorted(main, key=lambda r: r[2])[:a.worst]:
        note = labels[k].get("note", "")
        print(f"  {k} n={n} tau={tau:.2f}" + (f" | ghi chú: {note}" if note else ""))
