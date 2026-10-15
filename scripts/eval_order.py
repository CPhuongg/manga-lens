"""Đo thuật toán reading_order trên nhãn tự gán (scripts/label_order.py).

    python scripts/eval_order.py
    python scripts/eval_order.py --labels data/reading_order_labels.json --worst 10

Chỉ số: Kendall tau (1 = đúng hoàn toàn) và % trang đúng hoàn toàn. So với baseline ngây thơ
"trên->dưới, phải->trái theo tâm box". Input là các box frame GT (không lẫn lỗi detector).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reading_order import kendall_tau, reading_order  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--labels", type=Path, default=Path("data/reading_order_labels.json"))
ap.add_argument("--worst", type=int, default=0, help="in N trang có tau thấp nhất")
a = ap.parse_args()

labels = json.loads(a.labels.read_text(encoding="utf-8"))
pages = {k: v for k, v in labels.items() if not v["skip"] and len(v["order"]) == len(v["boxes"])}
print(f"{len(pages)} trang có nhãn ({len(labels) - len(pages)} trang bỏ qua)")


def naive(boxes):
    """Baseline: hàng từ trên xuống (tâm y), trong hàng từ phải sang trái. Không chia trang đôi."""
    return sorted(range(len(boxes)),
                  key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, -(boxes[i][0] + boxes[i][2]) / 2))


def evaluate(fn):
    rows = []
    for k, v in pages.items():
        pred = fn(v["boxes"], tuple(v["size"]))
        rows.append((k, len(v["boxes"]), kendall_tau(v["order"], pred), pred == v["order"]))
    return rows


def report(name, rows):
    taus = [r[2] for r in rows]
    print(f"{name:34s} tau mean {statistics.mean(taus):.3f} | median {statistics.median(taus):.3f} "
          f"| đúng hoàn toàn {sum(r[3] for r in rows) / len(rows):.1%}")


algo = evaluate(lambda b, s: reading_order(b, s, rtl=True))
base = evaluate(lambda b, s: naive(b))
report("reading_order (XY-cut + trang đôi)", algo)
report("baseline (y rồi phải->trái)", base)
nospread = evaluate(lambda b, s: reading_order(b, s, rtl=True, spread=False))
report("reading_order (tắt tách trang đôi)", nospread)

print("\ntheo số panel (reading_order):")
for lo, hi in ((2, 4), (5, 8), (9, 99)):
    sub = [r for r in algo if lo <= r[1] <= hi]
    if sub:
        print(f"  {lo}-{hi} panel: {len(sub)} trang | tau {statistics.mean(r[2] for r in sub):.3f} "
              f"| đúng hoàn toàn {sum(r[3] for r in sub) / len(sub):.1%}")

if a.worst:
    print(f"\n{a.worst} trang tau thấp nhất (key, số panel, tau):")
    for k, n, tau, ok in sorted(algo, key=lambda r: r[2])[:a.worst]:
        note = labels[k].get("note", "")
        print(f"  {k} n={n} tau={tau:.2f}" + (f" | ghi chú: {note}" if note else ""))
