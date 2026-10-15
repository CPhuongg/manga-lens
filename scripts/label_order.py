"""Công cụ gán nhãn thứ tự đọc (Gradio). Bấm vào các box THEO THỨ TỰ ĐỌC.

    # thứ tự THOẠI (chuẩn đo chính của sản phẩm; mặc định)
    python scripts/label_order.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21"
    # thứ tự PANEL (chỉ để chẩn đoán: lỗi do sắp panel hay sắp thoại)
    python scripts/label_order.py --root ... --mode panel

- Đơn vị: box GT trong XML Manga109 (`text` hoặc `frame`) -> đo riêng thuật toán sắp thứ tự.
- Ảnh là TRANG ĐÔI: gán thứ tự cho cả ảnh (đọc hết trang phải rồi sang trang trái).
- Mẫu: --n trang có >=2 box, ngẫu nhiên (seed cố định) từ các bộ truyện trong --titles
  (mặc định tập test). Chạy lại sẽ tiếp tục từ trang chưa gán.
- Không hiển thị gợi ý của thuật toán để tránh làm lệch nhãn.
- Nhãn lưu ở data/reading_order_{text_,}labels.json (data/ nằm trong .gitignore: chỉ có tọa độ,
  nhưng nguồn là Manga109 nên không commit).  Đo bằng: python scripts/eval_order.py --mode ...
"""
import argparse
import json
import random
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_manga109 import parse_book  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True, type=Path)
ap.add_argument("--mode", choices=["text", "panel"], default="text")
ap.add_argument("--titles", type=Path, default=Path("data/yolo/test_titles.txt"))
ap.add_argument("--n", type=int, default=None, help="số trang (mặc định 30 cho text, 50 cho panel)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--port", type=int, default=7861)
args = ap.parse_args()

KIND = "text" if args.mode == "text" else "frame"
N = args.n or (30 if args.mode == "text" else 50)
OUT = args.out or Path("data/reading_order_text_labels.json" if args.mode == "text"
                       else "data/reading_order_labels.json")
PAD = 6 if args.mode == "text" else 0  # dung sai khi bấm: box chữ nhỏ/hẹp


def build_sample():
    """(title, page, w, h, items, ctx): items = box cần sắp thứ tự, ctx = các frame (chỉ để vẽ/đo)."""
    pages = []
    for t in args.titles.read_text().split():
        for pg in parse_book(args.root / "annotations" / f"{t}.xml"):
            items = [b[1:5] for b in pg["boxes"] if b[0] == KIND]
            ctx = [b[1:5] for b in pg["boxes"] if b[0] == "frame"] if KIND == "text" else []
            if len(items) >= 2:
                pages.append((t, pg["index"], pg["w"], pg["h"], items, ctx))
    random.Random(args.seed).shuffle(pages)
    return pages[:N]


SAMPLE = build_sample()
KEYS = [f"{t}/{i:03d}" for t, i, *_ in SAMPLE]
LABELS = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
_FONTS = {}


def font(size):
    if size not in _FONTS:
        _FONTS[size] = ImageFont.load_default(size=size)
    return _FONTS[size]


def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(LABELS, ensure_ascii=False, indent=1), encoding="utf-8")


def next_idx(start=0):
    for k in range(start, len(KEYS)):
        if KEYS[k] not in LABELS:
            return k
    return None


def render(k, picked):
    t, i, w, h, items, ctx = SAMPLE[k]
    img = Image.open(args.root / "images" / t / f"{i:03d}.jpg").convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    for b in ctx:  # ở chế độ text: vẽ mờ ranh giới panel để dễ định hướng
        d.rectangle(b, outline=(40, 120, 220, 150), width=2)
    for b in items:
        d.rectangle(b, outline=(120, 120, 120, 255), width=2 if args.mode == "text" else 3)
    for n, j in enumerate(picked, 1):
        b = items[j]
        d.rectangle(b, fill=(230, 57, 70, 60), outline=(230, 57, 70, 255), width=4)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        r = int(max(11, min(34, 0.45 * min(b[2] - b[0], b[3] - b[1]))))  # số không che hết box nhỏ
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(230, 57, 70, 255))
        d.text((cx, cy), str(n), fill="white", font=font(max(14, int(r * 1.3))), anchor="mm")
    return img


def status(k, picked, msg=""):
    done = sum(KEYS[x] in LABELS for x in range(len(KEYS)))
    if k is None:
        return f"Xong! Đã xử lý {done}/{len(KEYS)} trang. Nhãn lưu ở {OUT}. {msg}"
    n = len(SAMPLE[k][4])
    return (f"[{args.mode}] Trang {k + 1}/{len(KEYS)} ({KEYS[k]}), {n} box | đã chọn {len(picked)}/{n} | "
            f"đã xử lý {done} {msg}")


def show(k, picked, msg=""):
    if k is None:
        return None, k, picked, status(k, picked, msg)
    return render(k, picked), k, picked, status(k, picked, msg)


def on_load():
    return show(next_idx(), [])


def on_click(k, picked, evt: gr.SelectData):
    if k is None:
        return show(k, picked)
    x, y = evt.index
    items = SAMPLE[k][4]

    def hit(pad):
        return [j for j, b in enumerate(items) if j not in picked
                and b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad]

    cand = hit(0) or hit(PAD)  # ưu tiên trúng hẳn, rồi mới đến vùng dung sai
    if not cand:
        return show(k, picked, "| (không trúng box chưa chọn nào)")
    j = min(cand, key=lambda j: (items[j][2] - items[j][0]) * (items[j][3] - items[j][1]))
    return show(k, picked + [j])


def on_undo(k, picked):
    return show(k, picked[:-1])


def on_reset(k, picked):
    return show(k, [])


def _record(k, picked, note, skip):
    t, i, w, h, items, ctx = SAMPLE[k]
    LABELS[KEYS[k]] = {"title": t, "page": i, "size": [w, h], "boxes": items,
                       "order": [] if skip else picked, "skip": skip, "note": note or ""}
    if ctx:
        LABELS[KEYS[k]]["frames"] = ctx
    save()
    nxt = next_idx(k + 1)
    return show(nxt if nxt is not None else next_idx(), [])


def on_save(k, picked, note):
    if k is None:
        return show(k, picked)
    n = len(SAMPLE[k][4])
    if len(picked) != n:
        return show(k, picked, f"| CHƯA LƯU: cần chọn đủ {n} box (hoặc bấm Bỏ qua)")
    return _record(k, picked, note, False)


def on_skip(k, picked, note):
    if k is None:
        return show(k, picked)
    return _record(k, picked, note, True)


what = "từng box thoại" if args.mode == "text" else "từng panel"
with gr.Blocks(title="Gán nhãn reading order") as demo:
    gr.Markdown(f"Bấm vào **{what} theo thứ tự đọc** (trang đôi: hết trang phải rồi trang trái). "
                "Bấm nhầm thì **Hoàn tác**. Layout lạ/không đọc được thứ tự -> ghi chú rồi **Bỏ qua**.")
    info = gr.Markdown()
    img = gr.Image(type="pil", interactive=False, label="Bấm để chọn", height=720, format="jpeg")
    note = gr.Textbox(label="Ghi chú (tuỳ chọn)", lines=1)
    with gr.Row():
        b_undo = gr.Button("Hoàn tác")
        b_reset = gr.Button("Chọn lại từ đầu")
        b_save = gr.Button("Lưu & trang tiếp", variant="primary")
        b_skip = gr.Button("Bỏ qua trang")
    k_state, picked_state = gr.State(None), gr.State([])
    outs = [img, k_state, picked_state, info]
    # trigger_mode="multiple": mặc định Gradio BỎ các lần bấm khi sự kiện trước chưa xong -> mất click;
    # show_progress="hidden": bỏ lớp phủ tải che ảnh. State nằm phía server nên xử lý tuần tự đúng thứ tự.
    fast = dict(trigger_mode="multiple", show_progress="hidden")
    demo.load(on_load, None, outs)
    img.select(on_click, [k_state, picked_state], outs, **fast)
    b_undo.click(on_undo, [k_state, picked_state], outs, **fast)
    b_reset.click(on_reset, [k_state, picked_state], outs, **fast)
    b_save.click(on_save, [k_state, picked_state, note], outs, **fast).then(lambda: "", None, note)
    b_skip.click(on_skip, [k_state, picked_state, note], outs, **fast).then(lambda: "", None, note)

demo.launch(server_port=args.port)
