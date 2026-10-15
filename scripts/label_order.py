"""Công cụ gán nhãn thứ tự đọc panel (Gradio). Bấm vào các panel THEO THỨ TỰ ĐỌC.

    python scripts/label_order.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21"

- Đơn vị: các box `frame` thật (XML Manga109) -> đo riêng thuật toán reading_order.
- Ảnh là TRANG ĐÔI: gán thứ tự cho cả ảnh (đọc hết trang phải rồi sang trang trái).
- Mẫu: --n trang có >=2 frame, ngẫu nhiên (seed cố định) từ các bộ truyện trong --titles
  (mặc định tập test). Chạy lại sẽ tiếp tục từ trang chưa gán.
- Không hiển thị gợi ý của thuật toán để tránh làm lệch nhãn.
- Nhãn lưu ở data/reading_order_labels.json (data/ nằm trong .gitignore: chỉ có tọa độ, nhưng
  nguồn là Manga109 nên không commit).  Đo bằng: python scripts/eval_order.py
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
ap.add_argument("--titles", type=Path, default=Path("data/yolo/test_titles.txt"))
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", type=Path, default=Path("data/reading_order_labels.json"))
ap.add_argument("--port", type=int, default=7861)
args = ap.parse_args()


def build_sample():
    pages = []
    for t in args.titles.read_text().split():
        for pg in parse_book(args.root / "annotations" / f"{t}.xml"):
            frames = [b[1:5] for b in pg["boxes"] if b[0] == "frame"]
            if len(frames) >= 2:
                pages.append((t, pg["index"], pg["w"], pg["h"], frames))
    random.Random(args.seed).shuffle(pages)
    return pages[:args.n]


SAMPLE = build_sample()
KEYS = [f"{t}/{i:03d}" for t, i, *_ in SAMPLE]
LABELS = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
FONT = ImageFont.load_default(size=44)
SMALL = ImageFont.load_default(size=22)


def save():
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(LABELS, ensure_ascii=False, indent=1), encoding="utf-8")


def next_idx(start=0):
    for k in range(start, len(KEYS)):
        if KEYS[k] not in LABELS:
            return k
    return None


def render(k, picked):
    t, i, w, h, frames = SAMPLE[k]
    img = Image.open(args.root / "images" / t / f"{i:03d}.jpg").convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    for j, b in enumerate(frames):
        d.rectangle(b, outline=(120, 120, 120, 255), width=3)
    for n, j in enumerate(picked, 1):
        b = frames[j]
        d.rectangle(b, fill=(230, 57, 70, 50), outline=(230, 57, 70, 255), width=6)
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        d.ellipse((cx - 34, cy - 34, cx + 34, cy + 34), fill=(230, 57, 70, 255))
        d.text((cx, cy), str(n), fill="white", font=FONT, anchor="mm")
    return img


def status(k, picked, msg=""):
    done = sum(KEYS[x] in LABELS for x in range(len(KEYS)))
    if k is None:
        return f"Xong! Đã xử lý {done}/{len(KEYS)} trang. Nhãn lưu ở {args.out}. {msg}"
    nf = len(SAMPLE[k][4])
    return (f"Trang {k + 1}/{len(KEYS)} ({KEYS[k]}), {nf} panel | đã chọn {len(picked)}/{nf} | "
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
    frames = SAMPLE[k][4]
    hit = [j for j, b in enumerate(frames) if b[0] <= x <= b[2] and b[1] <= y <= b[3] and j not in picked]
    if not hit:
        return show(k, picked, "| (không trúng panel chưa chọn nào)")
    j = min(hit, key=lambda j: (frames[j][2] - frames[j][0]) * (frames[j][3] - frames[j][1]))
    return show(k, picked + [j])


def on_undo(k, picked):
    return show(k, picked[:-1])


def on_reset(k, picked):
    return show(k, [])


def on_save(k, picked, note):
    if k is None:
        return show(k, picked)
    t, i, w, h, frames = SAMPLE[k]
    if len(picked) != len(frames):
        return show(k, picked, f"| CHƯA LƯU: cần chọn đủ {len(frames)} panel (hoặc bấm Bỏ qua)")
    LABELS[KEYS[k]] = {"title": t, "page": i, "size": [w, h], "boxes": frames,
                       "order": picked, "skip": False, "note": note or ""}
    save()
    return show(next_idx(k + 1) if next_idx(k + 1) is not None else next_idx(), [])


def on_skip(k, picked, note):
    if k is None:
        return show(k, picked)
    t, i, w, h, frames = SAMPLE[k]
    LABELS[KEYS[k]] = {"title": t, "page": i, "size": [w, h], "boxes": frames,
                       "order": [], "skip": True, "note": note or ""}
    save()
    return show(next_idx(k + 1) if next_idx(k + 1) is not None else next_idx(), [])


with gr.Blocks(title="Gán nhãn reading order") as demo:
    gr.Markdown("Bấm vào **từng panel theo thứ tự đọc** (trang đôi: hết trang phải rồi trang trái). "
                "Bấm nhầm thì **Hoàn tác**. Panel lạ/không đọc được thứ tự -> ghi chú rồi **Bỏ qua**.")
    info = gr.Markdown()
    img = gr.Image(type="pil", interactive=False, label="Bấm để chọn panel", height=720, format="jpeg")
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
