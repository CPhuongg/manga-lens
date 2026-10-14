"""Demo Gradio:  python app.py --weights runs/yolo11s/weights/best.pt"""
import argparse

import gradio as gr

from src.pipeline import MangaPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--no-ocr", action="store_true")
args = ap.parse_args()

pipe = MangaPipeline(args.weights, ocr=not args.no_ocr)


def run(img, rtl):
    res = pipe.run(img, rtl=rtl)
    lines = [f"[{t['order']}] (panel {p['panel_order']}) {t['text']}"
             for p in res["reading"] for t in p["texts"]]
    return pipe.draw(img, res), "\n".join(lines)


gr.Interface(
    run,
    [gr.Image(type="pil", label="Trang manga"), gr.Checkbox(True, label="Đọc phải→trái")],
    [gr.Image(label="Kết quả"), gr.Textbox(label="Thoại theo thứ tự đọc", lines=12)],
    title="manga-lens",
).launch()
