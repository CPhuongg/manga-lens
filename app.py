"""Demo Gradio: tải một trang manga -> panel/text/face/body, thứ tự đọc, OCR thoại.

    python app.py                              # dùng weights/best.pt
    python app.py --weights path/to/best.pt --no-ocr

Chạy được trên Hugging Face Spaces (không cần đối số). Ảnh người dùng chỉ xử lý trong bộ nhớ, không
lưu; file tạm của Gradio bị xóa sau ~1 phút.
"""
import argparse
import os

import gradio as gr
from PIL import Image

from src.pipeline import MangaPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=os.environ.get("MANGA_LENS_WEIGHTS", "weights/best.pt"))
ap.add_argument("--no-ocr", action="store_true")
ap.add_argument("--port", type=int, default=None)
args = ap.parse_args()

MAX_SIDE = 2400  # ảnh quá lớn làm chậm vô ích (model chạy imgsz <= 1024)
pipe = MangaPipeline(args.weights, ocr=not args.no_ocr)
if not args.no_ocr:  # nạp model OCR sẵn để lần bấm đầu không bị chờ
    from src.ocr import ocr_japanese
    ocr_japanese(Image.new("RGB", (48, 48), "white"))


def run(img, rtl, conf, imgsz, do_ocr):
    if img is None:
        raise gr.Error("Hãy tải lên một trang manga.")
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)))
    pipe.conf, pipe.imgsz = conf, int(imgsz)
    saved = pipe.ocr_fn
    if not do_ocr:
        pipe.ocr_fn = None
    try:
        res = pipe.run(img, rtl=rtl)
    finally:
        pipe.ocr_fn = saved
    lines = [f"[{t['order']}] (panel {p['panel_order']}) {t['text']}"
             for p in res["reading"] for t in p["texts"]]
    d = res["detections"]
    info = (f"{len(d['frame'])} panel, {len(d['text'])} text, {len(d['face'])} mặt, {len(d['body'])} thân"
            f" | ảnh {img.width}x{img.height}")
    return pipe.draw(img, res), "\n".join(lines) or "(không phát hiện thoại)", info


with gr.Blocks(title="manga-lens", delete_cache=(60, 60)) as demo:
    gr.Markdown(
        "# manga-lens\nPhát hiện panel / chữ / mặt / thân, sắp **thứ tự đọc** (trang đôi, phải→trái) và "
        "**OCR thoại tiếng Nhật**. Mô hình YOLO11s huấn luyện trên Manga109 (học thuật).\n\n"
        "Chỉ tải ảnh bạn có quyền sử dụng. Ảnh không được lưu lại. Trên CPU mỗi trang mất vài chục giây "
        "(OCR chiếm phần lớn); tắt OCR hoặc giảm độ phân giải để nhanh hơn.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Trang manga", sources=["upload", "clipboard"])
            rtl = gr.Checkbox(True, label="Đọc phải→trái")
            do_ocr = gr.Checkbox(not args.no_ocr, label="Bật OCR", interactive=not args.no_ocr)
            conf = gr.Slider(0.1, 0.6, value=0.25, step=0.05, label="Ngưỡng conf (0.25 là mặc định đã đo)")
            imgsz = gr.Radio([640, 800, 1024], value=1024, label="Độ phân giải (cao = chính xác hơn, chậm hơn)")
            btn = gr.Button("Phân tích", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Kết quả (panel đỏ P#, thoại xanh theo thứ tự đọc)")
            info = gr.Markdown()
            out_txt = gr.Textbox(label="Thoại theo thứ tự đọc", lines=12)
    btn.click(run, [inp, rtl, conf, imgsz, do_ocr], [out_img, out_txt, info])

if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_port=args.port)
