"""Pipeline: detect -> thứ tự đọc -> OCR -> JSON + ảnh vẽ."""
from __future__ import annotations

from PIL import Image, ImageDraw

from .ocr import crop, ocr_japanese
from .reading_order import order_page

NAMES = {0: "frame", 1: "text", 2: "face", 3: "body"}
COLORS = {"frame": "#e63946", "text": "#1d7af3", "face": "#2a9d8f", "body": "#f4a261"}


class MangaPipeline:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 1024,
                 device=None, ocr=True, ocr_fn=ocr_japanese):
        from ultralytics import YOLO  # lazy
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.ocr_fn = ocr_fn if ocr else None

    def detect(self, img: Image.Image):
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz,
                               device=self.device, verbose=False)[0]
        dets = {n: [] for n in NAMES.values()}
        for xyxy, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                              r.boxes.conf.tolist()):
            dets[NAMES[int(c)]].append({"box": xyxy, "score": s})
        return dets

    def run(self, image_path_or_pil, rtl=True):
        img = (image_path_or_pil if isinstance(image_path_or_pil, Image.Image)
               else Image.open(image_path_or_pil)).convert("RGB")
        dets = self.detect(img)
        panels = [d["box"] for d in dets["frame"]]
        texts = [d["box"] for d in dets["text"]]
        layout = order_page(panels, texts, img.size, rtl=rtl)

        reading, k = [], 1
        for pi, item in enumerate(layout):
            entries = []
            for ti in item["texts"]:
                txt = self.ocr_fn(crop(img, texts[ti])) if self.ocr_fn else ""
                entries.append({"order": k, "box": texts[ti], "text": txt})
                k += 1
            reading.append({"panel_order": pi + 1,
                            "panel_box": panels[item["panel"]] if item["panel"] is not None else None,
                            "texts": entries})
        return {"size": img.size, "detections": dets, "reading": reading}

    @staticmethod
    def draw(img: Image.Image, result) -> Image.Image:
        img = img.convert("RGB").copy()
        d = ImageDraw.Draw(img)
        for name in ("face", "body"):
            for det in result["detections"][name]:
                d.rectangle(det["box"], outline=COLORS[name], width=2)
        for p in result["reading"]:
            if p["panel_box"]:
                d.rectangle(p["panel_box"], outline=COLORS["frame"], width=4)
                d.text((p["panel_box"][0] + 6, p["panel_box"][1] + 6),
                       f"P{p['panel_order']}", fill=COLORS["frame"])
            for t in p["texts"]:
                d.rectangle(t["box"], outline=COLORS["text"], width=3)
                d.text((t["box"][0] + 3, t["box"][1] + 3), str(t["order"]),
                       fill=COLORS["text"])
        return img
