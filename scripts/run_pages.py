"""Chạy pipeline trên truyện của bạn: một ảnh, một thư mục ảnh (một chương) hoặc file .cbz/.zip.

    python scripts/run_pages.py "D:/truyen/chuong1" --weights weights/best.pt
    python scripts/run_pages.py "D:/truyen/chuong1.cbz" --ltr --no-ocr
    python scripts/run_pages.py trang.jpg --conf 0.2

Kết quả trong out/<tên>/ (đã gitignore; ảnh chú thích là trang truyện có bản quyền -> chỉ để xem local):
  page_XXX.jpg    ảnh vẽ panel (đỏ, P#), thoại (xanh, theo thứ tự đọc), mặt/thân
  transcript.txt  thoại theo thứ tự đọc, từng trang
  result.json     toàn bộ box + thứ tự + chữ OCR
Trang đọc theo thứ tự tên file (sắp "tự nhiên": 2 trước 10). Ảnh rộng hơn cao được coi là trang đôi.
OCR chỉ có tiếng Nhật; detector vẫn chạy với chữ ngôn ngữ khác nhưng phần chữ sẽ vô nghĩa (dùng --no-ocr).
"""
import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import MangaPipeline  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ap = argparse.ArgumentParser()
ap.add_argument("input", type=Path, help="ảnh, thư mục ảnh hoặc .cbz/.zip")
ap.add_argument("--weights", default="weights/best.pt")
ap.add_argument("--out", type=Path, default=None, help="mặc định out/<tên input>")
ap.add_argument("--ltr", action="store_true", help="đọc trái->phải (mặc định phải->trái như manga Nhật)")
ap.add_argument("--no-ocr", action="store_true")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--max-side", type=int, default=2400, help="thu nhỏ ảnh lớn hơn mức này (chỉ để chạy nhanh)")
ap.add_argument("--device", default=None)
a = ap.parse_args()


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def pages(path: Path):
    """Sinh (tên, PIL.Image). Đọc thẳng từ zip trong bộ nhớ, không giải nén ra đĩa."""
    if path.is_dir():
        for f in sorted((p for p in path.iterdir() if p.suffix.lower() in IMG_EXT), key=lambda p: natural(p.name)):
            yield f.stem, Image.open(f)
    elif path.suffix.lower() in {".cbz", ".zip"}:
        with zipfile.ZipFile(path) as z:
            names = sorted((n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXT), key=natural)
            for n in names:
                yield Path(n).stem, Image.open(io.BytesIO(z.read(n)))
    elif path.suffix.lower() in IMG_EXT:
        yield path.stem, Image.open(path)
    else:
        raise SystemExit(f"Không nhận dạng được input: {path}")


out = a.out or Path("out") / a.input.stem
out.mkdir(parents=True, exist_ok=True)
pipe = MangaPipeline(a.weights, conf=a.conf, imgsz=a.imgsz, device=a.device, ocr=not a.no_ocr)

results, lines, t0, n = {}, [], time.time(), 0
for name, img in pages(a.input):
    n += 1
    img = img.convert("RGB")
    scale = 1.0
    if max(img.size) > a.max_side:
        scale = a.max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    t = time.time()
    res = pipe.run(img, rtl=not a.ltr)
    pipe.draw(img, res).save(out / f"page_{n:03d}.jpg", quality=88)
    d = res["detections"]
    ntext = sum(len(p["texts"]) for p in res["reading"])
    print(f"[{n:3d}] {name}: {len(d['frame'])} panel, {ntext} thoại, {time.time() - t:.1f}s", flush=True)
    lines.append(f"=== trang {n}: {name} ===")
    lines += [f"[{tx['order']}] {tx['text']}" for p in res["reading"] for tx in p["texts"]] or ["(không có thoại)"]
    lines.append("")
    results[f"{n:03d}_{name}"] = dict(res, scale=scale)

(out / "transcript.txt").write_text("\n".join(lines), encoding="utf-8")
(out / "result.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Xong {n} trang trong {time.time() - t0:.0f}s -> {out}")
