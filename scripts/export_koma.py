"""Tạo file .mtp.json cho app dịch Koma (D:/MINI APP/manga_trans_pair) từ thư mục ảnh một chương:
model tự khoanh ô thoại, điền sẵn chữ Nhật (OCR) và đánh số theo thứ tự đọc; bản dịch để trống.

    python scripts/export_koma.py "D:/truyen/chuong1"
    python scripts/export_koma.py "D:/truyen/chuong1" --pad 10 --preview

Dùng trong Koma (không phải sửa app):
  1. Thư viện -> "Nhập .json" -> chọn file này (tạo chương mới, chưa có ảnh)
  2. Mở chương -> Xuất / Nhập -> "Gắn lại ảnh" -> chọn ĐÚNG thư mục ảnh vừa dùng (ghép theo tên file)
  3. Kiểm tra ô + chữ Nhật, rồi dịch.
CHỈ DÙNG CHO CHƯƠNG MỚI: nếu chương trùng mã đã có trong Koma, "Ghi đè" của app thay toàn bộ ô của các
trang trùng tên và sẽ xóa bản dịch có sẵn. File được ghi vào out/<tên>/ (đã gitignore), không ghi vào
thư mục truyện.
"""
import argparse
import json
import random
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ocr import iou  # noqa: E402
from src.pipeline import MangaPipeline  # noqa: E402
from src.speaker import SpeakerScorer, attribute  # noqa: E402

# Giống Koma: các đuôi ảnh nó nhận + sắp theo số trong tên file ("1, 2, 10")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp"}
DEFAULT_TRACK = "tr_main"

ap = argparse.ArgumentParser()
ap.add_argument("folder", type=Path, help="thư mục ảnh của MỘT chương")
ap.add_argument("--weights", default="weights/best.pt")
ap.add_argument("--name", default=None, help="tên chương trong Koma (mặc định: tên thư mục)")
ap.add_argument("--out", type=Path, default=None, help="mặc định out/<tên>/<tên>.mtp.json")
ap.add_argument("--pad", type=int, default=8, help="đệm quanh mỗi ô (px); ô model bó sát chữ, bạn khoanh cả ô thoại")
ap.add_argument("--ltr", action="store_true", help="đọc trái->phải (mặc định phải->trái)")
ap.add_argument("--no-ocr", action="store_true", help="không điền chữ Nhật, chỉ khoanh ô")
ap.add_argument("--order", choices=["panel", "direct"], default="panel",
                help="panel (mặc định): sắp panel rồi thoại; direct: sắp thẳng trên box chữ, bỏ qua panel "
                     "(thử khi panel bị phát hiện chồng lấn nên thứ tự sai)")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--device", default=None)
ap.add_argument("--characters", type=Path, default=None,
                help="characters.json do run_characters.py tạo cho CÙNG thư mục ảnh: bật gợi ý người nói "
                     "('[NV3]' = nhân vật 3) ghi vào ghi chú của từng ô")
ap.add_argument("--speaker-model", type=Path, default=Path("data/speaker/model.pt"))
ap.add_argument("--speaker-min-prob", type=float, default=0.6,
                help="chỉ gợi ý khi xác suất của mô hình >= mức này (ô còn lại để trống)")
ap.add_argument("--chars-max-side", type=int, default=2400, help="--max-side đã dùng khi chạy run_characters")
ap.add_argument("--preview", action="store_true", help="lưu ảnh vẽ ô cạnh file json để xem nhanh")
a = ap.parse_args()


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def uid(prefix):  # cùng dạng với uid() của Koma: prefix_ + 8 ký tự + 4 ký tự
    rnd = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{rnd}{base36(int(time.time() * 1000))[-4:]}"


def base36(n):
    digits = string.digits + string.ascii_lowercase
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out or "0"


def safe_name(s):  # giống safeName() của Koma
    return re.sub(r'[\\/:*?"<>|]+', "_", str(s)).strip() or "chuong"


if not a.folder.is_dir():
    raise SystemExit(f"Không phải thư mục: {a.folder}")
files = sorted((p for p in a.folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXT),
               key=lambda p: natural(p.name))
if not files:
    raise SystemExit(f"Không có ảnh trong: {a.folder}")

name = a.name or a.folder.name
out_dir = a.out.parent if a.out else Path("out") / safe_name(name)
out_path = a.out or out_dir / f"{safe_name(name)}.mtp.json"
out_dir.mkdir(parents=True, exist_ok=True)

chars_by_page, scorer = {}, None
if a.characters:
    cj = json.loads(a.characters.read_text(encoding="utf-8"))
    for it in cj["instances"]:
        chars_by_page.setdefault(it["page_name"], []).append(it)
    scorer = SpeakerScorer.load(a.speaker_model)


def cluster_of(ent, insts, scale):
    """Cụm nhân vật của một thực thể (khớp theo IoU>=0.5 với thực thể của run_characters); None nếu không có."""
    best, c = 0.5, None
    for it in insts:
        for key, box in (("body_box", ent["box"]), ("face_box", ent["face"])):
            if box is None or it[key] is None:
                continue
            v = iou(box, [x * scale for x in it[key]])
            if v >= best and it["cluster"] is not None:
                best, c = v, it["cluster"]
    return c


pipe = MangaPipeline(a.weights, conf=a.conf, imgsz=a.imgsz, device=a.device, ocr=not a.no_ocr)
now_ms = int(time.time() * 1000)
pages, n_boxes, n_hint, t0 = [], 0, 0, time.time()
for i, f in enumerate(files, 1):
    try:
        # exif_transpose: trình duyệt hiển thị ảnh theo EXIF, nên kích thước/toạ độ phải theo hướng đó
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    except Exception as err:  # ảnh hỏng/định dạng PIL không đọc được: giữ trang với 0 ô để số trang không lệch
        print(f"[{i:3d}] {f.name}: KHÔNG đọc được ({err}) -> trang trống", flush=True)
        pages.append({"id": uid("pg"), "fileName": f.name, "relPath": f"{a.folder.name}/{f.name}",
                      "w": 0, "h": 0, "boxes": []})
        continue
    res = pipe.run(img, rtl=not a.ltr, order=a.order)
    W, H = img.size
    hint = {}  # (box text) -> "[NVk]"
    if scorer and f.stem in chars_by_page:
        ents, attr = attribute(res["detections"], img.size, scorer)
        scale = max(W, H) / a.chars_max_side if max(W, H) > a.chars_max_side else 1.0
        for d, (ei, prob) in zip(res["detections"]["text"], attr):
            if ei is not None and prob >= a.speaker_min_prob:
                c = cluster_of(ents[ei], chars_by_page[f.stem], scale)
                if c is not None:
                    hint[tuple(d["box"])] = f"[NV{c}]"
    n_hint += len(hint)
    boxes = []
    for p in res["reading"]:
        for t in p["texts"]:  # đã theo đúng thứ tự đọc của cả trang (t["order"] tăng dần)
            x1, y1, x2, y2 = t["box"]
            x1, y1 = max(0, round(x1) - a.pad), max(0, round(y1) - a.pad)
            x2, y2 = min(W, round(x2) + a.pad), min(H, round(y2) + a.pad)
            # Koma lưu toạ độ ô CHUẨN HOÁ 0-1 theo kích thước trang (boxStyle: left = x * 100%), không phải px
            boxes.append({"id": uid("bx"), "x": round(x1 / W, 6), "y": round(y1 / H, 6),
                          "w": round((x2 - x1) / W, 6), "h": round((y2 - y1) / H, 6),
                          "jp": t["text"], "tr": {DEFAULT_TRACK: {"vi": "", "note": hint.get(tuple(t["box"]), "")}}})
    n_boxes += len(boxes)
    pages.append({"id": uid("pg"), "fileName": f.name, "relPath": f"{a.folder.name}/{f.name}",
                  "w": W, "h": H, "boxes": boxes})
    if a.preview:
        pipe.draw(img, res).save(out_dir / f"preview_{i:03d}.jpg", quality=85)
    print(f"[{i:3d}/{len(files)}] {f.name}: {len(boxes)} ô ({time.time() - t0:.0f}s)", flush=True)

payload = {
    "format": "manga_trans_pair",
    "version": 1,
    "exportedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "project": {
        "id": uid("prj"), "name": name, "createdAt": now_ms, "updatedAt": now_ms,
        "readingDir": "ltr" if a.ltr else "rtl", "info": {},
        "tracks": [{"id": DEFAULT_TRACK, "name": "Tiếng Việt"}], "activeTrack": DEFAULT_TRACK,
        "pages": pages,
    },
}
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nXong: {len(pages)} trang, {n_boxes} ô -> {out_path}")
print("Trong Koma: Thư viện -> Nhập .json -> chọn file này -> Xuất / Nhập -> Gắn lại ảnh -> chọn thư mục ảnh.")
