"""Worker cho app dịch Koma (D:/MINI APP/koma): serve.py của Koma gọi script này để phân tích MỘT HAY NHIỀU chương
(cả bộ truyện) một lượt: khoanh ô thoại + thứ tự đọc, OCR chữ Nhật, gom nhân vật (re-ID theo mặt) XUYÊN các chương,
và gợi ý người nói cho từng ô.

    .venv/Scripts/python.exe scripts/koma_lens.py --job <thư mục job>

Thư mục job do Koma tạo: manifest.json + ảnh trang. Kết quả ghi vào result.json; tiến độ in ra stdout từng dòng
"@@{json}" (serve.py đọc để báo lên app). Không tự chạy tay cũng được, nhưng định dạng manifest do Koma quyết định:

  {"options": {"ocr": true, "characters": true, "mode": "face"|"fused", "thr": null, "minSize": 3,
               "pad": 8, "minProb": 0.6, "order": "panel"|"direct"},
   "chapters": [{"key": "<id chương>", "rtl": true,
                 "pages": [{"key": "<id trang>", "file": "img/c0_p0.jpg",
                            "boxes": null | [{"id": "bx_..", "x":0-1, "y", "w", "h", "ocr": true|false}]}]}]}

boxes = null: model tự khoanh ô và sắp thứ tự. boxes = list: GIỮ ô của người dùng (đúng thứ tự đó), chỉ OCR ô có
"ocr": true và gán người nói. Toạ độ vào/ra đều CHUẨN HOÁ 0-1 theo kích thước trang (như Koma lưu).

Nhân vật: gom trên toàn bộ các trang của job (nên cả bộ truyện dùng chung một danh sách nhân vật). Ngưỡng mặc định
lấy từ run_characters.py (chọn trên val Manga109, không kiểm chứng được trên truyện thật). Re-ID hay CHIA THỪA cụm
(~1,4x số nhân vật thật) -> Koma có nút gộp nhân vật.
"""
import argparse
import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_THR = {"face": 0.4, "fused": 0.5}    # như run_characters.py
WEIGHTS = {"face": (1.0, 0.0), "fused": (0.7, 0.3)}
N_FACES = 6                                   # số mặt tiêu biểu gửi về cho mỗi nhân vật (lưu trong chương)


def emit(**kw):
    print("@@" + json.dumps(kw, ensure_ascii=False), flush=True)


def jpeg_data_url(arr, size=80):
    im = Image.fromarray(arr).resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def norm_box(b, W, H, pad=0):
    x1, y1 = max(0, round(b[0]) - pad), max(0, round(b[1]) - pad)
    x2, y2 = min(W, round(b[2]) + pad), min(H, round(b[3]) + pad)
    return {"x": round(x1 / W, 6), "y": round(y1 / H, 6), "w": round((x2 - x1) / W, 6), "h": round((y2 - y1) / H, 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=Path, required=True)
    ap.add_argument("--weights", default=str(ROOT / "weights/best.pt"))
    ap.add_argument("--face-reid", type=Path, default=ROOT / "data/reid/f_b/best.pt")
    ap.add_argument("--body-reid", type=Path, default=ROOT / "data/reid/b_a/best.pt")
    ap.add_argument("--speaker-model", type=Path, default=ROOT / "data/speaker/model.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    a = ap.parse_args()

    man = json.loads((a.job / "manifest.json").read_text(encoding="utf-8"))
    opt = man.get("options", {})
    do_ocr = bool(opt.get("ocr", True))
    do_chars = bool(opt.get("characters", True))
    mode = opt.get("mode") if opt.get("mode") in WEIGHTS else "face"
    thr = float(opt["thr"]) if opt.get("thr") is not None else DEFAULT_THR[mode]
    min_size = int(opt.get("minSize", 3))
    pad = int(opt.get("pad", 8))
    min_prob = float(opt.get("minProb", 0.6))
    order_mode = opt.get("order") if opt.get("order") in ("panel", "direct") else "panel"
    need_body = mode == "fused"
    total = sum(len(c["pages"]) for c in man["chapters"])

    emit(stage="load", msg="Đang nạp mô hình…", done=0, total=total)
    import torch
    from eval_cluster import cut
    from eval_fused import distance, link, linkage_gpu
    from run_characters import crop as crop_char
    from src.ocr import crop as crop_ocr, ocr_japanese
    from src.pipeline import MangaPipeline
    from src.reading_order import order_page, reading_order
    from src.speaker import SpeakerScorer, attribute

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = MangaPipeline(a.weights, conf=a.conf, imgsz=a.imgsz, ocr=False)
    scorer = None
    if do_chars:
        if a.speaker_model.exists():
            scorer = SpeakerScorer.load(a.speaker_model)
        else:
            emit(stage="warn", msg=f"Không có {a.speaker_model.name}: bỏ qua gợi ý người nói")

    # ---- 1. từng trang: detect -> ô thoại (+ thứ tự) -> OCR -> người nói (thực thể) -> crop mặt/thân
    out_chapters = []
    inst = []                 # mỗi thực thể re-ID: {ci, pi}
    crops_f, crops_b = [], []
    t0, done = time.time(), 0
    for ci, ch in enumerate(man["chapters"]):
        rtl = ch.get("rtl", True)
        out_pages = []
        for pi, pg in enumerate(ch["pages"]):
            done += 1
            rec = {"key": pg["key"], "boxes": []}
            out_pages.append(rec)
            try:
                img = ImageOps.exif_transpose(Image.open(a.job / pg["file"])).convert("RGB")
            except Exception as err:
                rec["error"] = f"không đọc được ảnh: {err}"
                emit(stage="pages", done=done, total=total, msg=f"Trang lỗi ({err})")
                continue
            W, H = img.size
            rec["w"], rec["h"] = W, H
            dets = pipe.detect(img)

            # Ô thoại: của người dùng (giữ thứ tự) hoặc do model khoanh + sắp thứ tự đọc
            user = pg.get("boxes")
            if user is not None:
                texts = [[b["x"] * W, b["y"] * H, (b["x"] + b["w"]) * W, (b["y"] + b["h"]) * H] for b in user]
                for b, t in zip(user, texts):
                    rec["boxes"].append({"id": b["id"], "_box": t, "_ocr": bool(b.get("ocr"))})
            else:
                det_texts = [d["box"] for d in dets["text"]]
                if order_mode == "direct":
                    seq = reading_order(det_texts, img.size, rtl=rtl) if det_texts else []
                else:
                    seq = [ti for item in order_page([d["box"] for d in dets["frame"]], det_texts, img.size, rtl=rtl)
                           for ti in item["texts"]]
                texts = [det_texts[ti] for ti in seq]
                for t in texts:
                    rec["boxes"].append({**norm_box(t, W, H, pad), "_box": t, "_ocr": True})

            if do_ocr:
                for b in rec["boxes"]:
                    if b["_ocr"]:
                        b["jp"] = ocr_japanese(crop_ocr(img, b["_box"]))

            if do_chars:
                fb, bb = [d["box"] for d in dets["face"]], [d["box"] for d in dets["body"]]
                face_inst, body_inst = {}, {}
                for fi, bi in link(fb, bb):
                    f_c = crop_char(img, fb[fi], "face") if fi is not None else None
                    b_c = crop_char(img, bb[bi], "body") if (bi is not None and need_body) else None
                    k = len(inst)
                    inst.append({"ci": ci, "pi": pi})
                    crops_f.append(f_c)
                    crops_b.append(b_c)
                    if fi is not None:
                        face_inst[fi] = k
                    if bi is not None:
                        body_inst.setdefault(bi, k)
                if scorer and texts:
                    tdets = dict(dets, text=[{"box": t} for t in texts])
                    ents, attr = attribute(tdets, img.size, scorer)
                    for b, (ei, prob) in zip(rec["boxes"], attr):
                        if ei is None:
                            continue
                        e = ents[ei]
                        k = None
                        if e["face"] is not None:   # thực thể có mặt -> thực thể re-ID của mặt đó
                            j = next((j for j, f in enumerate(fb) if list(f) == e["face"]), None)
                            k = face_inst.get(j)
                        if k is None and e["src"][0] == "body":
                            k = body_inst.get(e["src"][1])
                        if k is not None:
                            b["_inst"], b["prob"] = k, round(prob, 3)
            emit(stage="pages", done=done, total=total,
                 msg=f"{len(rec['boxes'])} ô · {time.time() - t0:.0f}s")
        out_chapters.append({"key": ch["key"], "pages": out_pages})

    # ---- 2. gom nhân vật trên toàn job
    characters = []
    cluster = [None] * len(inst)
    if do_chars and inst:
        emit(stage="embed", msg=f"Đang nhận dạng mặt ({len(inst)} lần nhân vật xuất hiện)…", done=total, total=total)
        del pipe
        if dev == "cuda":
            torch.cuda.empty_cache()
        from train_reid import ReID, embed_all, load_ckpt

        def embed(crops, ckpt):
            idx = [i for i, c in enumerate(crops) if c is not None]
            E = np.zeros((len(crops), 256), np.float32)
            if idx:
                m = ReID(emb=256).to(dev)
                load_ckpt(m, ckpt, dev)
                E[idx] = embed_all(m, np.stack([crops[i] for i in idx]), dev)
                del m
            return E, np.array([c is not None for c in crops])

        Ef, hf = embed(crops_f, a.face_reid)
        Eb, hb = embed(crops_b, a.body_reid) if need_body else (np.zeros_like(Ef), np.zeros(len(inst), bool))
        has_id = hf if mode == "face" else (hf | hb)
        n = len(inst)
        if int(has_id.sum()) >= 2:
            emit(stage="cluster", msg="Đang gom nhân vật…", done=total, total=total)
            wf, wb = WEIGHTS[mode]
            D = distance(dict(Ef=Ef, Eb=Eb, hf=hf, hb=hb), wf, wb, 1.0)
            lab = cut(linkage_gpu(D), n, thr)
            ids, cnt = np.unique(lab[has_id], return_counts=True)
            keep = [i for i, c in sorted(zip(ids, cnt), key=lambda x: -x[1]) if c >= min_size]
            for rank, cid in enumerate(keep, 1):
                ii = [i for i in range(n) if lab[i] == cid and has_id[i]]
                for i in ii:
                    cluster[i] = rank
                fi = [i for i in ii if hf[i]]
                faces = []
                if fi:   # mặt tiêu biểu: gần tâm cụm nhất
                    cen = Ef[fi].mean(0)
                    cen /= np.linalg.norm(cen) + 1e-9
                    faces = [jpeg_data_url(crops_f[fi[j]]) for j in np.argsort(-(Ef[fi] @ cen))[:N_FACES]]
                pages = sorted({(inst[i]["ci"], inst[i]["pi"]) for i in ii})
                characters.append({"k": rank, "n": len(ii), "pages": len(pages), "faces": faces})

    # ---- 3. ghi kết quả (bỏ trường nội bộ)
    n_box = n_spk = 0
    for ch in out_chapters:
        for pg in ch["pages"]:
            for b in pg["boxes"]:
                k = b.pop("_inst", None)
                b.pop("_box", None)
                b.pop("_ocr", None)
                c = cluster[k] if k is not None else None
                if c is not None and b.get("prob", 0) >= min_prob:
                    b["speaker"] = c
                    n_spk += 1
                n_box += 1
    result = {"chapters": out_chapters, "characters": characters,
              "stats": {"pages": total, "boxes": n_box, "speakers": n_spk, "instances": len(inst),
                        "mode": mode, "thr": thr, "seconds": round(time.time() - t0, 1), "device": dev}}
    tmp = a.job / "result.json.tmp"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(a.job / "result.json")
    emit(stage="done", done=total, total=total,
         msg=f"Xong: {total} trang, {n_box} ô, {len(characters)} nhân vật, {n_spk} ô có người nói")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        traceback.print_exc()
        emit(stage="error", msg=f"{type(err).__name__}: {err}")
        sys.exit(1)
