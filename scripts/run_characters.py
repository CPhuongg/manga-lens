"""Gom nhân vật trong một chương truyện: detect mặt/thân -> embedding re-ID -> gom cụm -> "nhân vật 1, 2, 3...".

    python scripts/run_characters.py "D:/truyen/chuong1" --draw
    python scripts/run_characters.py "D:/truyen/chuong1.cbz" --mode fused
    python scripts/run_characters.py "D:/truyen/chuong1" --sweep 0.3 0.4 0.5 0.6   # xem số cụm theo ngưỡng, không ghi gì
    python scripts/run_characters.py "D:/truyen/chuong1" --thr 0.5 --min-size 4

Input giống scripts/run_pages.py (ảnh, thư mục, .cbz/.zip, sắp tự nhiên theo tên). Cần:
  weights/best.pt (detector), data/reid/f_b/best.pt (face), data/reid/b_a/best.pt (body; chỉ khi --mode fused),
  facebook/dinov2-small (tải tự động lần đầu).

Kết quả trong out/<tên>/characters/ (đã gitignore; ảnh là trang truyện có bản quyền -> chỉ để xem local):
  characters.txt   mỗi nhân vật: số lần xuất hiện + các trang; cụm quá nhỏ gom vào "chưa gán"
  characters.json  mọi thực thể (box mặt/thân, cụm) + tóm tắt từng nhân vật (để nối với export_koma sau này)
  sheet_XX.jpg     ảnh ghép các mặt tiêu biểu của nhân vật XX (để bạn đặt tên)
  page_XXX.jpg     (--draw) trang gốc vẽ mặt + nhãn nhân vật

Chế độ: face (mặc định) chỉ gom theo MẶT, thực thể chỉ có thân để "chưa gán"; fused dùng cả thân (trọng số
mặt 0.7 / thân 0.3), bù được trang detector không bắt được mặt nhưng nhiều nhầm hơn ở vài bộ.
Số đo trên Manga109 test (box detector, 11 bộ, BCubed F1 end-to-end): face 0.59, fused 0.57 — chênh nằm trong sai
số; ARI/pairwise F1 của fused cao hơn. Ngưỡng mặc định lấy từ val, KHÔNG được kiểm chứng trên truyện của bạn
(không có nhãn): dùng --sweep để xem độ nhạy, rồi xem sheet_XX.jpg để biết có gộp nhầm / tách thừa không.
"""
import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_cluster import cut  # noqa: E402
from eval_fused import distance, linkage_gpu, link  # noqa: E402
from src.pipeline import MangaPipeline  # noqa: E402
from train_reid import SIZE, ReID, embed_all, load_ckpt  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_THR = {"face": 0.4, "fused": 0.5}   # chọn trên val Manga109 (xem docstring)
WEIGHTS = {"face": (1.0, 0.0), "fused": (0.7, 0.3)}


def natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def pages(path: Path):
    if path.is_dir():
        for f in sorted((p for p in path.iterdir() if p.suffix.lower() in IMG_EXT), key=lambda p: natural(p.name)):
            yield f.stem, Image.open(f)
    elif path.suffix.lower() in {".cbz", ".zip"}:
        with zipfile.ZipFile(path) as z:
            for n in sorted((n for n in z.namelist() if Path(n).suffix.lower() in IMG_EXT), key=natural):
                yield Path(n).stem, Image.open(io.BytesIO(z.read(n)))
    elif path.suffix.lower() in IMG_EXT:
        yield path.stem, Image.open(path)
    else:
        raise SystemExit(f"Không nhận dạng được input: {path}")


def crop(img, box, cls, margin=0.1):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    b = (max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h)),
         min(img.width, int(x2 + margin * w)), min(img.height, int(y2 + margin * h)))
    if b[2] - b[0] < 2 or b[3] - b[1] < 2:
        return None
    H, W = SIZE[cls]
    return np.asarray(img.crop(b).resize((W, H), Image.BILINEAR, reducing_gap=2.0))


def sheet(imgs, cols=6, cell=96):
    """Ghép tối đa 12 mặt thành 1 ảnh để người xem đặt tên."""
    rows = (len(imgs) + cols - 1) // cols
    S = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, im in enumerate(imgs):
        S.paste(Image.fromarray(im).resize((cell, cell)), ((i % cols) * cell, (i // cols) * cell))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--weights", default="weights/best.pt", help="detector")
    ap.add_argument("--face-reid", default="data/reid/f_b/best.pt", type=Path)
    ap.add_argument("--body-reid", default="data/reid/b_a/best.pt", type=Path)
    ap.add_argument("--mode", default="face", choices=["face", "fused"])
    ap.add_argument("--thr", type=float, default=None, help="ngưỡng cắt (mặc định theo --mode)")
    ap.add_argument("--min-size", type=int, default=3, help="cụm nhỏ hơn mức này -> 'chưa gán'")
    ap.add_argument("--sweep", type=float, nargs="+", default=None, help="chỉ in số cụm theo các ngưỡng này rồi thoát")
    ap.add_argument("--draw", action="store_true", help="lưu trang gốc có vẽ mặt + nhãn nhân vật")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-side", type=int, default=2400)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    out = (a.out or Path("out") / a.input.stem) / "characters"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det = MangaPipeline(a.weights, conf=a.conf, imgsz=a.imgsz, device=a.device, ocr=False)
    need_body = a.mode == "fused"

    # ---- 1. detect + ghép mặt-thân + crop
    t0 = time.time()
    names, imgs = [], []
    inst = []          # dict(page, fbox, bbox, fscore, bscore)
    crops_f, crops_b = [], []
    for n, (name, im) in enumerate(pages(a.input), 1):
        im = im.convert("RGB")
        if max(im.size) > a.max_side:
            s = a.max_side / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)))
        d = det.detect(im)
        fb, bb = [x["box"] for x in d["face"]], [x["box"] for x in d["body"]]
        names.append(name); imgs.append(im if a.draw else None)
        for fi, bi in link(fb, bb):
            f_c = crop(im, fb[fi], "face") if fi is not None else None
            b_c = crop(im, bb[bi], "body") if (bi is not None and need_body) else None
            if f_c is None and b_c is None and not (bi is not None and not need_body):
                continue
            inst.append(dict(page=n, fbox=fb[fi] if fi is not None else None, bbox=bb[bi] if bi is not None else None,
                             fscore=d["face"][fi]["score"] if fi is not None else None,
                             bscore=d["body"][bi]["score"] if bi is not None else None,
                             face_ok=f_c is not None, body_ok=b_c is not None))
            crops_f.append(f_c); crops_b.append(b_c)
        print(f"[{n:3d}] {name}: {len(d['face'])} mặt, {len(d['body'])} thân", flush=True)
    npages = len(names)
    print(f"detect xong {npages} trang, {len(inst)} thực thể, {time.time() - t0:.0f}s", flush=True)
    if not inst:
        raise SystemExit("Không phát hiện được thực thể nào.")

    # ---- 2. embedding
    def embed(cls, crops, ckpt):
        idx = [i for i, c in enumerate(crops) if c is not None]
        E = np.zeros((len(crops), 256), np.float32)
        if idx:
            m = ReID(emb=256).to(dev)
            load_ckpt(m, ckpt, dev)
            E[idx] = embed_all(m, np.stack([crops[i] for i in idx]), dev)
        return E, np.array([c is not None for c in crops])

    Ef, hf = embed("face", crops_f, a.face_reid)
    Eb, hb = embed("body", crops_b, a.body_reid) if need_body else (np.zeros_like(Ef), np.zeros(len(inst), bool))
    book = dict(Ef=Ef, Eb=Eb, hf=hf, hb=hb)
    wf, wb = WEIGHTS[a.mode]
    D = distance(book, wf, wb, 1.0)
    merges = linkage_gpu(D)
    n = len(inst)
    has_id = hf if a.mode == "face" else (hf | hb)   # thực thể có thông tin để gom

    def labels(thr):
        lab = cut(merges, n, thr)
        # đếm cụm theo thực thể có thông tin; cụm nhỏ hoặc không có thông tin -> -1
        out_l = np.full(n, -1)
        ids, cnt = np.unique(lab[has_id], return_counts=True)
        keep = [i for i, c in sorted(zip(ids, cnt), key=lambda x: -x[1]) if c >= a.min_size]
        for rank, cid in enumerate(keep):
            out_l[(lab == cid) & has_id] = rank
        return out_l

    if a.sweep:
        print(f"\nmode {a.mode}, {int(has_id.sum())}/{n} thực thể có thông tin để gom, min-size {a.min_size}")
        print(f"{'ngưỡng':>7s} {'nhân vật':>9s} {'chưa gán':>9s} {'lớn nhất':>9s}")
        for thr in a.sweep:
            l = labels(thr)
            k = l.max() + 1
            sizes = np.bincount(l[l >= 0]) if k else np.array([0])
            print(f"{thr:7.2f} {k:9d} {int((l < 0).sum()):9d} {int(sizes.max()):9d}")
        return

    thr = a.thr if a.thr is not None else DEFAULT_THR[a.mode]
    lab = labels(thr)
    K = int(lab.max()) + 1
    for i, it in enumerate(inst):
        it["cluster"] = int(lab[i]) + 1 if lab[i] >= 0 else None   # 1-based; None = chưa gán

    # ---- 3. ghi kết quả
    out.mkdir(parents=True, exist_ok=True)
    chars = {}
    for c in range(1, K + 1):
        ii = [i for i in range(n) if inst[i]["cluster"] == c]
        pg = sorted({inst[i]["page"] for i in ii})
        # mặt tiêu biểu: gần tâm cụm nhất (trong các thực thể có mặt)
        fi = [i for i in ii if hf[i]]
        rep = []
        if fi:
            cen = Ef[fi].mean(0); cen /= np.linalg.norm(cen) + 1e-9
            rep = [fi[j] for j in np.argsort(-(Ef[fi] @ cen))[:12]]
        chars[c] = dict(n=len(ii), n_pages=len(pg), pages=pg, rep=rep)
        if rep:
            sheet([crops_f[i] for i in rep]).save(out / f"sheet_{c:02d}.jpg", quality=88)
    unassigned = [i for i in range(n) if inst[i]["cluster"] is None]

    lines = [f"Chương: {a.input.name} | {npages} trang | mode {a.mode} | ngưỡng {thr} | min-size {a.min_size}",
             f"{K} nhân vật (cụm >= {a.min_size} thực thể), {len(unassigned)}/{n} thực thể chưa gán", ""]
    for c, v in chars.items():
        lines.append(f"Nhân vật {c}: {v['n']} lần, {v['n_pages']} trang: " + ", ".join(map(str, v["pages"])))
    (out / "characters.txt").write_text("\n".join(lines), encoding="utf-8")

    def rnd(b):
        return [round(x, 1) for x in b] if b else None
    (out / "characters.json").write_text(json.dumps(dict(
        params=dict(mode=a.mode, thr=thr, min_size=a.min_size, conf=a.conf, pages=names),
        characters={str(c): dict(n=v["n"], pages=v["pages"]) for c, v in chars.items()},
        instances=[dict(page=it["page"], page_name=names[it["page"] - 1], face_box=rnd(it["fbox"]), body_box=rnd(it["bbox"]),
                        face_score=it["fscore"], body_score=it["bscore"], cluster=it["cluster"]) for it in inst],
    ), ensure_ascii=False, indent=1), encoding="utf-8")

    if a.draw:
        pal = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#9a6324", "#469990"]
        for p in range(1, npages + 1):
            im = imgs[p - 1].copy()
            dr = ImageDraw.Draw(im)
            for it in inst:
                if it["page"] != p:
                    continue
                col = pal[(it["cluster"] - 1) % len(pal)] if it["cluster"] else "#888888"
                for b, w in ((it["bbox"], 2), (it["fbox"], 4)):
                    if b:
                        dr.rectangle(b, outline=col, width=w)
                if it["fbox"]:
                    dr.text((it["fbox"][0] + 3, it["fbox"][1] + 3), f"C{it['cluster']}" if it["cluster"] else "?", fill=col)
            im.save(out / f"page_{p:03d}.jpg", quality=88)
    print("\n".join(lines[:2 + min(K, 25)]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
