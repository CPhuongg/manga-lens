"""Gom nhân vật với box face/body do DETECTOR dự đoán (yolo11s), không dùng box GT.

3 giai đoạn, mỗi giai đoạn cache nên chạy lại được:
  1. detect : chạy detector trên ảnh split -> data/pred/det_<split>.json (box face/body + conf)
  2. embed  : cắt crop box dự đoán, embedding bằng model re-ID đã fine-tune -> data/pred/emb_<tag>_<split>.npz
  3. eval   : ghép box dự đoán với GT (IoU>=0.5, tham lam theo conf), gom theo bộ, đo như eval_cluster

Hai cách đo (cùng ngưỡng cắt, chọn trên val theo cách B):
  A "chỉ box ghép được": chỉ giữ box dự đoán ghép được với GT của nhân vật có >=2 box (không tính FP/sót).
  B "end-to-end": thêm box FP làm nhãn rác riêng lẻ (mỗi FP một nhãn thật), và GT bị sót thành cụm lẻ.
Bỏ 'Other' như eval_cluster; nhân vật xét = nhân vật không-Other có >= min_count box GT cùng lớp.

Dùng:
    python scripts/eval_cluster_pred.py --cls face --tag f_b --weights-reid data/reid/f_b/best.pt \
        --root "D:/MINI APP/dataset/Manga109_released_2026_05_21"
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_cluster import cut, linkage, metrics  # noqa: E402
from train_reid import SIZE, ReID, embed_all, load_ckpt  # noqa: E402

CLS_ID = {"face": 2, "body": 3}


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def match(pred, gt, thr):
    """Ghép tham lam 1-1 theo conf giảm dần. pred: (N,5) [x1,y1,x2,y2,conf]; gt: (M,4). Trả dict {pred_i: gt_j}."""
    if len(pred) == 0 or len(gt) == 0:
        return {}
    M = iou_matrix(np.asarray(pred)[:, :4], gt)
    used, out = set(), {}
    for i in np.argsort(-np.asarray(pred)[:, 4]):
        j = int(np.argmax(np.where([k in used for k in range(M.shape[1])], -1, M[i])))
        if M[i, j] >= thr and j not in used:
            out[int(i)] = j; used.add(j)
    return out


# ---------------------------------------------------------------- 1. detect
def stage_detect(split, a):
    f = a.pred / f"det_{split}.json"
    det = json.loads(f.read_text()) if f.exists() else {}
    names = sorted(p.stem for p in (a.yolo / "images" / split).glob("*.jpg"))
    todo = [n for n in names if n not in det]
    if not todo:
        return det
    from ultralytics import YOLO
    model = YOLO(a.weights)
    print(f"[detect {split}] {len(todo)}/{len(names)} ảnh cần chạy", flush=True)
    for s in range(0, len(todo), 16):
        chunk = todo[s:s + 16]
        res = model.predict([str(a.yolo / "images" / split / f"{n}.jpg") for n in chunk], conf=a.conf,
                            imgsz=a.imgsz, device=a.device, verbose=False)
        for n, r in zip(chunk, res):
            b = r.boxes
            xyxy, conf, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
            det[n] = {c: [[*map(float, xyxy[i]), float(conf[i])] for i in range(len(cls)) if int(cls[i]) == cid]
                      for c, cid in CLS_ID.items()}
        if (s // 16) % 20 == 0:
            f.write_text(json.dumps(det)); print(f"  {s + len(chunk)}/{len(todo)}", flush=True)
    f.write_text(json.dumps(det))
    return det


# ---------------------------------------------------------------- 2. embed
def stage_embed(split, det, a):
    f = a.pred / f"emb_{a.tag}_{split}.npz"
    if f.exists():
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ReID(emb=a.emb).to(dev)
    load_ckpt(model, a.weights_reid, dev)
    H, W = SIZE[a.cls]
    crops, meta = [], []
    for n, d in sorted(det.items()):
        if not d[a.cls]:
            continue
        title, page = n.rsplit("_", 1)
        img = Image.open(a.root / "images" / title / f"{int(page):03d}.jpg").convert("RGB")
        for k, (x1, y1, x2, y2, c) in enumerate(d[a.cls]):
            w, h = x2 - x1, y2 - y1
            box = (max(0, int(x1 - 0.1 * w)), max(0, int(y1 - 0.1 * h)),
                   min(img.width, int(x2 + 0.1 * w)), min(img.height, int(y2 + 0.1 * h)))
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            crops.append(np.asarray(img.crop(box).resize((W, H), Image.BILINEAR, reducing_gap=2.0)))
            meta.append((title, int(page), k))
    print(f"[embed {split}/{a.cls}] {len(crops)} crop", flush=True)
    emb = embed_all(model, np.stack(crops), dev)
    np.savez_compressed(f, book=np.array([m[0] for m in meta]), page=np.array([m[1] for m in meta]),
                        k=np.array([m[2] for m in meta]), emb=emb.astype(np.float16))


# ---------------------------------------------------------------- 3. eval
def build_books(split, det, a, chars):
    """Trả {bộ: dict(E, true_det, pred_extra...)} cho mode A/B."""
    z = np.load(a.pred / f"emb_{a.tag}_{split}.npz")
    emb = z["emb"].astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-8)
    by_book = defaultdict(list)
    for i, (b, p, k) in enumerate(zip(z["book"], z["page"], z["k"])):
        by_book[str(b)].append((i, int(p), int(k)))
    out, stat = {}, defaultdict(int)
    for title, book in chars.items():
        if book["split"] != split:
            continue
        items = [it for it in book["items"] if it["cls"] == a.cls]
        cnt = defaultdict(int)
        for it in items:
            if not book["characters"][it["char"]]["other"]:
                cnt[it["char"]] += 1
        valid = {c for c, n in cnt.items() if n >= a.min_count}
        gt_by_page = defaultdict(list)
        for it in items:
            gt_by_page[it["page"]].append(it)
        pred_ids = {(p, k): i for i, p, k in by_book.get(title, [])}
        E_rows, true_lab, tag_kind = [], [], []   # tag_kind: 'tp' | 'fp'
        missed = []
        pages = {int(n.rsplit("_", 1)[1]) for n in det if n.rsplit("_", 1)[0] == title} | set(gt_by_page)
        for page in sorted(pages):
            d = det.get(f"{title}_{page:03d}", {a.cls: []})[a.cls]
            gts = gt_by_page.get(page, [])
            m = match(d, [g["box"] for g in gts], a.iou) if d and gts else {}
            matched_gt = set(m.values())
            for k in range(len(d)):
                i = pred_ids.get((page, k))
                if i is None:
                    continue
                if k in m:
                    g = gts[m[k]]
                    if g["char"] in valid:
                        E_rows.append(emb[i]); true_lab.append(g["char"]); tag_kind.append("tp")
                    stat["pred_matched"] += 1
                else:
                    E_rows.append(emb[i]); true_lab.append(f"FP_{len(true_lab)}"); tag_kind.append("fp")
                    stat["pred_fp"] += 1
            for j, g in enumerate(gts):
                if g["char"] in valid:
                    stat["gt_valid"] += 1
                    if j not in matched_gt:
                        missed.append(g["char"])
                    else:
                        stat["gt_valid_matched"] += 1
        if len(E_rows) >= 5:
            out[title] = dict(E=np.stack(E_rows), true=np.array(true_lab), kind=np.array(tag_kind), missed=missed)
    return out, stat


def score(books, thr, trees, mode):
    rows = []
    for b, d in books.items():
        n = len(d["true"])
        lab = cut(trees[b], n, thr)
        if mode == "A":
            keep = d["kind"] == "tp"
            if keep.sum() < 2:
                continue
            m = metrics(d["true"][keep], lab[keep])
            kp = len(np.unique(lab[keep])); kt = len(np.unique(d["true"][keep]))
        else:
            true = np.concatenate([d["true"], np.array(d["missed"], dtype=d["true"].dtype)])
            pred = np.concatenate([lab, lab.max() + 1 + np.arange(len(d["missed"]))])
            m = metrics(true, pred)
            kp = len(np.unique(pred)); kt = len({t for t in true if not str(t).startswith("FP_")})
        m["k_pred"], m["k_true"] = kp, kt
        rows.append((b, m))
    return rows


def mean(rows, k):
    return float(np.mean([m[k] for _, m in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--chars", default=Path("data/characters.json"), type=Path)
    ap.add_argument("--yolo", default=Path("data/yolo"), type=Path)
    ap.add_argument("--pred", default=Path("data/pred"), type=Path)
    ap.add_argument("--weights", default="weights/best.pt", help="detector YOLO")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cls", default="face", choices=["face", "body"])
    ap.add_argument("--tag", required=True, help="tên model re-ID (dùng đặt tên cache)")
    ap.add_argument("--weights-reid", required=True, type=Path)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--thrs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    a.pred.mkdir(parents=True, exist_ok=True)
    chars = json.loads(a.chars.read_text(encoding="utf-8"))

    data = {}
    for s in a.splits:
        det = stage_detect(s, a)
        stage_embed(s, det, a)
        data[s] = build_books(s, det, a, chars)
        st = data[s][1]
        print(f"[{s}] {a.cls}: GT hợp lệ {st['gt_valid']}, ghép được {st['gt_valid_matched']} "
              f"(recall {st['gt_valid_matched'] / max(st['gt_valid'], 1):.3f}); box dự đoán đưa vào gom: "
              f"ghép GT {st['pred_matched']}, FP {st['pred_fp']}")
    trees = {s: {b: linkage(1 - d["E"] @ d["E"].T) for b, d in data[s][0].items()} for s in a.splits}

    best = (-1, None)
    for t in a.thrs:
        r = score(data["val"][0], t, trees["val"], "B")
        print(f"  val thr {t:.2f}: B bcF {mean(r, 'bcF'):.3f} ARI {mean(r, 'ARI'):.3f} "
              f"k_pred/k_true {mean(r, 'k_pred'):.1f}/{mean(r, 'k_true'):.1f}")
        if mean(r, "bcF") > best[0]:
            best = (mean(r, "bcF"), t)
    thr = best[1]
    print(f"-> chọn ngưỡng {thr} theo val (cách B)")
    res = {"thr": thr, "val_B_bcF": best[0]}
    if "test" in a.splits:
        for mode, name in (("A", "chỉ box ghép được"), ("B", "end-to-end")):
            rows = score(data["test"][0], thr, trees["test"], mode)
            print(f"\nTEST cách {mode} ({name}) — mean theo bộ ({len(rows)} bộ)")
            print(f"{'bộ':24s} {'P':>5s} {'R':>5s} {'pairF':>6s} {'bcF':>5s} {'ARI':>5s} {'k_pred':>6s} {'k_true':>6s}")
            for b, m in rows:
                print(f"{b:24s} {m['P']:5.3f} {m['R']:5.3f} {m['F']:6.3f} {m['bcF']:5.3f} {m['ARI']:5.3f} "
                      f"{m['k_pred']:6d} {m['k_true']:6d}")
            print(f"{'TRUNG BÌNH':24s} {mean(rows, 'P'):5.3f} {mean(rows, 'R'):5.3f} {mean(rows, 'F'):6.3f} "
                  f"{mean(rows, 'bcF'):5.3f} {mean(rows, 'ARI'):5.3f} {mean(rows, 'k_pred'):6.1f} {mean(rows, 'k_true'):6.1f}")
            res[mode] = {k: mean(rows, k) for k in ("P", "R", "F", "bcF", "ARI", "NMI", "k_pred", "k_true")}
    if a.json:
        a.json.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
