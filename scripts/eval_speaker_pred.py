"""Đo speaker attribution với box do DETECTOR dự đoán (frame/text/face/body), trên cùng các trang có nhãn Dialog.

    python scripts/eval_speaker_pred.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21" --split val
    python scripts/eval_speaker_pred.py --root ... --split test --model data/speaker/model.pt

Cách tính (giống eval_order_pred.py / eval_cluster_pred.py):
- text dự đoán ghép với text GT có nhãn bằng IoU>=0.5; ENTITY dự đoán (body, hoặc mặt đứng riêng) ghép với body GT
  (mặt đứng riêng: ghép với mặt GT rồi lấy body GT nhỏ nhất bao nó).
- Cách A: chỉ tính thoại GT có nhãn mà detector bắt được (cô lập chất lượng gán người nói).
- Cách B (end-to-end): thoại GT có nhãn mà detector bỏ sót tính SAI. Box text thừa không có nhãn nên không tính.
- Đúng ở mức box: thực thể dự đoán ghép được với một body người nói. Thực thể dự đoán không ghép được GT (box thừa) = sai.
Cache detector: data/pred/det_all_<split>.json (conf 0.25, imgsz 1024).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ocr import match_boxes  # noqa: E402
from src.speaker import ContextScorer, SpeakerScorer, entities_from_boxes, features, nearest_any, nearest_in_panel  # noqa: E402

CLS = {"frame": 0, "text": 1, "face": 2, "body": 3}


def detect_pages(keys, root, weights, conf, imgsz, device, cache):
    det = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [k for k in keys if k not in det]
    if todo:
        from ultralytics import YOLO
        model = YOLO(weights)
        print(f"[detect] {len(todo)}/{len(keys)} trang cần chạy", flush=True)
        for s in range(0, len(todo), 8):
            chunk = todo[s:s + 8]
            res = model.predict([str(root / "images" / k.split("/")[0] / f"{int(k.split('/')[1]):03d}.jpg")
                                 for k in chunk], conf=conf, imgsz=imgsz, device=device, verbose=False)
            for k, r in zip(chunk, res):
                b = r.boxes
                xyxy, c, cls = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
                det[k] = {n: [[*map(float, xyxy[i]), float(c[i])] for i in range(len(cls)) if int(cls[i]) == cid]
                          for n, cid in CLS.items()}
            if (s // 8) % 25 == 0:
                cache.write_text(json.dumps(det)); print(f"  {s + len(chunk)}/{len(todo)}", flush=True)
        cache.write_text(json.dumps(det))
    return det


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--labels", default="data/speaker_labels.json")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--weights", default="weights/best.pt")
    ap.add_argument("--model", default=None, help="ckpt SpeakerScorer; bỏ trống = chỉ baseline")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", type=Path, default=None)
    a = ap.parse_args()

    data = json.loads(Path(a.labels).read_text(encoding="utf-8"))["books"]
    keys, meta = [], {}
    for t, bk in sorted(data.items()):
        if bk["split"] != a.split:
            continue
        other = {i for i, c in bk["characters"].items() if c["other"]}
        for pi, pg in bk["pages"].items():
            if pg["links"]:
                k = f"{t}/{int(pi)}"
                keys.append(k); meta[k] = (t, pg, other)
    cache = a.cache or Path(f"data/pred/det_all_{a.split}.json")
    det = detect_pages(keys, a.root, a.weights, a.conf, a.imgsz, a.device, cache)

    methods = {"nearest toàn trang": nearest_any, "nearest cùng panel": nearest_in_panel}
    if a.model:
        import torch
        c = torch.load(a.model, map_location="cpu", weights_only=False)
        methods["mô hình học"] = (ContextScorer.load(a.model) if "s1" in c else SpeakerScorer.load(a.model)).predict
    res = {m: defaultdict(lambda: defaultdict(int)) for m in methods}
    for k in keys:
        t, pg, other = meta[k]
        d = det[k]
        P = {n: [x[:4] for x in d[n]] for n in CLS}
        gt_t = [x["box"] for x in pg["text"]]
        gt_b, gt_f = pg["body"], pg["face"]
        # GT thực thể: id body -> char; mặt GT -> body GT nhỏ nhất bao nó
        face_body = []
        for f in gt_f:
            cx, cy = (f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2
            ins = [b for b in gt_b if b["box"][0] <= cx <= b["box"][2] and b["box"][1] <= cy <= b["box"][3]]
            face_body.append(min(ins, key=lambda b: (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]))["id"]
                             if ins else None)
        ents = entities_from_boxes(P["body"], P["face"])
        gt_char = {b["id"]: b["char"] for b in gt_b}
        # ent dự đoán -> id body GT (None nếu không ghép được)
        ent_gt = [None] * len(ents)
        bi = [i for i, e in enumerate(ents) if e["src"][0] == "body"]
        for pi_, gj in match_boxes([P["body"][ents[i]["src"][1]] for i in bi], [b["box"] for b in gt_b], a.iou):
            ent_gt[bi[pi_]] = gt_b[gj]["id"]
        fi = [i for i, e in enumerate(ents) if e["src"][0] == "face"]
        for pi_, gj in match_boxes([P["face"][ents[i]["src"][1]] for i in fi], [f["box"] for f in gt_f], a.iou):
            ent_gt[fi[pi_]] = face_body[gj]
        # text dự đoán -> text GT
        pt = P["text"]
        pair = {g: p for p, g in match_boxes(pt, gt_t, a.iou)}  # gt idx -> pred idx
        gold = [(gi, x) for gi, x in enumerate(pg["text"]) if x["id"] in pg["links"]]
        W, H = pg["w"], pg["h"]
        if ents and pt:
            F, D, _, ctx = features(P["frame"], pt, ents, (W, H))
        for m, fn in methods.items():
            pred = fn(F, D, ents, ctx) if (ents and pt) else None
            r = res[m][t]
            for gi, x in gold:
                sp = set(pg["links"][x["id"]])
                sp_chars = {gt_char[s] for s in sp if gt_char.get(s) and gt_char[s] not in other}
                r["n"] += 1
                r["n_char"] += bool(sp_chars)
                if gi not in pair or pred is None:
                    r["missed_text"] += gi not in pair
                    r["no_ent"] += gi in pair
                    continue
                r["matched"] += 1
                e = int(pred[pair[gi]])
                g = ent_gt[e]
                r["box"] += g in sp
                if sp_chars:
                    r["char"] += (g is not None and gt_char.get(g) in sp_chars)
                    r["matched_char"] += 1
    for m, per in res.items():
        tot = defaultdict(int)
        for t in per:
            for kk, v in per[t].items():
                tot[kk] += v
        n, mt = tot["n"], max(1, tot["matched"])
        print(f"\n== {m} | {a.split}: {len(per)} bộ, {n} thoại có nhãn, detector bắt được {tot['matched']} "
              f"({tot['matched'] / n:.1%}), sót {tot['missed_text']}, không có thực thể {tot['no_ent']}")
        print(f"  A (chỉ thoại bắt được)   đúng body {tot['box'] / mt:.3f}   đúng nhân vật "
              f"{tot['char'] / max(1, tot['matched_char']):.3f} (n={tot['matched_char']})")
        print(f"  B (end-to-end, sót = sai) đúng body {tot['box'] / n:.3f}   đúng nhân vật "
              f"{tot['char'] / max(1, tot['n_char']):.3f} (n={tot['n_char']})")


if __name__ == "__main__":
    main()
