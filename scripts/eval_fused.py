"""Ghép face + body (box detector dự đoán) để gom nhân vật; so với chỉ-face / chỉ-body CÙNG tập thực thể.

Thực thể (instance) = 1 người trên 1 trang: mặt + body nhỏ nhất bao nó (mặt nằm >80% diện tích trong body),
hoặc mặt đứng riêng, hoặc body đứng riêng. GT dựng bằng đúng quy tắc ghép đó trên box GT.
Ghép dự đoán <-> GT: tham lam theo conf, điểm = max(IoU mặt, IoU body) >= 0.5. Bỏ 'Other'; nhân vật xét =
không-Other có >= min_count thực thể GT. Hai cách đo như eval_cluster_pred: A (chỉ thực thể ghép được),
B (end-to-end: FP thành nhãn rác riêng, GT sót thành cụm lẻ).

Khoảng cách 2 thực thể = trung bình có trọng số của (1-cos) trên các modality CẢ HAI cùng có
(wf*D_face, wb*D_body); không có modality chung (mặt-riêng vs body-riêng) -> khoảng cách --cross (1.0).
Mỗi cấu hình (wf, wb) chọn ngưỡng cắt trên VAL (cách B), báo test.

Cần data/pred/det_{val,test}.json và emb_f_b_*.npz, emb_b_a_*.npz (scripts/eval_cluster_pred.py).
    python scripts/eval_fused.py
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from eval_cluster import cut, metrics  # noqa: E402
from eval_cluster_pred import iou_matrix  # noqa: E402


def area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def link(faces, bodies):
    """faces/bodies: list box. Trả list (fi|None, bi|None): mặt + body nhỏ nhất bao nó; body không ai nhận đứng riêng."""
    used, inst = set(), []
    for fi, fb in enumerate(faces):
        best = None
        for bi, bb in enumerate(bodies):
            w = min(fb[2], bb[2]) - max(fb[0], bb[0]); h = min(fb[3], bb[3]) - max(fb[1], bb[1])
            if w > 0 and h > 0 and w * h / max(area(fb), 1e-9) > 0.8:
                if best is None or area(bb) < area(bodies[best]):
                    best = bi
        inst.append((fi, best))
        if best is not None:
            used.add(best)
    inst += [(None, bi) for bi in range(len(bodies)) if bi not in used]
    return inst


def load_emb(tag, split, pred):
    z = np.load(pred / f"emb_{tag}_{split}.npz")
    e = z["emb"].astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True).clip(1e-8)
    d = {}
    for i, (b, p, k) in enumerate(zip(z["book"], z["page"], z["k"])):
        d[(str(b), int(p), int(k))] = e[i]
    return d


def match_inst(pi, gi, pf, pb, gf, gb, pconf, thr):
    """Ghép tham lam 1-1 thực thể dự đoán <-> GT. pi/gi: list (fi,bi). Trả {pred_idx: gt_idx}."""
    IF = iou_matrix(pf, gf) if len(pf) and len(gf) else None
    IB = iou_matrix(pb, gb) if len(pb) and len(gb) else None
    used, out = set(), {}
    for i in np.argsort(-np.asarray(pconf)):
        best, bj = thr, None
        for j, (gfi, gbi) in enumerate(gi):
            if j in used:
                continue
            s = -1.0
            if pi[i][0] is not None and gfi is not None and IF is not None:
                s = max(s, IF[pi[i][0], gfi])
            if pi[i][1] is not None and gbi is not None and IB is not None:
                s = max(s, IB[pi[i][1], gbi])
            if s >= best:
                best, bj = s, j
        if bj is not None:
            out[int(i)] = bj; used.add(bj)
    return out


def build(split, a, chars, det, embf, embb):
    books, st = {}, defaultdict(int)
    for title, book in chars.items():
        if book["split"] != split:
            continue
        meta = book["characters"]
        gtf, gtb = defaultdict(list), defaultdict(list)
        for it in book["items"]:
            (gtf if it["cls"] == "face" else gtb)[it["page"]].append(it)
        gt_inst = {}
        cnt = defaultdict(int)
        for page in set(gtf) | set(gtb):
            gf, gb = gtf.get(page, []), gtb.get(page, [])
            inst = link([g["box"] for g in gf], [g["box"] for g in gb])
            lst = []
            for fi, bi in inst:
                ch = gf[fi]["char"] if fi is not None else gb[bi]["char"]
                lst.append((fi, bi, ch))
                if not meta[ch]["other"]:
                    cnt[ch] += 1
            gt_inst[page] = (gf, gb, lst)
        valid = {c for c, n in cnt.items() if n >= a.min_count}
        pages = {int(n.rsplit("_", 1)[1]) for n in det if n.rsplit("_", 1)[0] == title} | set(gt_inst)
        rows = []   # dict(ef, eb, true, kind)
        missed = []
        for page in sorted(pages):
            d = det.get(f"{title}_{page:03d}", {"face": [], "body": []})
            pfb = np.array([x[:4] for x in d["face"]]).reshape(-1, 4)
            pbb = np.array([x[:4] for x in d["body"]]).reshape(-1, 4)
            pinst = link(pfb, pbb)
            pconf = [max(d["face"][fi][4] if fi is not None else 0, d["body"][bi][4] if bi is not None else 0)
                     for fi, bi in pinst]
            gf, gb, glst = gt_inst.get(page, ([], [], []))
            gfb = np.array([g["box"] for g in gf]).reshape(-1, 4)
            gbb = np.array([g["box"] for g in gb]).reshape(-1, 4)
            m = match_inst(pinst, [(x[0], x[1]) for x in glst], pfb, pbb, gfb, gbb, pconf, a.iou) if pinst and glst else {}
            for i, (fi, bi) in enumerate(pinst):
                ef = embf.get((title, page, fi)) if fi is not None else None
                eb = embb.get((title, page, bi)) if bi is not None else None
                if ef is None and eb is None:
                    continue
                if i in m:
                    ch = glst[m[i]][2]
                    if ch not in valid:
                        continue
                    kind, lab = "tp", ch
                else:
                    kind, lab = "fp", f"FP_{len(rows)}"
                rows.append(dict(ef=ef, eb=eb, true=lab, kind=kind, page=page))
                st["pred_both" if (ef is not None and eb is not None) else "pred_face" if ef is not None else "pred_body"] += 1
                st["pred_" + kind] += 1
            matched = set(m.values())
            for j, (_, _, ch) in enumerate(glst):
                if ch in valid:
                    st["gt_valid"] += 1
                    if j in matched:
                        st["gt_matched"] += 1
                    else:
                        missed.append(ch)
        if len(rows) >= 5:
            dim = 256
            n = len(rows)
            Ef = np.zeros((n, dim), np.float32); Eb = np.zeros((n, dim), np.float32)
            hf = np.zeros(n, bool); hb = np.zeros(n, bool)
            for i, r in enumerate(rows):
                if r["ef"] is not None:
                    Ef[i] = r["ef"]; hf[i] = True
                if r["eb"] is not None:
                    Eb[i] = r["eb"]; hb[i] = True
            books[title] = dict(Ef=Ef, Eb=Eb, hf=hf, hb=hb, true=np.array([r["true"] for r in rows]),
                                kind=np.array([r["kind"] for r in rows]), missed=missed,
                                page=np.array([r["page"] for r in rows]))
    return books, st


def distance(b, wf, wb, cross, prio=False):
    hf, hb = b["hf"], b["hb"]
    Df = 1 - b["Ef"] @ b["Ef"].T
    Db = 1 - b["Eb"] @ b["Eb"].T
    if prio:   # ưu tiên mặt: cả hai có mặt -> D_face; nếu không, cả hai có body -> D_body; còn lại -> cross
        both_f = hf[:, None] & hf[None]
        both_b = hb[:, None] & hb[None]
        return np.where(both_f, Df, np.where(both_b, Db, cross)).astype(np.float32)
    mf = (hf[:, None] & hf[None]).astype(np.float32) * wf
    mb = (hb[:, None] & hb[None]).astype(np.float32) * wb
    den = mf + mb
    D = np.where(den > 0, (mf * Df + mb * Db) / np.maximum(den, 1e-9), cross)
    return D.astype(np.float32)


def linkage_gpu(D):
    """Average-linkage (như eval_cluster.linkage) nhưng argmin/cập nhật trên GPU nếu có."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(D)
    T = torch.from_numpy(D).to(dev)
    T.fill_diagonal_(float("inf"))
    size = torch.ones(n, device=dev)
    inf = float("inf")
    merges = []
    sz = [1.0] * n
    for _ in range(n - 1):
        k = int(torch.argmin(T))
        i, j = divmod(k, n)
        merges.append((float(T[i, j]), i, j))
        new = (T[i] * sz[i] + T[j] * sz[j]) / (sz[i] + sz[j])
        T[i, :] = new; T[:, i] = new; T[i, i] = inf
        T[j, :] = inf; T[:, j] = inf
        sz[i] += sz[j]
    return merges


def score(books, trees, thr, mode):
    rows = []
    for t, b in books.items():
        n = len(b["true"])
        lab = cut(trees[t], n, thr)
        if mode == "A":
            keep = b["kind"] == "tp"
            if keep.sum() < 2:
                continue
            m = metrics(b["true"][keep], lab[keep])
        else:
            true = np.concatenate([b["true"], np.array(b["missed"], dtype=b["true"].dtype)])
            pred = np.concatenate([lab, lab.max() + 1 + np.arange(len(b["missed"]))])
            m = metrics(true, pred)
        rows.append((t, m))
    return rows


def mean(rows, k):
    return float(np.mean([m[k] for _, m in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=Path("data/characters.json"), type=Path)
    ap.add_argument("--pred", default=Path("data/pred"), type=Path)
    ap.add_argument("--face-tag", default="f_b")
    ap.add_argument("--body-tag", default="b_a")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--cross", type=float, default=1.0, help="khoảng cách khi không có modality chung")
    ap.add_argument("--thrs", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--only", nargs="+", default=None, help="chỉ chạy các cấu hình này (theo tên)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    chars = json.loads(a.chars.read_text(encoding="utf-8"))

    data = {}
    for s in ("val", "test"):
        det = json.loads((a.pred / f"det_{s}.json").read_text())
        data[s], st = build(s, a, chars, det, load_emb(a.face_tag, s, a.pred), load_emb(a.body_tag, s, a.pred))
        print(f"[{s}] GT hợp lệ {st['gt_valid']}, ghép được {st['gt_matched']} (recall {st['gt_matched'] / max(st['gt_valid'], 1):.3f}); "
              f"thực thể dự đoán: mặt+body {st['pred_both']}, chỉ mặt {st['pred_face']}, chỉ body {st['pred_body']}; "
              f"ghép GT {st['pred_tp']}, FP {st['pred_fp']}", flush=True)

    configs = [("chỉ face", 1.0, 0.0, False), ("chỉ body", 0.0, 1.0, False), ("fused 0.5/0.5", 0.5, 0.5, False),
               ("fused 0.7/0.3", 0.7, 0.3, False), ("fused 0.85/0.15", 0.85, 0.15, False),
               ("ưu tiên mặt", 1.0, 1.0, True)]
    if a.only:
        configs = [c for c in configs if c[0] in a.only]
    results = {}
    for name, wf, wb, prio in configs:
        vt = {t: linkage_gpu(distance(b, wf, wb, a.cross, prio)) for t, b in data["val"].items()}
        best = (-1, None)
        for th in a.thrs:
            r = mean(score(data["val"], vt, th, "B"), "bcF")
            if r > best[0]:
                best = (r, th)
        th = best[1]
        tt = {t: linkage_gpu(distance(b, wf, wb, a.cross, prio)) for t, b in data["test"].items()}
        rA, rB = score(data["test"], tt, th, "A"), score(data["test"], tt, th, "B")
        results[name] = {"thr": th, "val_B_bcF": best[0],
                         "A": {k: mean(rA, k) for k in ("bcF", "F", "ARI", "NMI")},
                         "B": {k: mean(rB, k) for k in ("P", "R", "bcF", "F", "ARI", "NMI")}, "per_book_B": {t: m["bcF"] for t, m in rB}}
        print(f"{name:16s} thr {th} | val B bcF {best[0]:.3f} | test A bcF {results[name]['A']['bcF']:.3f} ARI {results[name]['A']['ARI']:.3f} "
              f"| test B bcF {results[name]['B']['bcF']:.3f} pairF {results[name]['B']['F']:.3f} ARI {results[name]['B']['ARI']:.3f}", flush=True)
    names = list(results)
    print(f"\n{'bộ (test, cách B, bcF)':26s}" + "".join(f"{n:>17s}" for n in names))
    for t in results[names[0]]["per_book_B"]:
        print(f"{t:26s}" + "".join(f"{results[n]['per_book_B'][t]:17.3f}" for n in names))
    if a.json:
        a.json.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
