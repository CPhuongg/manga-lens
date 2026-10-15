"""Đo gom nhân vật TRONG TỪNG BỘ từ embedding (scripts/embed_chars.py).

Clustering: agglomerative average-linkage, khoảng cách cosine, cắt theo ngưỡng. Ngưỡng chọn trên val
(tối đa BCubed F1 trung bình theo bộ), rồi áp nguyên cho test. Bỏ nhân vật 'Other' khỏi tập đo.
Metric (mỗi bộ, rồi trung bình): pairwise P/R/F1, BCubed F1, ARI, NMI, số cụm vs số nhân vật thật,
retrieval Recall@1 (cùng nhân vật, loại chính nó) và mAP.

Dùng:
    python scripts/eval_cluster.py --feat face            # face | body | both
    python scripts/eval_cluster.py --feat both --min-count 2
"""
import argparse
import json
from pathlib import Path

import numpy as np


def linkage(D):
    """Average-linkage trên ma trận khoảng cách D (n,n): trả list (khoảng cách, i, j) theo thứ tự gộp.
    Average-linkage đơn điệu nên cắt tại ngưỡng bất kỳ = áp các phép gộp có khoảng cách <= ngưỡng."""
    n = len(D)
    D = D.astype(np.float64).copy()
    np.fill_diagonal(D, np.inf)
    size = np.ones(n)
    merges = []
    for _ in range(n - 1):
        k = np.argmin(D)
        i, j = divmod(k, n)
        merges.append((D[i, j], i, j))
        new = (D[i] * size[i] + D[j] * size[j]) / (size[i] + size[j])
        D[i, :] = new; D[:, i] = new; D[i, i] = np.inf
        D[j, :] = np.inf; D[:, j] = np.inf
        size[i] += size[j]
    return merges


def cut(merges, n, thr):
    label = np.arange(n)
    for d, i, j in merges:
        if d > thr:
            break
        label[label == j] = i
    return label


def contingency(a, b):
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    M = np.zeros((len(ua), len(ub)))
    np.add.at(M, (ia, ib), 1)
    return M


def metrics(true, pred):
    M = contingency(true, pred)
    n = M.sum()
    comb = lambda x: x * (x - 1) / 2
    tp = comb(M).sum()
    p_pairs, t_pairs = comb(M.sum(0)).sum(), comb(M.sum(1)).sum()
    P = tp / p_pairs if p_pairs else 1.0
    R = tp / t_pairs if t_pairs else 1.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    # BCubed
    bp = (M ** 2 / M.sum(0, keepdims=True)).sum() / n
    br = (M ** 2 / M.sum(1, keepdims=True)).sum() / n
    bf = 2 * bp * br / (bp + br) if bp + br else 0.0
    exp = p_pairs * t_pairs / comb(n)
    ari = (tp - exp) / (0.5 * (p_pairs + t_pairs) - exp) if (0.5 * (p_pairs + t_pairs) - exp) else 1.0
    pi, pj = M.sum(1) / n, M.sum(0) / n
    nz = M > 0
    mi = (M[nz] / n * np.log(M[nz] * n / (M.sum(1)[:, None] * M.sum(0)[None])[nz])).sum()
    h = lambda p: -(p * np.log(p)).sum()
    nmi = mi / ((h(pi) + h(pj)) / 2) if h(pi) + h(pj) else 1.0
    return dict(P=P, R=R, F=F, bcF=bf, ARI=ari, NMI=nmi)


def retrieval(E, y):
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)
    r1, aps = [], []
    for i in range(len(y)):
        rel = (y[order[i]] == y[i])[:-1]  # bỏ chính nó (nằm cuối do -inf)
        if rel.sum() == 0:
            continue
        r1.append(rel[0])
        ranks = np.where(rel)[0] + 1
        aps.append((np.arange(1, len(ranks) + 1) / ranks).mean())
    return float(np.mean(r1)), float(np.mean(aps))


EMB_DIR = Path("data/emb")


def load(split, tag, feat, chars, min_count, part):
    z = np.load(EMB_DIR / f"{tag}_{split}.npz")
    d = int(z["dim_half"])
    emb = z["emb"].astype(np.float32)
    emb = emb[:, :d] if part == "cls" else emb[:, d:] if part == "mean" else emb
    books = {}
    for b in np.unique(z["book"]):
        sel = z["book"] == b
        idx, cls, e = z["idx"][sel], z["cls"][sel], emb[sel]
        cdef = chars[str(b)]["characters"]
        items = chars[str(b)]["items"]
        lab = np.array([items[i]["char"] for i in idx])
        keep = np.array([not cdef[c]["other"] for c in lab])
        if feat != "both":
            keep &= cls == feat
        e, lab = e[keep], lab[keep]
        e = e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-8)
        ids, cnt = np.unique(lab, return_counts=True)
        ok = np.isin(lab, ids[cnt >= min_count])
        if ok.sum() >= 5:
            books[str(b)] = (e[ok], lab[ok])
    return books


def dendro(books):
    return {b: linkage(1 - E @ E.T) for b, (E, y) in books.items()}


def run(books, trees, thr):
    rows = []
    for b, (E, y) in books.items():
        pred = cut(trees[b], len(y), thr)
        m = metrics(y, pred)
        m["k_pred"], m["k_true"] = len(np.unique(pred)), len(np.unique(y))
        rows.append((b, m))
    return rows


def mean(rows, key):
    return float(np.mean([m[key] for _, m in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=Path("data/characters.json"), type=Path)
    ap.add_argument("--tag", default="dinov2-small")
    ap.add_argument("--feat", default="face", choices=["face", "body", "both"])
    ap.add_argument("--part", default="both", choices=["cls", "mean", "both"])
    ap.add_argument("--min-count", type=int, default=2, help="chỉ giữ nhân vật có ≥ N box")
    ap.add_argument("--emb-dir", default=Path("data/emb"), type=Path)
    ap.add_argument("--no-test", action="store_true", help="chỉ đánh giá val (chọn tham số), không đụng test")
    ap.add_argument("--thrs", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    args = ap.parse_args()
    global EMB_DIR
    EMB_DIR = args.emb_dir
    chars = json.loads(args.chars.read_text(encoding="utf-8"))

    val = load("val", args.tag, args.feat, chars, args.min_count, args.part)
    test = {} if args.no_test else load("test", args.tag, args.feat, chars, args.min_count, args.part)
    print(f"feat={args.feat} part={args.part} min_count={args.min_count}: val {len(val)} bộ "
          f"({sum(len(v[1]) for v in val.values())} box), test {len(test)} bộ ({sum(len(v[1]) for v in test.values())} box)")
    best = (-1, None)
    vtrees = dendro(val)
    for t in args.thrs:
        r = run(val, vtrees, t)
        print(f"  val thr {t:.2f}: bcF {mean(r,'bcF'):.3f} pairF {mean(r,'F'):.3f} ARI {mean(r,'ARI'):.3f} "
              f"k_pred/k_true {mean(r,'k_pred'):.1f}/{mean(r,'k_true'):.1f}")
        if mean(r, "bcF") > best[0]:
            best = (mean(r, "bcF"), t)
    thr = best[1]
    print(f"-> chọn ngưỡng {thr} theo val")
    if args.no_test:
        return
    rows = run(test, dendro(test), thr)
    print(f"{'bộ':24s} {'P':>5s} {'R':>5s} {'pairF':>6s} {'bcF':>5s} {'ARI':>5s} {'NMI':>5s} {'k_pred':>6s} {'k_true':>6s} {'R@1':>5s} {'mAP':>5s}")
    r1s, maps = [], []
    for b, m in rows:
        E, y = test[b]
        r1, mp = retrieval(E, y)
        r1s.append(r1); maps.append(mp)
        print(f"{b:24s} {m['P']:5.3f} {m['R']:5.3f} {m['F']:6.3f} {m['bcF']:5.3f} {m['ARI']:5.3f} {m['NMI']:5.3f} "
              f"{m['k_pred']:6d} {m['k_true']:6d} {r1s[-1]:5.3f} {maps[-1]:5.3f}")
    print(f"{'TRUNG BÌNH (theo bộ)':24s} {mean(rows,'P'):5.3f} {mean(rows,'R'):5.3f} {mean(rows,'F'):6.3f} "
          f"{mean(rows,'bcF'):5.3f} {mean(rows,'ARI'):5.3f} {mean(rows,'NMI'):5.3f} {mean(rows,'k_pred'):6.1f} "
          f"{mean(rows,'k_true'):6.1f} {np.mean(r1s):5.3f} {np.mean(maps):5.3f}")
    # baseline ngẫu nhiên: nhãn xáo trộn, cùng số cụm thật
    rng = np.random.default_rng(0)
    rb = []
    for b, (E, y) in test.items():
        rb.append(metrics(y, rng.permutation(y))["bcF"])
    print(f"baseline xáo nhãn (bcF): {np.mean(rb):.3f}")


if __name__ == "__main__":
    main()
