"""Học cách ghép face+body bằng bộ phân loại nhỏ "2 thực thể có cùng nhân vật không?" (MLP 2 lớp ẩn).

Huấn luyện CHỈ trên cặp lấy từ VAL (thực thể ghép được với GT, nhãn nhân vật đã biết); test không được dùng để
học hay chọn gì ngoài số đo cuối. Khoảng cách gom = 1 - P(cùng người); ngưỡng cắt chọn trên val (cách B).
Đặc trưng cặp: D_face, D_body (0 nếu không có modality chung), cờ có-chung-face / có-chung-body, tích.
Biến thể "+trang": thêm cùng-trang và log khoảng cách trang (hai thực thể cùng trang gần như chắc khác người).

Cần cache data/pred/ (scripts/eval_cluster_pred.py) và data/pred/fused.json (scripts/eval_fused.py) để so sánh.
    python scripts/eval_fused_learned.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from eval_fused import build, linkage_gpu, load_emb, mean, score  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def tens(b):
    return {k: torch.from_numpy(b[k]).to(DEV) for k in ("Ef", "Eb", "hf", "hb", "page")}


def feats(T, I, J, use_page):
    mf = (T["hf"][I] & T["hf"][J]).float()
    mb = (T["hb"][I] & T["hb"][J]).float()
    df = (1 - (T["Ef"][I] * T["Ef"][J]).sum(-1)) * mf
    db = (1 - (T["Eb"][I] * T["Eb"][J]).sum(-1)) * mb
    cols = [df, mf, db, mb, mf * mb, df * db]
    if use_page:
        pi, pj = T["page"][I], T["page"][J]
        cols += [(pi == pj).float(), torch.log1p((pi - pj).abs().float()) / 7]
    return torch.stack(cols, -1)


def sample_pairs(b, rng, cap):
    """Cặp (i,j) giữa thực thể 'tp' của một bộ: <=cap cặp dương + cùng số cặp âm ngẫu nhiên."""
    idx = np.where(b["kind"] == "tp")[0]
    lab = b["true"][idx]
    pos, neg = [], []
    for c in np.unique(lab):
        m = idx[lab == c]
        if len(m) >= 2:
            ii, jj = np.triu_indices(len(m), 1)
            pos.append(np.stack([m[ii], m[jj]], 1))
    if not pos:
        return None
    pos = np.concatenate(pos)
    if len(pos) > cap:
        pos = pos[rng.choice(len(pos), cap, replace=False)]
    a = rng.choice(idx, len(pos) * 3); c = rng.choice(idx, len(pos) * 3)
    ok = b["true"][a] != b["true"][c]
    neg = np.stack([a[ok], c[ok]], 1)[:len(pos)]
    P = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    return P, y


def make_set(books, rng, cap, use_page):
    X, Y = [], []
    for t, b in books.items():
        r = sample_pairs(b, rng, cap)
        if r is None:
            continue
        P, y = r
        T = tens(b)
        for s in range(0, len(P), 50_000):
            X.append(feats(T, torch.from_numpy(P[s:s + 50_000, 0]).to(DEV),
                           torch.from_numpy(P[s:s + 50_000, 1]).to(DEV), use_page).cpu())
        Y.append(torch.from_numpy(y).float())
    return torch.cat(X), torch.cat(Y)


def train_mlp(X, Y, epochs=30, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    X, Y = X.to(DEV), Y.to(DEV)
    for ep in range(epochs):
        perm = torch.randperm(len(X), device=DEV)
        tot = 0.0
        for s in range(0, len(X), 8192):
            b = perm[s:s + 8192]
            loss = nn.functional.binary_cross_entropy_with_logits(net(X[b]).squeeze(-1), Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b)
    print(f"   huấn luyện {len(X)} cặp, loss cuối {tot / len(X):.4f}", flush=True)
    return net.eval()


@torch.no_grad()
def dist_matrix(net, b, use_page):
    T = tens(b)
    n = len(b["true"])
    D = torch.empty(n, n, device=DEV)
    ar = torch.arange(n, device=DEV)
    step = max(1, 200_000 // n)   # <=200k cặp mỗi khối (GPU 2 GB)
    for r0 in range(0, n, step):
        r1 = min(n, r0 + step)
        I = ar[r0:r1].repeat_interleave(n); J = ar.repeat(r1 - r0)
        D[r0:r1] = 1 - torch.sigmoid(net(feats(T, I, J, use_page)).squeeze(-1)).view(r1 - r0, n)
    D = (D + D.T) / 2
    return D.cpu().numpy().astype(np.float32)


def auc(score_pos, score_neg):
    """AUC xếp hạng (điểm cao = giống hơn)."""
    s = np.concatenate([score_pos, score_neg])
    r = s.argsort().argsort() + 1.0
    # xử lý hoà: trung bình hạng
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt)
    avg = (csum - (cnt - 1) / 2.0)
    r = avg[inv]
    return (r[:len(score_pos)].sum() - len(score_pos) * (len(score_pos) + 1) / 2) / (len(score_pos) * len(score_neg))


def pair_auc(nets, books, rng, cap=30000):
    """AUC mức cặp trên test: học vs khoảng cách cố định (chỉ-face, chỉ-body, ghép 0.7/0.3)."""
    out = {}
    scores = {k: ([], []) for k in ("chỉ face", "chỉ body", "ghép 0.7/0.3", *nets)}
    for t, b in books.items():
        r = sample_pairs(b, rng, cap)
        if r is None:
            continue
        P, y = r
        T = tens(b)
        I = torch.from_numpy(P[:, 0]).to(DEV); J = torch.from_numpy(P[:, 1]).to(DEV)
        f = feats(T, I, J, True).cpu().numpy()
        df, mf, db, mb = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
        d_face = np.where(mf > 0, df, 1.0); d_body = np.where(mb > 0, db, 1.0)
        d_mix = np.where(mf + mb > 0, (0.7 * mf * df + 0.3 * mb * db) / np.maximum(0.7 * mf + 0.3 * mb, 1e-9), 1.0)
        cand = {"chỉ face": d_face, "chỉ body": d_body, "ghép 0.7/0.3": d_mix}
        for name, (net, up) in nets.items():
            with torch.no_grad():
                cand[name] = 1 - torch.sigmoid(net(feats(T, I, J, up)).squeeze(-1)).cpu().numpy()
        for k, d in cand.items():
            scores[k][0].append(-d[y == 1]); scores[k][1].append(-d[y == 0])
    for k, (p, n) in scores.items():
        out[k] = auc(np.concatenate(p), np.concatenate(n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=Path("data/characters.json"), type=Path)
    ap.add_argument("--pred", default=Path("data/pred"), type=Path)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--cap", type=int, default=100000, help="số cặp dương tối đa mỗi bộ khi học")
    ap.add_argument("--thrs", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    chars = json.loads(a.chars.read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)

    data = {}
    for s in ("val", "test"):
        det = json.loads((a.pred / f"det_{s}.json").read_text())
        data[s], _ = build(s, a, chars, det, load_emb("f_b", s, a.pred), load_emb("b_a", s, a.pred))

    nets, res = {}, {}
    for name, use_page in (("học (face+body)", False), ("học + trang", True)):
        print(f"== {name}", flush=True)
        X, Y = make_set(data["val"], rng, a.cap, use_page)
        net = train_mlp(X, Y)
        nets[name] = (net, use_page)
        vt = {t: linkage_gpu(dist_matrix(net, b, use_page)) for t, b in data["val"].items()}
        best = (-1, None)
        for th in a.thrs:
            r = mean(score(data["val"], vt, th, "B"), "bcF")
            if r > best[0]:
                best = (r, th)
        th = best[1]
        tt = {t: linkage_gpu(dist_matrix(net, b, use_page)) for t, b in data["test"].items()}
        rA, rB = score(data["test"], tt, th, "A"), score(data["test"], tt, th, "B")
        res[name] = {"thr": th, "val_B_bcF": best[0], "A": {k: mean(rA, k) for k in ("bcF", "F", "ARI")},
                     "B": {k: mean(rB, k) for k in ("P", "R", "bcF", "F", "ARI")}, "per_book_B": {t: m["bcF"] for t, m in rB}}
        print(f"{name:18s} thr {th} | val B bcF {best[0]:.3f} | test A bcF {res[name]['A']['bcF']:.3f} ARI {res[name]['A']['ARI']:.3f} "
              f"| test B bcF {res[name]['B']['bcF']:.3f} pairF {res[name]['B']['F']:.3f} ARI {res[name]['B']['ARI']:.3f}", flush=True)

    print("\nAUC mức cặp (test, cặp lấy từ thực thể ghép được, cân bằng dương/âm):")
    au = pair_auc(nets, data["test"], np.random.default_rng(1))
    for k, v in au.items():
        print(f"  {k:18s} {v:.4f}")
    res["pair_auc_test"] = au

    ref = json.loads((a.pred / "fused.json").read_text())
    names = ["chỉ face", "fused 0.7/0.3"]
    print(f"\n{'(test, cách B)':18s} {'bcF':>6s} {'pairF':>6s} {'ARI':>6s}")
    for n in names:
        m = ref[n]["B"]; print(f"{n:18s} {m['bcF']:6.3f} {m['F']:6.3f} {m['ARI']:6.3f}   (đã đo trước)")
    for n in nets:
        m = res[n]["B"]; print(f"{n:18s} {m['bcF']:6.3f} {m['F']:6.3f} {m['ARI']:6.3f}")
    cols = names + list(nets)
    print(f"\n{'bộ (test, B, bcF)':24s}" + "".join(f"{c:>18s}" for c in cols))
    for t in res[list(nets)[0]]["per_book_B"]:
        vals = [ref[n]["per_book_B"][t] for n in names] + [res[n]["per_book_B"][t] for n in nets]
        print(f"{t:24s}" + "".join(f"{v:18.3f}" for v in vals))
    if a.json:
        a.json.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
