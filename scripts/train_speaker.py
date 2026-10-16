"""Train bộ chấm điểm người nói (MLP listwise) trên nhãn Dialog đã ánh xạ, box GT. Train 87 bộ, chọn epoch trên val.

    python scripts/train_speaker.py                  # giai đoạn 1 (hình học) -> data/speaker/model.pt
    python scripts/train_speaker.py --context        # + giai đoạn 2 (xác suất câu trước/sau) -> data/speaker/model_ctx.pt
    python scripts/train_speaker.py --context --test # thêm báo TEST một lần bằng ckpt tốt nhất theo val
Không bật fliplr/ tăng cường lật: vị trí trái/phải mang ý nghĩa hướng đọc.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_speaker import evaluate, load_pages, report  # noqa: E402
from src.speaker import (N_CTX, N_FEAT, TOPK, ContextScorer, SpeakerScorer, make_net, nearest_in_panel,  # noqa: E402
                         stage2_inputs, topk_idx)


def build(pages, s1=None):
    """X [N, TOPK, n_in], Y [N, 2, TOPK] (mask hợp lệ, mask đúng). s1 != None -> đặc trưng giai đoạn 2."""
    X, Y = [], []
    for t, pi, pg, ents, F, D, tp, gold, ent_char, other, ctx in pages:
        if len(ents) == 0:
            continue
        if s1 is None:
            idx = topk_idx(D)
            XP = np.take_along_axis(F, idx[:, :, None], axis=1)
        else:
            idx, p1 = s1.probs(F, D)
            XP = stage2_inputs(F, idx, p1, ctx)
        for ti, (es, ch, _) in gold.items():
            y = np.isin(idx[ti], list(es))
            if not y.any():
                continue
            x = np.zeros((TOPK, XP.shape[-1]), np.float32)
            m = np.zeros(TOPK, bool)
            k = len(idx[ti])
            x[:k] = XP[ti]
            m[:k] = True
            yy = np.zeros(TOPK, bool)
            yy[:k] = y
            X.append(x), Y.append(np.stack([m, yy]))
    return np.stack(X), np.stack(Y)


def fit(X, Y, va, make_scorer, a, out, n_in):
    mu = X[Y[:, 0]].mean(0)
    sd = X[Y[:, 0]].std(0) + 1e-6
    Xn = torch.from_numpy(((X - mu) / sd).astype(np.float32))
    M, G = torch.from_numpy(Y[:, 0]), torch.from_numpy(Y[:, 1])
    net = make_net(a.hidden, n_in)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    best, best_ep, best_ck = -1.0, 0, None
    for ep in range(1, a.epochs + 1):
        net.train()
        perm = torch.randperm(len(Xn))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            s = net(Xn[b]).squeeze(-1).masked_fill(~M[b], -1e9)
            lse_gold = torch.logsumexp(s.masked_fill(~G[b], -1e9), 1)  # nhiều người nói: tổng xác suất đúng
            loss = (torch.logsumexp(s, 1) - lse_gold).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        ck = {"state": {k: v.clone() for k, v in net.state_dict().items()}, "mu": mu, "sd": sd, "hidden": a.hidden}
        per = evaluate(va, make_scorer(ck).predict)
        acc = sum(r["box"] for r in per.values()) / sum(r["n"] for r in per.values())
        print(f"  epoch {ep:2d}  loss {loss.item():.3f}  val box acc {acc:.4f}", flush=True)
        if acc > best:
            best, best_ep, best_ck = acc, ep, ck
    print(f"  tốt nhất theo val: epoch {best_ep}, {best:.4f}")
    return best_ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/speaker_labels.json")
    ap.add_argument("--out", default="data/speaker/model.pt")
    ap.add_argument("--out-ctx", default="data/speaker/model_ctx.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", action="store_true", help="train thêm giai đoạn 2 (dùng ckpt giai đoạn 1 có sẵn nếu --reuse-s1)")
    ap.add_argument("--reuse-s1", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    tr, va = list(load_pages(a.labels, "train")), list(load_pages(a.labels, "val"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.reuse_s1 and Path(a.out).exists():
        s1 = SpeakerScorer.load(a.out)
    else:
        X, Y = build(tr)
        print(f"giai đoạn 1: {len(tr)} trang, {len(X)} thoại train")
        ck1 = fit(X, Y, va, SpeakerScorer, a, a.out, N_FEAT)
        torch.save(ck1, a.out)
        s1 = SpeakerScorer(ck1)
    final = s1
    if a.context:
        X, Y = build(tr, s1)
        print(f"giai đoạn 2: {len(X)} thoại train (xác suất giai đoạn 1 tính trên chính tập train)")
        ck2 = fit(X, Y, va, lambda c: ContextScorer(s1, c), a, a.out_ctx, N_FEAT + N_CTX)
        torch.save({"s1": s1_ckpt(a.out), "s2": ck2}, a.out_ctx)
        final = ContextScorer.load(a.out_ctx)
    report(evaluate(va, nearest_in_panel), "VAL baseline nearest cùng panel")
    report(evaluate(va, s1.predict), "VAL giai đoạn 1 (hình học)")
    if a.context:
        report(evaluate(va, final.predict), "VAL giai đoạn 2 (+ ngữ cảnh câu trước/sau)")
    if a.test:
        te = list(load_pages(a.labels, "test"))
        report(evaluate(te, nearest_in_panel), "TEST baseline nearest cùng panel")
        report(evaluate(te, s1.predict), "TEST giai đoạn 1")
        if a.context:
            report(evaluate(te, final.predict), "TEST giai đoạn 2")


def s1_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


if __name__ == "__main__":
    main()
