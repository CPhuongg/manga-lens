"""MLP chấm điểm người nói + đặc trưng ĐUÔI bong bóng (mask SAM -> src/bubble.py). Cần cache từ scripts/sam_tails.py.

    python scripts/train_speaker_tail.py                 # train 87 bộ, chọn epoch trên val, báo val
    python scripts/train_speaker_tail.py --test          # thêm báo TEST một lần
So sánh công bằng với train_speaker.py (cùng hyper-parameter, cùng hạt giống); in cả mô hình hình học thuần để đối chiếu.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_speaker import evaluate, load_pages, report  # noqa: E402
from src.speaker import N_FEAT, SpeakerScorer, _dist_to_box, nearest_in_panel  # noqa: E402
from train_speaker import build, fit  # noqa: E402

N_TF = 8


def tail_feats(T, tbox, ents, H):
    """T: [7] của một text; trả [ne, N_TF]."""
    ne = len(ents)
    f = np.zeros((ne, N_TF), np.float32)
    tc = np.array([(tbox[0] + tbox[2]) / 2, (tbox[1] + tbox[3]) / 2])
    has = T[0] > 0.5
    tip = T[1:3] if has else tc
    d = np.array([_dist_to_box(tip, e["box"]) for e in ents]) / H
    f[:, 0] = float(has)
    f[:, 1] = T[5] if has else 0.0
    f[:, 2] = np.clip(np.log(max(T[6], 1e-3)), -1, 5)
    f[:, 3] = d
    f[:, 4] = d.argsort().argsort()
    f[:, 5] = d - d.min()
    if has:
        for i, e in enumerate(ents):
            c = np.array([(e["box"][0] + e["box"][2]) / 2, (e["box"][1] + e["box"][3]) / 2])
            v1, v2 = c - tc, c - tip
            f[i, 6] = float(T[3:5] @ v1 / (np.linalg.norm(v1) + 1e-6))
            f[i, 7] = float(T[3:5] @ v2 / (np.linalg.norm(v2) + 1e-6))
    return f


def extend(pages, cache, only=False):
    """only=True: chỉ giữ các trang có đặc trưng đuôi (khi cache chỉ chạy một phần trang)."""
    z = np.load(cache)
    out = list(pages)
    rows = {}
    for (pi, ti), T in zip(z["M"], z["T"]):
        rows.setdefault(int(pi), {})[int(ti)] = T
    for pi, ts in rows.items():
        t, pidx, pg, ents, F, D, tp, gold, ent_char, other, ctx = pages[pi]
        F2 = np.zeros(F.shape[:2] + (N_FEAT + N_TF,), np.float32)
        F2[:, :, :N_FEAT] = F
        for ti, T in ts.items():
            F2[ti, :, N_FEAT:] = tail_feats(T, pg["text"][ti]["box"], ents, pg["h"])
        out[pi] = (t, pidx, pg, ents, F2, D, tp, gold, ent_char, other, ctx)
    return [out[pi] for pi in sorted(rows)] if only else out


class Sliced:
    """Bọc SpeakerScorer chỉ nhìn N cột đặc trưng đầu (đối chứng hình học thuần trên cùng tập trang train)."""

    def __init__(self, sc, n):
        self.sc, self.n = sc, n

    def predict(self, F, D, ents=None, ctx=None):
        return self.sc.predict(F[:, :, :self.n], D, ents, ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/speaker_labels.json")
    ap.add_argument("--out", default="data/speaker/model_tail.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    tr = extend(list(load_pages(a.labels, "train")), "data/speaker/tail_train.npz", only=True)
    va = extend(list(load_pages(a.labels, "val")), "data/speaker/tail_val.npz")
    T = np.load("data/speaker/tail_val.npz")["T"]
    print(f"val: có đuôi {T[:, 0].mean():.1%}, diện tích mask/text trung vị {np.median(T[:, 6]):.1f}", flush=True)
    X, Y = build(tr)
    print(f"MLP + đuôi: {len(tr)} trang / {len(X)} thoại train", flush=True)
    ck = fit(X, Y, va, SpeakerScorer, a, a.out, N_FEAT + N_TF)
    torch.save(ck, a.out)
    sc = SpeakerScorer(ck)
    print("đối chứng: hình học thuần, cùng các trang train", flush=True)
    ck0 = fit(X[:, :, :N_FEAT], Y, [(*p[:4], p[4][:, :, :N_FEAT], *p[5:]) for p in va], SpeakerScorer, a,
              a.out + ".control", N_FEAT)
    ctl = Sliced(SpeakerScorer(ck0), N_FEAT)
    report(evaluate(va, nearest_in_panel), "VAL baseline nearest cùng panel")
    report(evaluate(va, SpeakerScorer.load("data/speaker/model.pt").predict), "VAL hình học thuần, train ĐỦ 87 bộ (model.pt)")
    report(evaluate(va, ctl.predict), "VAL hình học thuần, cùng tập trang train với mô hình đuôi")
    report(evaluate(va, sc.predict), "VAL hình học + đuôi bong bóng")
    if a.test:
        te = extend(list(load_pages(a.labels, "test")), "data/speaker/tail_test.npz")
        report(evaluate(te, nearest_in_panel), "TEST baseline nearest cùng panel")
        report(evaluate(te, SpeakerScorer.load("data/speaker/model.pt").predict), "TEST hình học thuần, train đủ")
        report(evaluate(te, ctl.predict), "TEST hình học thuần, cùng tập trang train")
        report(evaluate(te, sc.predict), "TEST hình học + đuôi bong bóng")


if __name__ == "__main__":
    main()
