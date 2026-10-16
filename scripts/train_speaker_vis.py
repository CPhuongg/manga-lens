"""Thử nghiệm: CNN nhỏ nhìn vùng ảnh quanh box thoại (thấy đuôi bóng thoại) dự đoán HƯỚNG tới người nói; hướng này
thành đặc trưng thêm cho MLP chấm điểm. CNN train 2-fold theo bộ (dự đoán OOF cho train, trung bình hai CNN cho val/test)
để MLP không học trên đặc trưng quá khớp.

    python scripts/train_speaker_vis.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21" [--test]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_speaker import evaluate, load_pages, report  # noqa: E402
from src.speaker import N_FEAT, SpeakerScorer, nearest_in_panel  # noqa: E402
from train_speaker import build, fit  # noqa: E402

S = 64


def crops(pages, root, cache):
    if cache.exists():
        z = np.load(cache)
        return z["X"], z["V"], z["M"]
    X, V, M = [], [], []
    for pi_, (t, pidx, pg, ents, F, D, tp, gold, ent_char, other, ctx) in enumerate(pages):
        if not gold or not ents:
            continue
        im = Image.open(root / "images" / t / f"{pidx:03d}.jpg").convert("L")
        for ti, (es, _, _) in gold.items():
            b = pg["text"][ti]["box"]
            cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
            side = max(2.5 * max(w, h), 128)
            c = im.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))
            X.append(np.asarray(c.resize((S, S), Image.BILINEAR, reducing_gap=2.0)))
            e = min(es, key=lambda e: np.hypot((ents[e]["box"][0] + ents[e]["box"][2]) / 2 - cx,
                                               (ents[e]["box"][1] + ents[e]["box"][3]) / 2 - cy))
            v = np.array([(ents[e]["box"][0] + ents[e]["box"][2]) / 2 - cx, (ents[e]["box"][1] + ents[e]["box"][3]) / 2 - cy])
            V.append(v / (np.linalg.norm(v) + 1e-6)); M.append((pi_, ti))
    X, V, M = np.stack(X), np.stack(V).astype(np.float32), np.array(M)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, V=V, M=M)
    return X, V, M


def cnn():
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
    return nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 128), nn.Flatten(),
                         nn.Linear(128 * 16, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))


def predict(net, X, dev):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            x = torch.from_numpy(X[i:i + 1024]).float().div(255).sub(0.5).unsqueeze(1).to(dev)
            out.append(net(x).cpu())
    o = torch.cat(out)
    return (o / (o.norm(dim=1, keepdim=True) + 1e-6)).numpy()


def train_cnn(X, V, dev, epochs=8):
    net = cnn().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X)
    Vt = torch.from_numpy(V)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        tot = 0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            x = Xt[b].float().div(255).sub(0.5).unsqueeze(1).to(dev)
            o = net(x)
            loss = (1 - (o / (o.norm(dim=1, keepdim=True) + 1e-6) * Vt[b].to(dev)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"    cnn epoch {ep + 1} loss(1-cos) {tot / len(X):.3f}", flush=True)
    return net


def extend(pages, M, U):
    """Thêm 4 cột vào F: hướng dự đoán (ux, uy), cos(ứng viên), cos - cos lớn nhất. Text không có dự đoán = 0."""
    out = list(pages)
    byp = {}
    for (pi_, ti), u in zip(M, U):
        byp.setdefault(int(pi_), {})[int(ti)] = u
    for pi_, us in byp.items():
        t, pidx, pg, ents, F, D, tp, gold, ent_char, other, ctx = pages[pi_]
        F2 = np.zeros(F.shape[:2] + (N_FEAT + 4,), np.float32)
        F2[:, :, :N_FEAT] = F
        for ti, u in us.items():
            F2[ti, :, N_FEAT:N_FEAT + 2] = u
            F2[ti, :, N_FEAT + 2] = F[ti, :, 0] * u[0] / (np.hypot(F[ti, :, 0], F[ti, :, 1]) + 1e-6) + \
                F[ti, :, 1] * u[1] / (np.hypot(F[ti, :, 0], F[ti, :, 1]) + 1e-6)
            F2[ti, :, N_FEAT + 3] = F2[ti, :, N_FEAT + 2] - F2[ti, :, N_FEAT + 2].max()
        out[pi_] = (t, pidx, pg, ents, F2, D, tp, gold, ent_char, other, ctx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--labels", default="data/speaker_labels.json")
    ap.add_argument("--out", default="data/speaker/model_vis.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sp = {s: list(load_pages(a.labels, s)) for s in ("train", "val") + (("test",) if a.test else ())}
    data = {s: crops(sp[s], a.root, Path(f"data/speaker/vis_{s}.npz")) for s in sp}
    for s, (X, V, M) in data.items():
        print(f"{s}: {len(X)} crop", flush=True)
    Xtr, Vtr, Mtr = data["train"]
    titles = sorted({sp["train"][int(p)][0] for p in Mtr[:, 0]})
    fold = np.array([titles.index(sp["train"][int(p)][0]) % 2 for p in Mtr[:, 0]])
    Utr = np.zeros_like(Vtr)
    Uev = {s: np.zeros_like(data[s][1]) for s in sp if s != "train"}
    for k in (0, 1):
        print(f"CNN fold {k}", flush=True)
        net = train_cnn(Xtr[fold != k], Vtr[fold != k], dev)
        Utr[fold == k] = predict(net, Xtr[fold == k], dev)
        for s in Uev:
            Uev[s] += predict(net, data[s][0], dev)
    for s in Uev:
        Uev[s] /= np.linalg.norm(Uev[s], axis=1, keepdims=True) + 1e-6
    for name, U, V in [("train OOF", Utr, Vtr)] + [(s, Uev[s], data[s][1]) for s in Uev]:
        cos = (U * V).sum(1)
        print(f"  hướng dự đoán {name}: cos trung bình {cos.mean():.3f}, sai lệch góc < 45°: {(cos > 0.707).mean():.1%}")
    ext = {"train": extend(sp["train"], Mtr, Utr), "val": extend(sp["val"], data["val"][2], Uev["val"])}
    X, Y = build(ext["train"])
    print(f"MLP: {len(X)} thoại train")
    ck = fit(X, Y, ext["val"], SpeakerScorer, a, a.out, N_FEAT + 4)
    torch.save(ck, a.out)
    sc = SpeakerScorer(ck)
    report(evaluate(ext["val"], nearest_in_panel), "VAL baseline nearest cùng panel")
    report(evaluate(ext["val"], sc.predict), "VAL MLP + hướng CNN")
    if a.test:
        ext["test"] = extend(sp["test"], data["test"][2], Uev["test"])
        report(evaluate(ext["test"], nearest_in_panel), "TEST baseline nearest cùng panel")
        report(evaluate(ext["test"], sc.predict), "TEST MLP + hướng CNN")


if __name__ == "__main__":
    main()
