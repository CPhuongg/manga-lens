"""Fine-tune DINOv2-small bằng metric learning (SupCon) để gom nhân vật theo face hoặc body.

Nhãn = (bộ, nhân vật) lấy từ data/characters.json (scripts/extract_characters.py); bỏ 'Other'.
Train trên split train, chọn epoch/tham số theo val (retrieval mAP mỗi epoch, rồi eval_cluster --no-test).
Test CHỈ đụng khi chạy --eval-only (một lần, cho cấu hình đã chọn trên val).

Crop được cache ra uint8 (face 126x126, body 224x112) trong --cache; ảnh đọc từ <root>/images/<bộ>/<trang>.jpg.

    python scripts/train_reid.py --root <Manga109> --cls face --tag f_a --epochs 10
    python scripts/eval_cluster.py --tag f_a --feat face --emb-dir out/reid --no-test        # val
    python scripts/train_reid.py --root <Manga109> --cls face --tag f_a --eval-only          # ghi emb val+test
    python scripts/eval_cluster.py --tag f_a --feat face --emb-dir out/reid                  # test
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent))
from eval_cluster import retrieval  # noqa: E402

SIZE = {"face": (126, 126), "body": (224, 112)}  # (H, W), chia hết cho patch 14
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- dữ liệu
def build_crops(books, split, root, cls, cache, margin=0.1, max_books=0):
    """Cắt + resize toàn bộ box `cls` của split -> memmap uint8 + meta. Có cache nên chạy lại rất nhanh."""
    cache.mkdir(parents=True, exist_ok=True)
    f_img, f_meta = cache / f"{cls}_{split}.npy", cache / f"{cls}_{split}.json"
    if f_img.exists() and f_meta.exists():
        return np.load(f_img, mmap_mode="r"), json.loads(f_meta.read_text(encoding="utf-8"))
    H, W = SIZE[cls]
    titles = [t for t, b in books.items() if b["split"] == split]
    if max_books:
        titles = titles[:max_books]
    n = sum(1 for t in titles for it in books[t]["items"] if it["cls"] == cls)
    arr = np.lib.format.open_memmap(f_img, mode="w+", dtype=np.uint8, shape=(n, H, W, 3))
    meta, k = [], 0
    t0 = time.time()
    for ti, title in enumerate(titles):
        book = books[title]
        by_page = defaultdict(list)
        for i, it in enumerate(book["items"]):
            if it["cls"] == cls:
                by_page[it["page"]].append(i)
        for page, ids in by_page.items():
            p = root / "images" / title / f"{page:03d}.jpg"
            if not p.exists():
                continue
            img = Image.open(p).convert("RGB")
            for i in ids:
                it = book["items"][i]
                x1, y1, x2, y2 = it["box"]
                w, h = x2 - x1, y2 - y1
                box = (max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h)),
                       min(img.width, int(x2 + margin * w)), min(img.height, int(y2 + margin * h)))
                if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                    continue
                arr[k] = np.asarray(img.crop(box).resize((W, H), Image.BILINEAR, reducing_gap=2.0))
                meta.append({"book": title, "idx": i, "char": it["char"],
                             "other": book["characters"][it["char"]]["other"]})
                k += 1
        print(f"  [{split}/{cls}] {ti + 1}/{len(titles)} {title} ({k} crop, {time.time() - t0:.0f}s)", flush=True)
    arr.flush()
    del arr
    arr = np.load(f_img, mmap_mode="r")[:k]  # k <= n nếu có crop bị bỏ
    f_meta.write_text(json.dumps(meta), encoding="utf-8")
    return arr, meta


# ---------------------------------------------------------------- mô hình
class ReID(nn.Module):
    def __init__(self, name="facebook/dinov2-small", emb=256, unfreeze=4):
        super().__init__()
        self.bb = AutoModel.from_pretrained(name)
        d = self.bb.config.hidden_size
        for p in self.bb.parameters():
            p.requires_grad = False
        layers = self.bb.encoder.layer
        for blk in layers[len(layers) - unfreeze:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.bb.layernorm.parameters():
            p.requires_grad = unfreeze > 0
        self.head = nn.Linear(2 * d, emb)

    def forward(self, x):
        o = self.bb(pixel_values=x).last_hidden_state
        f = torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=1)
        return F.normalize(self.head(f), dim=-1)


def prep(xu8, dev, aug):
    """uint8 (B,H,W,3) -> tensor chuẩn hoá trên GPU; aug = lật, zoom/dịch, sáng/tương phản, xám."""
    x = torch.from_numpy(np.ascontiguousarray(xu8)).to(dev).permute(0, 3, 1, 2).float() / 255
    B, _, H, W = x.shape
    if aug:
        s = torch.empty(B, device=dev).uniform_(0.8, 1.0)
        flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
        tx = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
        ty = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0] = s * flip; theta[:, 0, 2] = tx
        theta[:, 1, 1] = s; theta[:, 1, 2] = ty
        x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False),
                          padding_mode="border", align_corners=False)
        b = torch.empty(B, 1, 1, 1, device=dev).uniform_(-0.15, 0.15)
        c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * c + m + b).clamp(0, 1)
        g = (torch.rand(B, 1, 1, 1, device=dev) < 0.2).float()
        x = g * x.mean(1, keepdim=True).expand(-1, 3, -1, -1) + (1 - g) * x
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def embed_all(model, arr, dev, bs=256):
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model(prep(arr[i:i + bs], dev, False)).float().cpu())
    return torch.cat(out).numpy()


def supcon(z, y, temp):
    sim = z @ z.T / temp
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None]) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    ok = n_pos > 0
    return -((logp * pos).sum(1)[ok] / n_pos[ok]).mean()


def val_retrieval(emb, meta, min_count=2):
    """Recall@1 / mAP trung bình theo bộ (bỏ Other, giữ nhân vật có >= min_count box) — như eval_cluster."""
    by_book = defaultdict(list)
    for i, m in enumerate(meta):
        if not m["other"]:
            by_book[m["book"]].append(i)
    r1s, maps = [], []
    for b, ids in by_book.items():
        y = np.array([meta[i]["char"] for i in ids])
        u, cnt = np.unique(y, return_counts=True)
        keep = np.isin(y, u[cnt >= min_count])
        if keep.sum() < 5:
            continue
        r1, mp = retrieval(emb[np.array(ids)[keep]], y[keep])
        r1s.append(r1); maps.append(mp)
    return float(np.mean(r1s)), float(np.mean(maps))


def save_emb(path, emb, meta, cls):
    np.savez_compressed(path, book=np.array([m["book"] for m in meta]), idx=np.array([m["idx"] for m in meta]),
                        cls=np.array([cls] * len(meta)), emb=emb.astype(np.float16), dim_half=emb.shape[1] // 2)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="thư mục Manga109 (có images/)")
    ap.add_argument("--chars", default=Path("data/characters.json"), type=Path)
    ap.add_argument("--cls", default="face", choices=["face", "body"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=Path("out/reid"), type=Path)
    ap.add_argument("--cache", default=Path("data/crops"), type=Path)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--unfreeze", type=int, default=4, help="số block cuối của backbone được train")
    ap.add_argument("--lr-bb", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--P", type=int, default=32, help="số nhân vật mỗi batch")
    ap.add_argument("--K", type=int, default=4, help="số mẫu mỗi nhân vật")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--iters", type=int, default=0, help="bước mỗi epoch (0 = n_mẫu/(P*K))")
    ap.add_argument("--max-books", type=int, default=0, help="debug: giới hạn số bộ mỗi split")
    ap.add_argument("--eval-only", action="store_true", help="nạp best.pt, ghi embedding val+TEST")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    books = json.loads(args.chars.read_text(encoding="utf-8"))
    out = args.out / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model = ReID(args.model, args.emb, args.unfreeze).to(dev)

    splits = ["val", "test"] if args.eval_only else ["train", "val"]
    data = {s: build_crops(books, s, args.root, args.cls, args.cache, max_books=args.max_books) for s in splits}

    if args.eval_only:
        model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
        for s in ("val", "test"):
            arr, meta = data[s]
            emb = embed_all(model, arr, dev)
            save_emb(args.out / f"{args.tag}_{s}.npz", emb, meta, args.cls)
            print(s, "->", args.out / f"{args.tag}_{s}.npz", emb.shape)
        return

    # --- danh sách mẫu train theo nhân vật
    tr_arr, tr_meta = data["train"]
    tr_arr = np.ascontiguousarray(tr_arr)  # nạp vào RAM
    groups = defaultdict(list)
    for i, m in enumerate(tr_meta):
        if not m["other"]:
            groups[(m["book"], m["char"])].append(i)
    groups = {k: np.array(v) for k, v in groups.items() if len(v) >= 2}
    keys = list(groups)
    w = np.array([np.sqrt(len(groups[k])) for k in keys]); w /= w.sum()
    n_samples = sum(len(v) for v in groups.values())
    args.P = min(args.P, len(keys))  # smoke với ít bộ có thể ít nhân vật hơn P
    iters = args.iters or max(1, n_samples // (args.P * args.K))
    print(f"train: {len(keys)} nhân vật, {n_samples} crop, {iters} bước/epoch", flush=True)

    params = [{"params": [p for p in model.bb.parameters() if p.requires_grad], "lr": args.lr_bb},
              {"params": model.head.parameters(), "lr": args.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_bb, args.lr_head],
                                                total_steps=max(args.epochs * iters, 20), pct_start=0.1)  # >=20: OneCycle chia 0 nếu quá ít bước (smoke)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    va_arr, va_meta = data["val"]

    # mốc 0: chưa fine-tune (head ngẫu nhiên) — chỉ để biết val ban đầu
    best, log = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for it in range(iters):
            sel = rng.choice(len(keys), size=args.P, replace=False, p=w)
            idx, lab = [], []
            for j, k in enumerate(sel):
                g = groups[keys[k]]
                pick = rng.choice(g, size=args.K, replace=len(g) < args.K)
                idx.extend(pick.tolist()); lab.extend([j] * args.K)
            order = np.argsort(idx)  # đọc tuần tự hơn; nhãn đi theo
            idx = np.array(idx)[order]; lab = torch.tensor(np.array(lab)[order], device=dev)
            x = prep(tr_arr[idx], dev, True)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                z = model(x)
            loss = supcon(z.float(), lab, args.temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        emb = embed_all(model, va_arr, dev)
        r1, mp = val_retrieval(emb, va_meta)
        log.append({"epoch": ep, "loss": tot / iters, "val_R1": r1, "val_mAP": mp})
        print(f"epoch {ep}/{args.epochs} loss {tot / iters:.4f} val R@1 {r1:.3f} mAP {mp:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if mp > best:
            best = mp
            torch.save(model.state_dict(), out / "best.pt")
        (out / "log.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "log": log}))
    print(f"best val mAP {best:.3f} (epoch {max(log, key=lambda r: r['val_mAP'])['epoch']})")

    # embedding val của epoch tốt nhất để chạy eval_cluster --no-test
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    save_emb(args.out / f"{args.tag}_val.npz", embed_all(model, va_arr, dev), va_meta, args.cls)
    print("val embedding ->", args.out / f"{args.tag}_val.npz")


if __name__ == "__main__":
    main()
