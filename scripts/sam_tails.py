"""Chạy SAM (mobile_sam) với box thoại (nới 3x) -> mask bong bóng -> đặc trưng đuôi (src/bubble.py).
Cache theo split: data/speaker/tail_<split>.npz {M: [N,2] (chỉ số trang trong load_pages, chỉ số text), T: [N,7]}.

    python scripts/sam_tails.py --root "D:/MINI APP/dataset/Manga109_released_2026_05_21" --splits val train test
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_speaker import load_pages  # noqa: E402
from src.bubble import N_TAIL, tail_info  # noqa: E402

PROMPT_SCALE = 3.0


def grow(b, f, W, H):
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return [max(0, cx - w / 2), max(0, cy - h / 2), min(W, cx + w / 2), min(H, cy + h / 2)]


def masks(sam, img, bboxes, fill=None):
    r = sam(img, bboxes=bboxes, verbose=False)[0]
    if r.masks is None:
        return [np.zeros(fill, bool)] if fill else []
    return list(r.masks.data.cpu().numpy().astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--labels", default="data/speaker_labels.json")
    ap.add_argument("--splits", nargs="+", default=["val", "train", "test"])
    ap.add_argument("--weights", default="weights/mobile_sam.pt")
    ap.add_argument("--train-pages", type=int, default=0,
                    help="chỉ chạy N trang train ngẫu nhiên (seed 0) cho nhanh; 0 = tất cả. train_speaker_tail chỉ dùng các trang có cache")
    a = ap.parse_args()
    from ultralytics import SAM
    sam = SAM(a.weights)
    for split in a.splits:
        out = Path(f"data/speaker/tail_{split}.npz")
        if out.exists():
            print(f"{split}: đã có cache", flush=True)
            continue
        pages = list(load_pages(a.labels, split))
        M, T, t0, n_fix = [], [], time.time(), 0
        keep = None
        if split == "train" and a.train_pages:
            keep = set(np.random.default_rng(0).permutation(len(pages))[:a.train_pages].tolist())
        for pi, (t, pidx, pg, ents, F, D, tp, gold, ent_char, other, ctx) in enumerate(pages):
            if not gold or not ents or (keep is not None and pi not in keep):
                continue
            tis = list(gold)
            W, H = pg["w"], pg["h"]
            img = str(a.root / "images" / t / f"{pidx:03d}.jpg")
            bbs = [grow(pg["text"][ti]["box"], PROMPT_SCALE, W, H) for ti in tis]
            ms = masks(sam, img, bbs)
            if len(ms) != len(tis):  # ultralytics bỏ box suy biến: chạy từng box để giữ đúng thứ tự
                ms = []
                for bb in bbs:
                    one = masks(sam, img, [bb])
                    ms.append(one[0] if one else np.zeros((H, W), bool))
                n_fix += 1
            for k, ti in enumerate(tis):
                M.append((pi, ti)); T.append(tail_info(ms[k], pg["text"][ti]["box"]))
            if pi % 200 == 0:
                print(f"  {split} {pi}/{len(pages)} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(out, M=np.array(M), T=np.stack(T).astype(np.float32))
        T = np.stack(T)
        print(f"{split}: {n_fix} trang phải chạy lại từng box")
        print(f"{split}: {len(T)} thoại, có đuôi {T[:, 0].mean():.1%}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
