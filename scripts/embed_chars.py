"""Trích embedding DINOv2 cho crop face/body GT (data/characters.json) -> data/emb/<model>_<split>.npz.

Mỗi npz: book, idx (vị trí trong items của bộ), cls ('face'/'body'), emb (N, D) float16 đã chuẩn hoá L2.
Dùng:
    python scripts/embed_chars.py --splits val test
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def crop(img, box, margin):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - margin * h, y2 + margin * h
    c = img.crop((max(0, int(x1)), max(0, int(y1)), min(img.width, int(x2)), min(img.height, int(y2))))
    return c if min(c.size) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=Path("data/characters.json"), type=Path)
    ap.add_argument("--yolo", default=Path("data/yolo"), type=Path)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", default=Path("data/emb"), type=Path)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=64)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(dev).eval().half() if dev == "cuda" \
        else AutoModel.from_pretrained(args.model).eval()
    mean = torch.tensor(proc.image_mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=dev).view(1, 3, 1, 1)
    books = json.loads(args.chars.read_text(encoding="utf-8"))
    tag = args.model.split("/")[-1]
    args.out.mkdir(parents=True, exist_ok=True)


    parts = args.out / f"{tag}_parts"   # checkpoint theo bộ: dừng giữa chừng rồi chạy lại sẽ bỏ qua bộ đã xong
    parts.mkdir(exist_ok=True)
    crops, meta, out_e = [], [], []

    def flush():
        if not crops:
            return
        px = torch.from_numpy(np.stack(crops)).to(dev).permute(0, 3, 1, 2).float() / 255
        px = ((px - mean) / std).to(model.dtype)
        with torch.no_grad():
            o = model(pixel_values=px).last_hidden_state
        out_e.append(torch.cat([o[:, 0], o[:, 1:].mean(1)], dim=1).float().cpu().numpy())  # CLS + mean patch
        crops.clear()

    for split in args.splits:
        for title, book in books.items():
            if book["split"] != split:
                continue
            part = parts / f"{split}_{title}.npz"
            if part.exists():
                print(split, title, "(đã có, bỏ qua)", flush=True)
                continue
            meta.clear(); out_e.clear()
            by_page = defaultdict(list)
            for i, it in enumerate(book["items"]):
                by_page[it["page"]].append(i)
            for page, ids in by_page.items():
                p = args.yolo / "images" / split / f"{title}_{page:03d}.jpg"
                if not p.exists():
                    continue
                img = Image.open(p).convert("RGB")
                for i in ids:
                    it = book["items"][i]
                    c = crop(img, it["box"], args.margin)
                    if c is None:
                        continue
                    crops.append(np.asarray(c.resize((224, 224), Image.BILINEAR, reducing_gap=2.0)))
                    meta.append((i, it["cls"]))
                    if len(crops) >= args.bs:
                        flush()
            flush()
            emb = np.concatenate(out_e)
            tmp = part.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, idx=np.array([m[0] for m in meta]), cls=np.array([m[1] for m in meta]),
                                emb=emb.astype(np.float16))
            tmp.replace(part)
            print(split, title, len(meta), flush=True)
        # gộp các bộ của split thành 1 file
        bs, ix, cl, em = [], [], [], []
        for title, book in books.items():
            if book["split"] != split:
                continue
            z = np.load(parts / f"{split}_{title}.npz")
            bs += [title] * len(z["idx"]); ix.append(z["idx"]); cl.append(z["cls"]); em.append(z["emb"])
        emb = np.concatenate(em)
        np.savez_compressed(args.out / f"{tag}_{split}.npz", book=np.array(bs), idx=np.concatenate(ix),
                            cls=np.concatenate(cl), emb=emb, dim_half=emb.shape[1] // 2)
        print("saved", split, emb.shape, flush=True)


if __name__ == "__main__":
    main()
