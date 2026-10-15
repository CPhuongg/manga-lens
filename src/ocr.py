"""OCR cho text box. Tiếng Nhật: manga-ocr. Có thể thay backend cho tiếng Việt."""
from __future__ import annotations

_mocr = None


def ocr_japanese(pil_img) -> str:
    global _mocr
    if _mocr is None:
        from manga_ocr import MangaOcr  # lazy: nặng, tải model lần đầu
        _mocr = MangaOcr()
    return _mocr(pil_img)


def crop(pil_img, box, pad=4):
    x1, y1, x2, y2 = box
    w, h = pil_img.size
    return pil_img.crop((max(0, int(x1) - pad), max(0, int(y1) - pad),
                         min(w, int(x2) + pad), min(h, int(y2) + pad)))


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(preds, gts, thr: float = 0.5):
    """Ghép tham lam theo IoU giảm dần. Trả về [(idx_pred, idx_gt)], mỗi box dùng tối đa 1 lần."""
    cand = sorted(((iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
                  reverse=True)
    used_p, used_g, out = set(), set(), []
    for v, i, j in cand:
        if v < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j))
    return out


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize(s: str) -> str:
    return "".join(s.split())


def cer(pred: str, gt: str) -> float:
    pred, gt = normalize(pred), normalize(gt)
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, gt) / len(gt)
