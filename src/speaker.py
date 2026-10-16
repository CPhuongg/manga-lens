"""Speaker attribution: gán mỗi box `text` cho một thực thể nhân vật (body, hoặc mặt đứng riêng) trên cùng trang.

Thực thể = {"box": body (hoặc mặt nếu không có body bao nó), "face": box mặt hoặc None}. Ứng viên của một text
là mọi thực thể trên trang; `features()` dựng đặc trưng cho từng cặp (text, ứng viên), `nearest*` là baseline,
`SpeakerNet` là bộ chấm điểm học được (softmax trong các ứng viên của một text).
Chỉ dùng hình học + thứ tự đọc (chưa dùng hướng đuôi bóng thoại, vì chưa có mask bong bóng).
"""
import numpy as np

from src.reading_order import assign_texts_to_panels, order_page


def _c(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _dist_to_box(p, b):
    dx = max(b[0] - p[0], 0, p[0] - b[2])
    dy = max(b[1] - p[1], 0, p[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


def entities_from_boxes(bodies, faces):
    """Mỗi body -> một thực thể (kèm mặt nhỏ nhất có tâm nằm trong body, nếu có); mặt không nằm trong body
    nào -> thực thể chỉ có mặt. Trả [{"box","face","src":("body",i)|("face",j)}]."""
    ents, used = [], set()
    for i, b in enumerate(bodies):
        inside = [j for j, f in enumerate(faces)
                  if b[0] <= _c(f)[0] <= b[2] and b[1] <= _c(f)[1] <= b[3]]
        j = min(inside, key=lambda j: _area(faces[j])) if inside else None
        ents.append({"box": list(b), "face": list(faces[j]) if j is not None else None, "src": ("body", i)})
    for j, f in enumerate(faces):
        cx, cy = _c(f)
        if not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in bodies):
            ents.append({"box": list(f), "face": list(f), "src": ("face", j)})
    return ents


N_FEAT = 28


def features(panels, texts, ents, size):
    """Trả (F[nt, ne, N_FEAT], dist[nt, ne], text_panel[nt], ctx) cho một trang. size = (W, H).
    ctx = {"prev": [...], "next": [...]}: chỉ số text đứng trước/sau theo thứ tự đọc (-1 nếu không có)."""
    W, H = size
    s = float(H)
    nt, ne = len(texts), len(ents)
    F = np.zeros((nt, ne, N_FEAT), np.float32)
    if nt == 0 or ne == 0:
        return F, np.zeros((nt, ne)), [None] * nt, {"prev": [-1] * nt, "next": [-1] * nt}
    pbox = panels if panels else [[0, 0, W, H]]
    tp = assign_texts_to_panels(pbox, texts)
    ep = assign_texts_to_panels(pbox, [e["box"] for e in ents])
    layout = order_page(panels, texts, size) if panels else order_page([], texts, size)
    rank_page, rank_panel, n_panel = {}, {}, {}
    k = 0
    for item in layout:
        for r, ti in enumerate(item["texts"]):
            rank_page[ti], rank_panel[ti], n_panel[ti] = k, r, len(item["texts"])
            k += 1
    inv = [None] * nt
    for ti, r in rank_page.items():
        inv[r] = ti
    D = np.zeros((nt, ne))
    for ti, t in enumerate(texts):
        tc = _c(t)
        for ei, e in enumerate(ents):
            D[ti, ei] = _dist_to_box(tc, e["box"])
    same = np.array([[tp[ti] == ep[ei] for ei in range(ne)] for ti in range(nt)])
    nearest = D.argmin(1)
    for ti, t in enumerate(texts):
        tc, ta, pa = _c(t), _area(t), _area(pbox[tp[ti]])
        d = D[ti]
        order_all = d.argsort().argsort()
        same_idx = np.where(same[ti])[0]
        rank_same = np.full(ne, 9.0)
        if len(same_idx):
            rank_same[same_idx[d[same_idx].argsort()]] = np.arange(len(same_idx))
        dmin = d.min()
        dmin_same = d[same_idx].min() if len(same_idx) else dmin
        r = rank_page.get(ti, 0)
        prev_t = inv[r - 1] if r > 0 else None
        next_t = inv[r + 1] if r + 1 < nt and inv[r + 1] is not None else None
        for ei, e in enumerate(ents):
            b = e["box"]
            bc, ba = _c(b), max(1.0, _area(b))
            bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
            f = e["face"]
            ep_box = pbox[ep[ei]]
            F[ti, ei] = [
                (bc[0] - tc[0]) / s, (bc[1] - tc[1]) / s,
                d[ei] / s, np.hypot(bc[0] - tc[0], bc[1] - tc[1]) / s,
                float(b[0] <= tc[0] <= b[2] and b[1] <= tc[1] <= b[3]),
                _inter(t, b) / max(1.0, ta),
                float(same[ti, ei]), _inter(b, pbox[tp[ti]]) / ba,
                float(order_all[ei]), rank_same[ei], float(len(same_idx)),
                (d[ei] - dmin) / s, ((d[ei] - dmin_same) / s) if same[ti, ei] else 1.0,
                min(2.0, ba / max(1.0, pa)), np.log(ba / (W * H)),
                float(f is not None), (_area(f) / ba) if f is not None else 0.0,
                (_dist_to_box(tc, f) / s) if f is not None else d[ei] / s,
                min(2.0, ta / max(1.0, pa)), np.log(max(1.0, t[3] - t[1]) / max(1.0, t[2] - t[0])),
                (rank_panel.get(ti, 0) / max(1, n_panel.get(ti, 1) - 1)) if n_panel.get(ti, 1) > 1 else 0.0,
                float(n_panel.get(ti, 1)),
                float(r == 0),
                float(prev_t is not None and nearest[prev_t] == ei),
                float(next_t is not None and nearest[next_t] == ei),
                float(nearest[ti] == ei),
                float(np.clip((tc[0] - b[0]) / bw, -1, 2)), float(np.clip((tc[1] - b[1]) / bh, -1, 2)),
            ]
    prev = [inv[rank_page[ti] - 1] if rank_page.get(ti, 0) > 0 else -1 for ti in range(nt)]
    nxt = [inv[rank_page[ti] + 1] if ti in rank_page and rank_page[ti] + 1 < nt and inv[rank_page[ti] + 1] is not None
           else -1 for ti in range(nt)]
    return F, D, tp, {"prev": prev, "next": nxt}


def nearest_any(F, D, ents=None, ctx=None):
    """Baseline 0: thực thể gần nhất (khoảng cách từ tâm text tới box) trên cả trang."""
    return D.argmin(1)


def nearest_in_panel(F, D, ents=None, ctx=None):
    """Baseline 1: thực thể gần nhất TRONG CÙNG PANEL với text; không có ứng viên nào cùng panel -> gần nhất cả trang."""
    same = F[:, :, 6] > 0.5
    Dm = np.where(same, D, np.inf)
    out = Dm.argmin(1)
    none = ~same.any(1)
    out[none] = D.argmin(1)[none]
    return out


TOPK = 20  # chỉ chấm điểm K ứng viên gần nhất (theo khoảng cách tới box) của mỗi text


def topk_idx(D, k=TOPK):
    """Chỉ số K ứng viên gần nhất cho từng text: [nt, min(k, ne)]."""
    return np.argsort(D, axis=1, kind="stable")[:, :k]


class SpeakerScorer:
    """MLP chấm điểm từng cặp (text, ứng viên); softmax trong các ứng viên của một text.
    ckpt = {"state": state_dict, "mu": [F], "sd": [F], "hidden": int}."""

    def __init__(self, ckpt):
        import torch
        self.torch = torch
        self.mu = np.asarray(ckpt["mu"], np.float32)
        self.sd = np.asarray(ckpt["sd"], np.float32)
        self.net = make_net(ckpt["hidden"], len(ckpt["mu"]))
        self.net.load_state_dict(ckpt["state"])
        self.net.eval()

    @classmethod
    def load(cls, path):
        import torch
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def probs(self, F, D, ctx=None):
        """-> (idx[nt, k], prob[nt, k]): xác suất cho từng ứng viên trong top-K."""
        idx = topk_idx(D)
        X = np.take_along_axis(F, idx[:, :, None], axis=1)
        X = (X - self.mu) / self.sd
        with self.torch.no_grad():
            s = self.net(self.torch.from_numpy(X.astype(np.float32))).squeeze(-1)
            p = self.torch.softmax(s, dim=1).numpy()
        return idx, p

    def predict(self, F, D, ents=None, ctx=None):
        idx, p = self.probs(F, D, ctx)
        return idx[np.arange(len(idx)), p.argmax(1)]


def make_net(hidden=128, n_in=None):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(n_in or N_FEAT, hidden), nn.ReLU(), nn.Dropout(0.1),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden, 1))


def attribute(dets, size, scorer):
    """Gán người nói cho từng text của một trang từ kết quả detector (MangaPipeline.detect).
    dets: {"frame","text","face","body": [{"box":[..],"score":..}]}. Trả (ents, [(ent_idx|None, prob)] theo thứ tự dets["text"])."""
    panels = [d["box"] for d in dets["frame"]]
    texts = [d["box"] for d in dets["text"]]
    ents = entities_from_boxes([d["box"] for d in dets["body"]], [d["box"] for d in dets["face"]])
    if not texts or not ents:
        return ents, [(None, 0.0)] * len(texts)
    F, D, _, ctx = features(panels, texts, ents, size)
    idx, p = scorer.probs(F, D, ctx)
    best = p.argmax(1)
    return ents, [(int(idx[i, best[i]]), float(p[i, best[i]])) for i in range(len(texts))]


N_CTX = 5


def stage2_inputs(F, idx, p1, ctx):
    """Đặc trưng giai đoạn 2: F gốc + [p của chính text, p của text TRƯỚC/SAU (theo thứ tự đọc) cho cùng thực thể,
    có trước, có sau]. p1 = xác suất giai đoạn 1 trong top-K (idx, p1: [nt, k]). Trả [nt, k, N_FEAT + N_CTX]."""
    nt, k = idx.shape
    ne = F.shape[1]
    full = np.zeros((nt, ne), np.float32)
    np.put_along_axis(full, idx, p1, axis=1)
    X = np.take_along_axis(F, idx[:, :, None], axis=1)
    pv = np.array(ctx["prev"]); nx = np.array(ctx["next"])
    p_prev = np.where((pv >= 0)[:, None], full[np.maximum(pv, 0)], 0.0)
    p_next = np.where((nx >= 0)[:, None], full[np.maximum(nx, 0)], 0.0)
    ex = np.stack([p1, np.take_along_axis(p_prev, idx, 1), np.take_along_axis(p_next, idx, 1),
                   np.repeat((pv >= 0)[:, None], k, 1), np.repeat((nx >= 0)[:, None], k, 1)], axis=-1)
    return np.concatenate([X, ex.astype(np.float32)], axis=-1)


class ContextScorer:
    """Hai giai đoạn: SpeakerScorer (hình học) -> mô hình thứ hai dùng thêm xác suất của câu thoại liền trước/sau."""

    def __init__(self, s1, ckpt2):
        import torch
        self.torch = torch
        self.s1 = s1
        self.mu = np.asarray(ckpt2["mu"], np.float32)
        self.sd = np.asarray(ckpt2["sd"], np.float32)
        self.net = make_net(ckpt2["hidden"], N_FEAT + N_CTX)
        self.net.load_state_dict(ckpt2["state"])
        self.net.eval()

    @classmethod
    def load(cls, path):
        import torch
        c = torch.load(path, map_location="cpu", weights_only=False)
        return cls(SpeakerScorer(c["s1"]), c["s2"])

    def probs(self, F, D, ctx):
        idx, p1 = self.s1.probs(F, D)
        X = (stage2_inputs(F, idx, p1, ctx) - self.mu) / self.sd
        with self.torch.no_grad():
            s = self.net(self.torch.from_numpy(X.astype(np.float32))).squeeze(-1)
            return idx, self.torch.softmax(s, dim=1).numpy()

    def predict(self, F, D, ents=None, ctx=None):
        idx, p = self.probs(F, D, ctx)
        return idx[np.arange(len(idx)), p.argmax(1)]
