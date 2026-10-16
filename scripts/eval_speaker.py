"""Đo speaker attribution (gán thoại cho người nói) với nhãn Manga109Dialog đã ánh xạ sang annotations 2026.

    python scripts/build_speaker_labels.py --root ...        # dựng data/speaker_labels.json (một lần)
    python scripts/eval_speaker.py --split val               # baseline hình học, box GT
    python scripts/eval_speaker.py --split test --no-learned

Đơn vị: thực thể = body (kèm mặt nằm trong nó). Đúng ở mức BOX khi thực thể dự đoán là một trong các body người nói
của Dialog; đúng ở mức NHÂN VẬT khi id nhân vật của nó trùng id nhân vật của một người nói (id Other không tính là danh
tính: chỉ đếm khi cùng một body). Chỉ đo text CÓ nhãn Dialog (text không nhãn = không gán, không phải "không người nói").
"""
import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.speaker import entities_from_boxes, features, nearest_any, nearest_in_panel  # noqa: E402


def load_pages(labels_path, split, cache=True):
    """Có cache pickle ở data/speaker/cache_<split>.pkl (dựng đặc trưng cả train mất vài phút). Sinh (title, page_idx, pg, ents, F, D, tp, gold) cho mỗi trang của split (box GT)."""
    cf = Path("data/speaker") / f"cache_{split}.pkl"
    if cache and cf.exists() and cf.stat().st_mtime > Path(labels_path).stat().st_mtime:
        yield from pickle.loads(cf.read_bytes())
        return
    out = list(_load_pages(labels_path, split))
    if cache:
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_bytes(pickle.dumps(out, protocol=4))
    yield from out


def _load_pages(labels_path, split):
    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))["books"]
    for t, bk in sorted(data.items()):
        if bk["split"] != split:
            continue
        other = {i for i, c in bk["characters"].items() if re_other(c)}
        for pi, pg in sorted(bk["pages"].items(), key=lambda kv: int(kv[0])):
            bodies, faces = pg["body"], pg["face"]
            ents = entities_from_boxes([b["box"] for b in bodies], [f["box"] for f in faces])
            texts = [x["box"] for x in pg["text"]]
            F, D, tp, ctx = features([x["box"] for x in pg["frame"]], texts, ents, (pg["w"], pg["h"]))
            body_ent = {b["id"]: ei for ei, e in enumerate(ents) if e["src"][0] == "body"
                        for b in [bodies[e["src"][1]]]}
            ent_char = [bodies[e["src"][1]]["char"] if e["src"][0] == "body" else None for e in ents]
            gold = {}  # chỉ số text -> (tập ent đúng, tập nhân vật đúng, có người nói Other)
            for ti, x in enumerate(pg["text"]):
                sp = pg["links"].get(x["id"])
                if not sp:
                    continue
                es = {body_ent[s] for s in sp if s in body_ent}
                ch = {ent_char[e] for e in es if ent_char[e] and ent_char[e] not in other}
                gold[ti] = (es, ch, any(ent_char[e] in other for e in es))
            if gold:
                yield t, int(pi), pg, ents, F, D, tp, gold, ent_char, other, ctx


def re_other(c):
    return bool(c.get("other"))


def evaluate(pages, predict):
    """predict(F, D, ents, ctx) -> chỉ số ứng viên cho mỗi text. Trả bảng gộp + theo bộ."""
    per = defaultdict(lambda: defaultdict(int))
    for t, pi, pg, ents, F, D, tp, gold, ent_char, other, ctx in pages:
        if len(ents) == 0:
            for ti in gold:
                per[t]["n"] += 1
                per[t]["no_cand"] += 1
            continue
        pred = predict(F, D, ents, ctx)
        for ti, (es, ch, has_other) in gold.items():
            p = int(pred[ti])
            r = per[t]
            r["n"] += 1
            r["box"] += p in es
            r["panel_oracle"] += any(F[ti, e, 6] > 0.5 for e in es)
            if ch:  # có ít nhất một người nói là nhân vật thật (không phải Other)
                r["n_char"] += 1
                r["char"] += ent_char[p] in ch
                r["box_char"] += p in es
    return per


def report(per, title):
    keys = sorted(per)
    tot = defaultdict(int)
    for k in keys:
        for m, v in per[k].items():
            tot[m] += v
    n = max(1, tot["n"])
    print(f"\n== {title} ({len(keys)} bộ, {tot['n']} thoại có nhãn)")
    print(f"  đúng body (micro) {tot['box'] / n:.3f} | trung bình theo bộ "
          f"{np.mean([per[k]['box'] / max(1, per[k]['n']) for k in keys]):.3f} | "
          f"trần 'người nói cùng panel' {tot['panel_oracle'] / n:.3f} | không có ứng viên {tot['no_cand']}")
    nc = max(1, tot["n_char"])
    print(f"  đúng nhân vật (chỉ thoại có người nói không phải Other, n={tot['n_char']}): "
          f"{tot['char'] / nc:.3f} (đúng body trên cùng tập: {tot['box_char'] / nc:.3f})")
    return tot


def calibration(pages, scorer, thrs=(0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Độ chính xác (đúng body) theo ngưỡng xác suất tối thiểu: chọn ngưỡng gợi ý cho Koma trên VAL."""
    pr, ok = [], []
    for t, pi, pg, ents, F, D, tp, gold, ent_char, other, ctx in pages:
        if len(ents) == 0:
            continue
        idx, p = scorer.probs(F, D, ctx)
        best = p.argmax(1)
        for ti, (es, ch, _) in gold.items():
            pr.append(p[ti, best[ti]]); ok.append(int(idx[ti, best[ti]]) in es)
    pr, ok = np.array(pr), np.array(ok)
    print(f"\n{'p >=':>6s} {'phủ':>7s} {'chính xác':>10s}   (n={len(pr)})")
    for th in thrs:
        m = pr >= th
        print(f"{th:6.2f} {m.mean():7.1%} {ok[m].mean():10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/speaker_labels.json")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--by-title", action="store_true")
    ap.add_argument("--model", default=None, help="ckpt giai đoạn 1: thêm dòng mô hình học + bảng độ tin cậy")
    a = ap.parse_args()
    pages = list(load_pages(a.labels, a.split))
    print(f"{len(pages)} trang có nhãn ({a.split})")
    methods = [("nearest toàn trang", nearest_any), ("nearest cùng panel", nearest_in_panel)]
    if a.model:
        from src.speaker import SpeakerScorer
        sc = SpeakerScorer.load(a.model)
        methods.append(("mô hình học", sc.predict))
        calibration(pages, sc)
    for name, fn in methods:
        per = evaluate(pages, fn)
        report(per, name)
        if a.by_title:
            for k in sorted(per):
                print(f"    {k:22s} n={per[k]['n']:5d} box {per[k]['box'] / max(1, per[k]['n']):.3f}")


if __name__ == "__main__":
    main()
