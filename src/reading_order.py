"""Thứ tự đọc manga (phải->trái, trên->dưới) bằng XY-cut đệ quy.

Box dạng (x1, y1, x2, y2). Hàm trả về list chỉ số theo thứ tự đọc.
"""
from __future__ import annotations


def _cut(boxes, idxs, axis, tol):
    """Tách idxs thành các nhóm không chồng nhau theo trục; None nếu không tách được."""
    if axis == "y":      # các hàng, từ trên xuống
        order = sorted(idxs, key=lambda i: boxes[i][1])
        lo, hi = 1, 3
        groups, cur, edge = [], [order[0]], boxes[order[0]][hi]
        for i in order[1:]:
            if boxes[i][lo] >= edge - tol:
                groups.append(cur); cur = [i]; edge = boxes[i][hi]
            else:
                cur.append(i); edge = max(edge, boxes[i][hi])
    else:                # các cột, từ PHẢI sang trái (manga)
        order = sorted(idxs, key=lambda i: -boxes[i][2])
        groups, cur, edge = [], [order[0]], boxes[order[0]][0]
        for i in order[1:]:
            if boxes[i][2] <= edge + tol:
                groups.append(cur); cur = [i]; edge = boxes[i][0]
            else:
                cur.append(i); edge = min(edge, boxes[i][0])
    groups.append(cur)
    return groups if len(groups) > 1 else None


def reading_order(boxes, page_size=None, tol_ratio=0.01, rtl=True):
    """boxes: list (x1,y1,x2,y2). page_size: (w,h) để đặt dung sai chồng lấn."""
    if not boxes:
        return []
    if page_size:
        tol = tol_ratio * max(page_size)
    else:
        tol = tol_ratio * max(max(b[2] for b in boxes), max(b[3] for b in boxes))

    def rec(idxs):
        if len(idxs) <= 1:
            return list(idxs)
        for axis in ("y", "x"):
            groups = _cut(boxes, idxs, axis, tol)
            if groups:
                if axis == "x" and not rtl:
                    groups = groups[::-1]
                return [j for g in groups for j in rec(g)]
        # không tách được (panel chéo/chồng lấn): xếp theo tâm
        sgn = -1 if rtl else 1
        return sorted(idxs, key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2,
                                           sgn * (boxes[i][0] + boxes[i][2]) / 2))

    return rec(list(range(len(boxes))))


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def _dist_to_box(pt, b):
    dx = max(b[0] - pt[0], 0, pt[0] - b[2])
    dy = max(b[1] - pt[1], 0, pt[1] - b[3])
    return (dx * dx + dy * dy) ** 0.5


def assign_texts_to_panels(panels, texts):
    """Mỗi text -> panel chứa tâm nó (chọn panel nhỏ nhất); không có thì panel gần nhất."""
    out = []
    for t in texts:
        c = _center(t)
        inside = [i for i, p in enumerate(panels)
                  if p[0] <= c[0] <= p[2] and p[1] <= c[1] <= p[3]]
        if inside:
            out.append(min(inside, key=lambda i: (panels[i][2] - panels[i][0]) *
                                                 (panels[i][3] - panels[i][1])))
        else:
            out.append(min(range(len(panels)), key=lambda i: _dist_to_box(c, panels[i])))
    return out


def order_page(panels, texts, page_size=None, rtl=True):
    """Trả về list các panel theo thứ tự đọc, mỗi panel kèm chỉ số text theo thứ tự đọc.

    [{'panel': idx, 'texts': [idx, ...]}, ...]; text không thuộc panel nào (khi
    không có panel) được gom vào một mục panel=None.
    """
    if not panels:
        order = reading_order(texts, page_size, rtl=rtl)
        return [{"panel": None, "texts": order}] if texts else []
    p_order = reading_order(panels, page_size, rtl=rtl)
    owner = assign_texts_to_panels(panels, texts)
    result = []
    for p in p_order:
        mine = [i for i, o in enumerate(owner) if o == p]
        sub = reading_order([texts[i] for i in mine], page_size, rtl=rtl)
        result.append({"panel": p, "texts": [mine[k] for k in sub]})
    return result
