"""Rút "đuôi" bong bóng thoại từ mask bong bóng (SAM): phần mask lồi mỏng nối bong bóng với nhân vật.

tail_info(mask, text_box) -> vector 7 số [có_đuôi, tip_x, tip_y, dir_x, dir_y, độ_dài_đuôi/căn_diện_tích, diện_tích_mask/diện_tích_text]
(toạ độ ảnh gốc). Đuôi = phần mask còn lại sau khi trừ đi phép MỞ hình thái bằng nhân tròn lớn (bán kính ~0,2 căn diện tích),
lấy mảnh lớn nhất; đầu đuôi = điểm của mảnh xa tâm "thân" bong bóng nhất. Không có đuôi đủ lớn -> có_đuôi=0.
"""
import cv2
import numpy as np

N_TAIL = 7


def tail_info(mask, text_box):
    out = np.zeros(N_TAIL, np.float32)
    ys, xs = np.nonzero(mask)
    ta = max(1.0, (text_box[2] - text_box[0]) * (text_box[3] - text_box[1]))
    if len(xs) < 20:
        return out
    pad = 6
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    sub = mask[y0:y1, x0:x1].astype(np.uint8)
    n, lab = cv2.connectedComponents(sub, connectivity=8)
    tc = (int(np.clip((text_box[0] + text_box[2]) / 2 - x0, 0, sub.shape[1] - 1)),
          int(np.clip((text_box[1] + text_box[3]) / 2 - y0, 0, sub.shape[0] - 1)))
    l = lab[tc[1], tc[0]]
    if l == 0:  # tâm text rơi vào chỗ rỗng (chữ): lấy mảnh lớn nhất
        l = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    comp = (lab == l).astype(np.uint8)
    comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    area = float(filled.sum())
    out[6] = area / ta
    r = max(4, int(0.2 * np.sqrt(area)))
    body = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if body.sum() == 0:
        body = filled
    prot = cv2.morphologyEx(filled & (1 - body), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n2, lab2, stats, _ = cv2.connectedComponentsWithStats(prot, connectivity=8)
    if n2 <= 1:
        return out
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < max(40, 0.015 * area):
        return out
    m = cv2.moments(body, binaryImage=True)
    bc = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
    py, px = np.nonzero(lab2 == k)
    d = np.hypot(px - bc[0], py - bc[1])
    j = int(np.argmax(d))
    tip = np.array([px[j], py[j]], float)
    v = tip - bc
    out[:] = [1.0, tip[0] + x0, tip[1] + y0, v[0] / (d[j] + 1e-6), v[1] / (d[j] + 1e-6), d[j] / np.sqrt(area), out[6]]
    return out
