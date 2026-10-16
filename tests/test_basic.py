"""Chạy: python tests/test_basic.py   (không cần Manga109 hay GPU)"""
import sys
import tempfile

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from convert_manga109 import parse_book, split_titles, to_yolo_line  # noqa: E402
from src.ocr import cer  # noqa: E402
from src.reading_order import order_page, reading_order  # noqa: E402


def test_reading_order_classic_layout():
    # Trang 1000x1400: cột phải 1 panel cao (A); cột trái 2 panel xếp dọc (B trên, C dưới)
    A, B, C = (520, 0, 1000, 1400), (0, 0, 480, 650), (0, 700, 480, 1400)
    assert reading_order([C, A, B], (1000, 1400)) == [1, 2, 0]  # A, B, C


def test_reading_order_rows():
    # 2 hàng x 2 cột: phải trước trái, hàng trên trước hàng dưới
    tl, tr, bl, br = (0, 0, 480, 600), (520, 0, 1000, 600), (0, 650, 480, 1300), (520, 650, 1000, 1300)
    assert reading_order([tl, tr, bl, br], (1000, 1300)) == [1, 0, 3, 2]
    assert reading_order([tl, tr, bl, br], (1000, 1300), rtl=False) == [0, 1, 2, 3]


def test_reading_order_spread():
    # Trang đôi 1600x1100 (kiểu Manga109): 2 hàng thẳng hàng giữa hai trang.
    # Phải đọc hết trang phải (R1,R2) rồi mới sang trang trái (L1,L2), không xen kẽ theo hàng.
    R1, R2 = (850, 0, 1600, 500), (850, 550, 1600, 1100)
    L1, L2 = (0, 0, 780, 500), (0, 550, 780, 1100)
    assert reading_order([L1, R1, L2, R2], (1600, 1100)) == [1, 3, 0, 2]
    assert reading_order([L1, R1, L2, R2], (1600, 1100), rtl=False) == [0, 2, 1, 3]
    assert reading_order([L1, R1, L2, R2], (1600, 1100), spread=False) == [1, 0, 3, 2]  # lỗi cũ


def test_reading_order_spread_wide_panel():
    # Panel tràn cả trang đôi: không tách rãnh, nó là một hàng riêng đọc trước
    W, R, L = (0, 0, 1600, 400), (850, 450, 1600, 1100), (0, 450, 780, 1100)
    assert reading_order([L, R, W], (1600, 1100)) == [2, 1, 0]


def test_order_page_assignment():
    panels = [(0, 0, 480, 600), (520, 0, 1000, 600)]
    texts = [(50, 50, 120, 200), (600, 50, 700, 200), (800, 300, 900, 500)]
    res = order_page(panels, texts, (1000, 600))
    assert [r["panel"] for r in res] == [1, 0]
    assert res[0]["texts"] == [1, 2]  # text 1 nằm cao hơn nên đọc trước (tách hàng trước)
    assert res[1]["texts"] == [0]


def test_order_page_overlapping_panels():
    # Hình học thật (SeisinkiVulnus/026, trang đôi 1654x1170): panel 0 và 1 chồng nhau 56 px (3.4%) nên
    # không có đường cắt sạch; phương án dự phòng (hàng theo cạnh trên, phải->trái) phải đọc panel PHẢI (0)
    # trước panel trái (1). (Cách cũ sắp theo tâm y đọc ngược [1, 0, ...].)
    F = [(1234, 2, 1565, 512), (887, 1, 1290, 507), (0, 2, 762, 1169), (899, 526, 1567, 1165)]
    T = [(1250, 20, 1300, 100), (950, 20, 1000, 100), (100, 50, 200, 150), (1000, 700, 1100, 800)]
    assert reading_order(F, (1654, 1170), tol_ratio=0.01) == [0, 1, 3, 2]
    assert [r["panel"] for r in order_page(F, T, (1654, 1170))] == [0, 1, 3, 2]
    assert [r["panel"] for r in order_page(F, T, (1654, 1170), panel_tol=0.01)] == [0, 1, 3, 2]


def test_reading_order_overlapping_panels_fallback():
    # Hình học thật (trang đơn 1280x1791, panel borderless chồng lên nhau): không có đường cắt sạch.
    # Cách cũ (sắp theo tâm y) đọc panel TRÁI (idx 1) trước panel PHẢI (idx 4) dù cùng bắt đầu ở mép trên.
    F = [(0, 1164, 566, 1787), (0, 0, 564, 1145), (469, 245, 1280, 1791), (1059, 197, 1280, 1620),
         (477, 0, 1244, 1573)]
    assert reading_order(F, (1280, 1791), tol_ratio=0.05) == [4, 1, 3, 2, 0]


def test_cer():
    assert cer("こんにちは", "こんにちは") == 0
    assert abs(cer("こんにちわ", "こんにちは") - 0.2) < 1e-9
    # khác biệt chỉ do cách viết dấu câu thì không tính là lỗi
    assert cer("．．．", "…") == 0 and cer("．．．．．．", "………") == 0 and cer("．．", "・・・") == 0
    assert cer("ＧＯＡＬ！", "GOAL!") == 0 and cer("わう．．．！？", "わう…!?") == 0
    assert cer("な～～～", "な〜〜〜") == 0
    assert cer("．．．", "…", strict=True) > 0  # --strict giữ cách so cũ
    assert cer("そう．．．そうだな", "････････") > 0  # ảo giác chữ vẫn bị tính là sai


def test_match_boxes():
    from src.ocr import iou, match_boxes
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    gts = [(0, 0, 10, 10), (50, 50, 60, 60)]
    preds = [(51, 51, 61, 61), (0, 0, 10, 11), (200, 200, 210, 210)]  # pred 2 là box thừa
    assert sorted(match_boxes(preds, gts)) == [(0, 1), (1, 0)]
    two = match_boxes([(0, 0, 10, 10), (0, 0, 10, 10)], [(0, 0, 10, 10)])
    assert len(two) == 1 and two[0][1] == 0  # 1 GT chỉ ghép 1 lần


def test_kendall_tau():
    from src.reading_order import kendall_tau
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0
    assert abs(kendall_tau([0, 1, 2, 3], [1, 0, 2, 3]) - (4 / 6)) < 1e-9  # 1 cặp đảo / 6 cặp
    assert kendall_tau([5], [5]) == 1.0 and kendall_tau([], []) == 1.0


def test_add_low_conf():
    from src.ocr import add_low_conf
    hi = [(0, 0, 100, 100)]
    cands = [((200, 0, 300, 100), 0.2),     # chỗ trống -> nhận
             ((210, 0, 310, 100), 0.1),     # chồng box vừa nhận -> loại (không trùng)
             ((10, 10, 50, 50), 0.2),       # nằm trong box cao -> loại
             ((90, 0, 190, 100), 0.15)]     # chồng 10% box cao: dưới ngưỡng 0.3 -> nhận
    got = add_low_conf(hi, cands, 0.3)
    assert (200, 0, 300, 100) in got and (90, 0, 190, 100) in got and len(got) == 2
    assert add_low_conf(hi, [((0, 0, 100, 100), 0.2)], 0.3) == []   # trùng hẳn -> loại
    assert add_low_conf([], []) == []


def test_diagnose_miss():
    from src.ocr import diagnose_miss
    gt = (0, 0, 100, 100)
    d = lambda preds: diagnose_miss(gt, preds, 0.25)[0]
    assert d([((0, 0, 200, 100), 0.9)]) == "merged"        # box lớn gộp phủ GT
    assert d([((0, 0, 100, 100), 0.1)]) == "low_conf"      # chỉ khớp ở conf thấp
    assert d([((0, 0, 40, 100), 0.9), ((50, 0, 90, 100), 0.9)]) == "split"  # xé mảnh
    assert d([((50, 50, 150, 150), 0.9)]) == "loc"       # chồng lệch (IoU ~0.14)
    assert d([((300, 300, 400, 400), 0.9)]) == "absent"
    assert d([]) == "absent"


def test_converter():
    xml = """<book title="T"><characters/><pages>
      <page index="0" width="1000" height="1400">
        <frame id="a" xmin="0" ymin="0" xmax="500" ymax="700"/>
        <text id="b" xmin="10" ymin="10" xmax="110" ymax="210">あ</text>
        <face id="c" xmin="1" ymin="1" xmax="1" ymax="1"/>
      </page></pages></book>"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "T.xml"
        p.write_text(xml, encoding="utf-8")
        pg = parse_book(p)[0]
    assert pg["w"] == 1000 and len(pg["boxes"]) == 3
    assert to_yolo_line("frame", 0, 0, 500, 700, 1000, 1400) == "0 0.250000 0.250000 0.500000 0.500000"
    assert to_yolo_line("face", 1, 1, 1, 1, 1000, 1400) is None  # box suy biến bị bỏ


def test_split_by_title():
    s = split_titles([f"t{i}" for i in range(109)])
    allt = s["train"] + s["val"] + s["test"]
    assert len(allt) == 109 and len(set(allt)) == 109
    assert len(s["test"]) == 11 and len(s["val"]) == 11


def test_speaker_baselines():
    from src.speaker import entities_from_boxes, features, nearest_any, nearest_in_panel
    panels = [[0, 0, 500, 500], [500, 0, 1000, 500]]
    bodies = [[300, 100, 400, 400], [600, 100, 700, 400]]  # body 0 trong panel 0, body 1 trong panel 1
    faces = [[320, 110, 380, 170], [900, 10, 950, 60]]     # mặt 1 không nằm trong body nào -> thực thể riêng
    ents = entities_from_boxes(bodies, faces)
    assert len(ents) == 3 and ents[0]["face"] == faces[0] and ents[1]["face"] is None
    texts = [[460, 200, 490, 260]]  # sát mép panel 0; body 1 (panel 1) cách 110px, body 0 cách 60px
    F, D, tp, ctx = features(panels, texts, ents, (1000, 500))
    assert F.shape[:2] == (1, 3) and tp == [0] and ctx == {"prev": [-1], "next": [-1]}
    assert nearest_any(F, D)[0] == 0 and nearest_in_panel(F, D)[0] == 0
    texts2 = [[505, 200, 530, 260]]  # panel 1: nearest_in_panel chọn thực thể trong panel 1 dù body 0 cũng gần
    F, D, tp, _ = features(panels, texts2, ents, (1000, 500))
    assert tp == [1] and nearest_in_panel(F, D)[0] in (1, 2)


def test_bubble_tail():
    import cv2
    from src.bubble import tail_info
    m = np.zeros((400, 400), np.uint8)
    cv2.ellipse(m, (200, 150), (90, 70), 0, 0, 360, 1, -1)            # bong bóng
    cv2.fillPoly(m, [np.array([[170, 205], [215, 205], [230, 330]])], 1)  # đuôi chỉ xuống dưới
    t = tail_info(m.astype(bool), [150, 110, 250, 190])
    assert t[0] == 1 and t[4] > 0.8 and abs(t[3]) < 0.5 and t[2] > 280     # đầu đuôi ở phía dưới
    m2 = np.zeros((400, 400), np.uint8)
    cv2.ellipse(m2, (200, 150), (90, 70), 0, 0, 360, 1, -1)
    assert tail_info(m2.astype(bool), [150, 110, 250, 190])[0] == 0         # bong bóng tròn: không đuôi


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
