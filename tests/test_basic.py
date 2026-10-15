"""Chạy: python tests/test_basic.py   (không cần Manga109 hay GPU)"""
import sys
import tempfile
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


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
