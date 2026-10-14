# manga-lens

Phân tích trang manga: phát hiện panel / text (bong bóng thoại) / mặt / thân, sắp thứ tự đọc,
OCR thoại. Dự án cá nhân luyện CV (mức "Mid").

## Chạy

```bash
pip install -r requirements.txt
python tests/test_basic.py                                   # kiểm tra nhanh, không cần dữ liệu

# 1) Xin Manga109 tại manga109.org (học thuật), giải nén, rồi:
python scripts/convert_manga109.py --root /path/Manga109 --out data/yolo   # --copy nếu Windows

# 2) Kiểm tra GT trước khi tin (xem thứ tự XML có phải thứ tự đọc không)
python scripts/viz_gt.py --root /path/Manga109 --title ARMS --page 10

# 3) Train + đánh giá
python scripts/train.py --model yolo11s.pt --epochs 50 --batch 8
python scripts/eval_det.py --weights runs/yolo11s/weights/best.pt
python scripts/eval_ocr.py --root /path/Manga109 --titles data/yolo/test_titles.txt

# 4) Demo
python app.py --weights runs/yolo11s/weights/best.pt
```

## Việc cần làm tiếp (thứ tự gợi ý)
1. Baseline: yolo11n / s / m (+ rtdetr-l) -> bảng mAP theo lớp, ghi cả thời gian suy luận.
2. Phân tích lỗi: lưu ~30 trang sai nhiều nhất (text bị cắt, panel chồng, body bị che) -> đưa vào README.
3. Reading order: tự gán nhãn thứ tự đọc ~50 trang (nếu xác nhận thứ tự XML không phải thứ tự đọc),
   đo Kendall tau / tỉ lệ trang đúng hoàn toàn. Thử cải tiến XY-cut (panel chéo).
4. Mask bong bóng: SAM (box prompt từ text box) rồi lọc tay vài chục trang để đo IoU.
5. OCR: CER trên GT box vs trên box do detector dự đoán (đo lỗi lan truyền).

## Lưu ý
- Chia train/val/test THEO BỘ TRUYỆN (đã làm sẵn trong converter), không chia theo trang.
- Không bật lật ngang (fliplr) khi train: lật làm sai hướng đọc/bố cục.
- Không đăng ảnh trang truyện có bản quyền lên repo/demo; chỉ đẩy code + trọng số, và tuân thủ điều khoản Manga109.
