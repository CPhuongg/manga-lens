# manga-lens — ghi chú bàn giao cho Claude Code

Dự án cá nhân (luyện CV, không gắn với nghiên cứu): phân tích 1 trang manga -> detect
panel/text/face/body, sắp thứ tự đọc (phải->trái), OCR thoại. Phạm vi đã chốt: **mức Mid**
(chưa làm speaker attribution / clustering nhân vật).

## Cấu trúc
- `scripts/convert_manga109.py` — XML Manga109 -> YOLO, split theo BỘ TRUYỆN (seed 0, 80/10/10), ghi `data/yolo/data.yaml` + `{split}_titles.txt`
- `scripts/train.py`, `scripts/eval_det.py` — train/eval Ultralytics (mAP theo lớp)
- `scripts/eval_ocr.py` — CER của manga-ocr trên text box GT
- `scripts/viz_gt.py` — vẽ GT + số thứ tự trong XML
- `src/reading_order.py` — XY-cut đệ quy RTL + gán text vào panel
- `src/ocr.py`, `src/pipeline.py`, `app.py` — OCR, pipeline end-to-end, demo Gradio
- `tests/test_basic.py` — chạy `python tests/test_basic.py` (không cần dữ liệu), hiện 6/6 pass

## Trạng thái
- ĐÃ test: converter (XML giả), split, reading order (vài layout), CER.
- CHƯA chạy thật: train, eval_det, eval_ocr, pipeline, app (chưa có Manga109/GPU khi viết).
  Lỗi đầu tiên khi chạy thật nhiều khả năng nằm ở đây, sửa là việc bình thường.
- Lớp: 0 frame, 1 text, 2 face, 3 body. `text` = bbox CHỮ, không phải đường viền bong bóng.

## Quy ước / lưu ý
- Luôn split theo bộ truyện, không theo trang.
- Không bật `fliplr` khi train (làm sai hướng đọc).
- Thứ tự box trong XML Manga109 CHƯA được xác nhận là thứ tự đọc -> chạy `viz_gt.py` kiểm tra
  trước khi dùng làm GT reading order; nếu không trùng thì tự gán nhãn ~50 trang.
- Không commit ảnh/dataset/weights; không đăng trang truyện có bản quyền (README, demo).
  Thêm `data/`, `runs/`, `*.pt` vào `.gitignore`.

## Việc tiếp theo (theo thứ tự)
1. Có Manga109 -> `convert_manga109.py` -> `viz_gt.py` kiểm tra vài trang.
2. Train `yolo11n/s/m` (+ `rtdetr-l`), ghi bảng mAP theo lớp + thời gian suy luận.
3. `eval_ocr.py` (CER trên GT box) và so với CER trên box do detector dự đoán.
4. Reading order: đo Kendall tau / % trang đúng hoàn toàn; xử lý panel chéo.
5. Mask bong bóng bằng SAM (box prompt từ text box), lọc tay vài chục trang để đo IoU.
6. Phân tích lỗi (~30 trang sai nhiều nhất) + README có bảng kết quả + demo HF Spaces.
