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
- `tests/test_basic.py` — chạy `python tests/test_basic.py` (không cần dữ liệu), hiện 8/8 pass

## Môi trường (Windows)
- venv ở `.venv/` (Python 3.13); chạy bằng `.venv/Scripts/python.exe`. torch hiện là bản CPU.
- GPU: MX570 A, chỉ 2 GB VRAM -> train mặc định (yolo11s, imgsz 1024, batch 8) nhiều khả năng
  không đủ; cân nhắc yolo11n / batch nhỏ / imgsz thấp, hoặc train trên Colab/Kaggle.
- Console Windows là cp1252: chạy script có `print` tiếng Việt phải đặt `PYTHONUTF8=1`.
- Manga109 để NGOÀI repo: `D:\MINI APP\dataset\Manga109_released_2026_05_21` (4.1 GB), dùng `--root` trỏ tới đó.
  Đã convert với `--copy` -> `data/yolo/` (train 8325 / val 1057 / test 1220 ảnh; train có
  79319 frame, 127783 text, 91321 face, 120841 body). `data/` đã trong `.gitignore`.

## Trạng thái
- ĐÃ test: converter (XML giả), split, reading order (vài layout), CER.
- ĐÃ chạy thật: `convert_manga109.py`, `viz_gt.py` (ARMS p10, LoveHina_vol01 p20).
- CHƯA chạy thật: train, eval_det, eval_ocr, pipeline, app.
  Lỗi đầu tiên khi chạy thật nhiều khả năng nằm ở đây, sửa là việc bình thường.

## Phát hiện từ dữ liệu thật
- Mỗi "trang" Manga109 là TRANG ĐÔI (ảnh ~1654x1170, hai trang liền nhau, đọc phải->trái).
- Thứ tự box trong XML KHÔNG phải thứ tự đọc (LoveHina p20: nhãn panel nhảy lung tung)
  -> không dùng làm GT reading order, phải tự gán nhãn ~50 trang.
- `reading_order()` có tham số `spread` (tự bật khi ảnh rộng hơn cao): tách trang đôi tại rãnh
  giữa, đọc hết trang trước rồi mới sang trang sau; nếu có panel tràn qua rãnh thì không tách.
  Khi gán nhãn/đo Kendall tau cần quyết định tính trên trang đôi hay từng trang đơn.
- Lớp: 0 frame, 1 text, 2 face, 3 body. `text` = bbox CHỮ, không phải đường viền bong bóng.

## Quy ước / lưu ý
- Luôn split theo bộ truyện, không theo trang.
- Không bật `fliplr` khi train (làm sai hướng đọc).
- Không commit ảnh/dataset/weights; không đăng trang truyện có bản quyền (README, demo).
  Thêm `data/`, `runs/`, `*.pt` vào `.gitignore`.

## Việc tiếp theo (theo thứ tự)
1. (Xong) convert + kiểm tra GT bằng `viz_gt.py`.
2. Train `yolo11n/s/m` (+ `rtdetr-l`), ghi bảng mAP theo lớp + thời gian suy luận.
3. `eval_ocr.py` (CER trên GT box) và so với CER trên box do detector dự đoán.
4. Reading order: đo Kendall tau / % trang đúng hoàn toàn; xử lý panel chéo.
5. Mask bong bóng bằng SAM (box prompt từ text box), lọc tay vài chục trang để đo IoU.
6. Phân tích lỗi (~30 trang sai nhiều nhất) + README có bảng kết quả + demo HF Spaces.
