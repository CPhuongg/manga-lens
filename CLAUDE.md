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
- `tests/test_basic.py` — chạy `python tests/test_basic.py` (không cần dữ liệu), hiện 9/9 pass

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
- ĐÃ chạy thật (Kaggle, notebook `kaggle_train.ipynb`): train yolo11s (50 epoch, imgsz 1024,
  batch 16, T4, ~5.5 phút/epoch, best = epoch 31) và `eval_det.py` trên test.
- ĐÃ chạy thật (Kaggle): `eval_ocr.py` (manga-ocr trên GPU, 200 trang test, seed 0).
- CHƯA chạy thật: pipeline, app. Lỗi đầu tiên khi chạy thật nhiều khả năng nằm ở đây,
  sửa là việc bình thường.

### Kết quả OCR (manga-ocr, 200 trang test ngẫu nhiên, seed 0, 2987 text box GT)
`normalize()` mặc định bỏ khác biệt chỉ do dấu câu (NFKC, gộp dãy `…・･．.`, `〜`->`~`); manga-ocr
luôn trả ký tự toàn chiều rộng còn GT hay dùng `…`/`・`. Cờ `--strict` = so từng ký tự (cách cũ).
| Cách so | CER mean theo box | Median | CER theo tổng ký tự | Đúng hoàn toàn |
|---|---|---|---|---|
| Chuẩn hóa dấu câu (mặc định) | **0.034** | 0.000 | **0.039** (37590 ký tự) | **84.3%** |
| `--strict` (cũ) | 0.156 | 0.000 | 0.130 | 58.0% |
=> ~4x "lỗi" trước đó chỉ là khác cách viết dấu câu. CHƯA chạy lại box dự đoán / end-to-end với
chuẩn hóa mới (số cũ theo --strict: dự đoán 0.152, end-to-end 0.176, recall box text 97.2% ở
conf 0.25, FP 125/3028) -> chạy lại `eval_ocr.py --weights ...`. Detector không phải nút cổ chai.

Lỗi thật còn lại (40 box tệ nhất, `--worst 40`):
- Chữ dài trên box rất hẹp và dài (Raphael: chú thích 30x945, 52x1028; MiraiSan/097): CER 0.8-0.9,
  có thể chiếm phần lớn ký tự sai (chưa tính chính xác) -> thử cắt box dài thành đoạn ~vuông.
- Ảo giác trên box gần như chỉ có dấu chấm (ToutaMairimasu/091,092; CER tới 6.0 vì GT chỉ 1 ký tự
  sau chuẩn hóa) -> thử lọc box rất nhỏ / ít mực.
- Chữ viết tay/cách điệu/SFX (Hamlet, Akuhamu, LoveHina), nhiều dòng/cột & chữ lặp (`ポジティブー`).
- Nghi nhãn GT nhiễu (`性病`->`はぁ`, `鳥コも茄子・・・？`), kéo dài âm (`きーっ` vs `きーーーっ`).
Ảnh crop `ocr_worst/` chỉ xem local, KHÔNG đưa lên repo/README (bản quyền).

### Kết quả detector (yolo11s, tập TEST 1220 ảnh, imgsz 1024)
| Lớp | mAP50 | mAP50-95 |
|---|---|---|
| frame | 0.993 | 0.970 |
| text | 0.990 | 0.889 |
| face | 0.896 | 0.457 |
| body | 0.920 | 0.690 |
| **ALL** | **0.949** | **0.752** |

P 0.930 / R 0.904. Suy luận trên T4: 15.8 ms/ảnh (mạng) + 1.9 tiền xử lý + 3.7 hậu xử lý.
Val (epoch 31) mAP50-95 0.746 -> khớp test, không overfit. `face`/`body` thấp ở mAP50-95
(box không khít: nhãn không nhất quán, nhỏ, bị che) -> đáng đưa vào phần phân tích lỗi.
Weights: `weights/best.pt` (local, không commit); đường cong/ma trận nhầm lẫn: `runs_kaggle/`.

### Chạy trên Kaggle
- Dữ liệu tải từ HF gated `hal-utokyo/Manga109` (cần token Read trong Secrets `HF_TOKEN`),
  convert ngay trên máy Kaggle; không upload dataset. `data.yaml` phải trỏ path Linux.
- File văn bản tải từ Kaggle có thể bị bọc markdown (```yaml, dấu `"` quanh từng dòng csv) và
  đổi đuôi (.csv->.txt); `.pt` có thể bị đổi thành .zip (chỉ cần đổi đuôi lại, đừng giải nén).
- Output notebook cũ gắn làm input ở `/kaggle/input/notebooks/<user>/<notebook>/`
  (chứa `yolo/` và `out/runs/detect/runs/<name>/weights/best.pt`).

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
