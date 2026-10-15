# manga-lens — ghi chú bàn giao cho Claude Code

Dự án cá nhân (luyện CV, không gắn với nghiên cứu): phân tích 1 trang manga -> detect
panel/text/face/body, sắp thứ tự đọc (phải->trái), OCR thoại. Phạm vi đã chốt: **mức Mid**
(chưa làm speaker attribution / clustering nhân vật).

## Cấu trúc
- `scripts/convert_manga109.py` — XML Manga109 -> YOLO, split theo BỘ TRUYỆN (seed 0, 80/10/10), ghi `data/yolo/data.yaml` + `{split}_titles.txt`
- `scripts/train.py`, `scripts/eval_det.py` — train/eval Ultralytics (mAP theo lớp)
- `scripts/eval_ocr.py` — CER của manga-ocr trên text box GT
- `scripts/viz_gt.py` — vẽ GT + số thứ tự trong XML
- `scripts/label_order.py` — công cụ gán nhãn thứ tự đọc (Gradio, bấm box theo thứ tự đọc; nguyên trang
  đôi; tiếp tục được). `--mode text` (mặc định, 30 trang): thứ tự THOẠI trên box `text` GT = chuẩn đo chính
  -> `data/reading_order_text_labels.json`. `--mode panel` (50 trang): thứ tự panel, chỉ để chẩn đoán
  -> `data/reading_order_labels.json`. Mẫu lấy từ tập test, seed 0.
- `scripts/eval_order_pred.py` — như eval_order nhưng với box detector dự đoán (tách 4 tổ hợp panel|text GT/dự đoán)
- `scripts/eval_order.py --mode text|panel` — Kendall tau + % trang đúng. Chế độ text so: qua panel
  (pipeline hiện tại) / XY-cut thẳng trên text (không panel) / baseline -> trả lời "có cần panel để sắp thoại".
- `src/reading_order.py` — XY-cut đệ quy RTL + gán text vào panel (+ `kendall_tau`)
- `src/ocr.py`, `src/pipeline.py`, `app.py` — OCR, pipeline end-to-end, demo Gradio
- `tests/test_basic.py` — chạy `python tests/test_basic.py` (không cần dữ liệu), hiện 9/9 pass

## Môi trường (Windows)
- venv ở `.venv/` (Python 3.13); chạy bằng `.venv/Scripts/python.exe`. torch hiện là bản CUDA (2.14.1+cu130, cuda=True), gradio 6.30.
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
- ĐÃ chạy thật (local, MX570 2GB): `src/pipeline.py` + `app.py` (Gradio, gọi API /run OK) trên 1 trang
  test (Akuhamu_008, trang đôi 1654x1170): 9 frame / 23 text / 15 face / 19 body, không lỗi; ~18 s
  (GPU) / ~23 s (CPU) cho cả detect+OCR 23 box. Thứ tự đọc đúng khi xem ảnh: tách trang đôi (phải
  trước), panel phải->trái theo hàng, thoại phải->trái trong panel. Mới kiểm tra 1 trang bằng mắt,
  CHƯA đo định lượng (Kendall tau) -> xem việc 4. Tiêu đề trên đầu trang (パニック×2, 逆効果) được gán
  vào panel gần nhất như thoại đầu tiên (chưa tách loại).
- CHƯA làm: mask SAM, README, bảng so sánh model (yolo11n/m), đo end-to-end gộp thứ tự + OCR trên toàn bộ trang.

### Kết quả OCR (manga-ocr, 200 trang test ngẫu nhiên, seed 0, 2987 text box GT)
`normalize()` mặc định bỏ khác biệt chỉ do dấu câu (NFKC, gộp dãy `…・･．.`, `〜`->`~`); manga-ocr
luôn trả ký tự toàn chiều rộng còn GT hay dùng `…`/`・`. Cờ `--strict` = so từng ký tự (cách cũ).
| Cách so | CER mean theo box | Median | CER theo tổng ký tự | Đúng hoàn toàn |
|---|---|---|---|---|
| Chuẩn hóa dấu câu (mặc định) | **0.034** | 0.000 | **0.039** (37590 ký tự) | **84.3%** |
| `--strict` (cũ) | 0.156 | 0.000 | 0.130 | 58.0% |
=> ~4x "lỗi" trước đó chỉ là khác cách viết dấu câu.

Theo từng nhóm box (chuẩn hóa dấu câu, cùng 200 trang, detector yolo11s conf>=0.25, ghép IoU>=0.5):
| Nhóm | Số box | CER mean | Median | Đúng hoàn toàn |
|---|---|---|---|---|
| GT box (giới hạn trên của OCR) | 2987 | 0.034 | 0.000 | 84.3% |
| Box dự đoán (chỉ box ghép được với GT) | 2903 | 0.027 | 0.000 | 84.4% |
| **End-to-end** (GT bị sót tính CER 1.0) | 2987 | **0.055** | 0.000 | 82.1% |

Recall box text 97.2%, FP 125/3028 box dự đoán. "Box dự đoán" thấp hơn GT vì loại 84 box bị sót
(hay là box khó) -> lệch mẫu, không phải detector đọc tốt hơn. Chênh GT -> end-to-end (0.021) gần
như toàn bộ do 84 box bị sót (2.8%) -> ~38% lỗi end-to-end đến từ recall detector chứ không phải
OCR. Cách tính hơi bi quan (detector gộp 2 box sát nhau thì cả hai bị tính sót, IoU<0.5).

Thử hạ conf 0.25 -> 0.15: ghép thêm 13 box GT (recall 97.2% -> 97.6%), e2e CER 0.055 -> 0.051,
nhưng FP 125 -> 179 (+43%; e2e CER KHÔNG tính FP, FP thật sẽ đẩy chữ rác vào kết quả) -> ~4 FP
cho mỗi box thu về, KHÔNG đáng; giữ conf 0.25.

Chẩn đoán 84 box GT bị sót ở conf 0.25 (`--diag`): low_conf 44 (52%, detector tìm đúng chỗ,
IoU 0.6-0.93, nhưng conf 0.01-0.25), merged 19 (23%, detector gộp nhiều GT làm một: Manga109
chia box theo cột, detector gộp cả cột -> phần lớn là khác cách chia box, e2e hơi bi quan),
absent 13 (15%, detector không thấy = chỉ 0.4% tổng box), split 5, loc 3.
Thử lấp chỗ trống (`--conf-low 0.05`, nhận box conf thấp nếu không chồng box đã nhận):
thêm 156 box, chỉ 27 đúng (17%) + 129 FP mới (FP 125 -> 254); recall 97.2% -> 98.1%, e2e CER
0.055 -> 0.048. ~4.8 FP/box đúng, tệ hơn cả hạ conf toàn cục -> KHÔNG dùng. Box conf thấp phần
lớn là rác (độ chính xác 17% so với 96% ở conf>=0.25) => ngưỡng 0.25 đã hợp lý.
Kết luận: dừng tinh chỉnh detector cho text box. Việc còn lại có thể làm: metric many-to-one cho
nhóm `merged`; train yolo11m nếu muốn tăng mAP (không cần cho recall).

Lỗi thật còn lại (40 box tệ nhất, `--worst 40`):
- Chữ dài trên box rất hẹp và dài (Raphael: chú thích 30x945, 52x1028; MiraiSan/097): CER 0.8-0.9,
  có thể chiếm phần lớn ký tự sai (chưa tính chính xác) -> thử cắt box dài thành đoạn ~vuông.
- Ảo giác trên box gần như chỉ có dấu chấm (ToutaMairimasu/091,092; CER tới 6.0 vì GT chỉ 1 ký tự
  sau chuẩn hóa) -> thử lọc box rất nhỏ / ít mực.
- Chữ viết tay/cách điệu/SFX (Hamlet, Akuhamu, LoveHina), nhiều dòng/cột & chữ lặp (`ポジティブー`).
- Nghi nhãn GT nhiễu (`性病`->`はぁ`, `鳥コも茄子・・・？`), kéo dài âm (`きーっ` vs `きーーーっ`).
Ảnh crop `ocr_worst/` chỉ xem local, KHÔNG đưa lên repo/README (bản quyền).

### Kết quả reading order (box GT, trang đôi đọc PHẢI trước; nhãn tự gán bằng `scripts/label_order.py`)
Thứ tự THOẠI, 30 trang test (chuẩn đo chính), `scripts/eval_order.py --mode text`:
| Cách | Tau mean | Đúng hoàn toàn |
|---|---|---|
| **qua panel, `panel_tol=0.05` (pipeline)** | **0.986** | **28/30 (93.3%)** |
| qua panel, dung sai cũ 0.01 | 0.882 | 20/30 (66.7%) |
| XY-cut thẳng trên text (không panel) | 0.940 | 18/30 (60%) |
| thẳng, tắt tách trang đôi | 0.606 | 1/30 |
| baseline (y rồi phải->trái) | 0.410 | 0/30 |
Nguyên nhân lỗi cũ: viền panel thật hay chồng/lệch nhau 30-60 px (2-4% bề rộng trang đôi) > dung sai 1%
của XY-cut -> không cắt được -> rơi về sắp theo tâm y -> đọc panel trái trước phải (vd SeisinkiVulnus/026).
`order_page(..., panel_tol=0.05)` dùng dung sai riêng cho panel; dung sai text (trong panel) gần như không
ảnh hưởng. Panel: 16 trang nhãn độc lập cũng tăng 14/16 -> 15/16 (tau 0.994). LƯU Ý: 0.05 chọn trên chính
30 trang này (in-sample); leave-one-title-out 28/30 vs 20/30 -> không giống overfit, nhưng mẫu nhỏ.
Còn sai: WarewareHaOniDearu/090 (layout lệch hàng, nhãn đọc theo cột; tau 0.60) -> layout nhập nhằng.
Với box DETECTOR dự đoán (yolo11s conf 0.25, `scripts/eval_order_pred.py`, cùng 30 trang; tau/đúng tính trên text
ghép được với GT, IoU>=0.5): panel recall 98.9% / precision 95.9% (279/291 dự đoán); text recall 98.1%
(411/419), FP 12 (2.8%).
| panel \| text | Tau mean | Đúng hoàn toàn |
|---|---|---|
| GT \| GT (tham chiếu) | 0.986 | 28/30 |
| dự đoán \| GT | 0.986 | 28/30 |
| GT \| dự đoán | 0.987 | 29/30 |
| **dự đoán \| dự đoán (pipeline thật)** | **0.987** | **29/30 (97%)** |
| dự đoán, thẳng (không panel) | 0.942 | 19/30 (63%) |
=> detector gần như không làm giảm độ chính xác thứ tự (vẫn là in-sample cho panel_tol; mẫu 30 trang;
8 text bị sót không nằm trong số tau, 12 box thừa chưa đo ảnh hưởng). Trang sai duy nhất: Ware/090.
Nhãn panel (`data/reading_order_labels.json`, 16 trang) ban đầu gán TRANG TRÁI TRƯỚC (ngược quy ước,
tau ~ 0 với thuật toán); đã hoán đổi hai nửa bằng mã, bản gốc ở `data/reading_order_labels.leftfirst.bak.json`.
Gán nhãn mới luôn đọc hết TRANG PHẢI rồi mới sang trang trái.

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
