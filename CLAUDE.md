# manga-lens — ghi chú bàn giao cho Claude Code

Dự án cá nhân (luyện CV, không gắn với nghiên cứu): phân tích 1 trang manga -> detect
panel/text/face/body, sắp thứ tự đọc (phải->trái), OCR thoại. Phạm vi ban đầu là **mức Mid**; đã MỞ RỘNG sang
**nhận dạng nhân vật** (gom mặt/thân theo nhân vật trong một bộ truyện, xem mục "Nhận dạng nhân vật").
Speaker attribution (gán thoại cho nhân vật nói) đã làm mức hình học (xem mục "Speaker attribution"); chưa dùng hướng đuôi bóng thoại.

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
- `scripts/run_pages.py` — chạy pipeline trên truyện thật (ảnh/thư mục/.cbz) -> `out/<tên>/` (gitignored): ảnh chú thích, transcript.txt, result.json
- `scripts/export_koma.py` — tạo `.mtp.json` cho app dịch Koma (`D:/MINI APP/manga_trans_pair`, 1 file HTML + serve.py :8777):
  model khoanh ô thoại + điền chữ Nhật (OCR) + đánh số theo thứ tự đọc. Dùng: Koma -> Nhập .json -> Gắn lại ảnh.
  Koma lưu ô dạng CHUẨN HOÁ 0-1 (x,y,w,h theo kích thước trang), không phải px; chỉ dùng cho chương MỚI ("Ghi đè" của
  Koma xoá bản dịch cũ). Đã kiểm tra bằng cách nhập vào Koma chạy ở cổng riêng 8790 (kho IndexedDB tách theo cổng).
- Nhận dạng nhân vật (re-ID):
  - `scripts/extract_characters.py` — giữ thuộc tính `character` của face/body trong XML -> `data/characters.json` (kèm split theo bộ)
  - `scripts/train_reid.py` — fine-tune DINOv2-small bằng SupCon (face hoặc body); cache crop ở `data/crops/`;
    test chỉ đụng khi `--eval-only`. `load_ckpt` nạp `best.pt` từ Kaggle (đổi tên khoá attention giữa phiên bản transformers)
  - `kaggle_reid.ipynb` — train trên Kaggle, Run All một lần (clone repo, 3 cấu hình face tự chọn theo val, rồi body)
  - `scripts/embed_chars.py` — embedding DINOv2 thô (baseline, chưa fine-tune), checkpoint theo bộ
  - `scripts/eval_cluster.py` — gom theo bộ + metric (BCubed/pairwise/ARI/NMI, Recall@1, mAP); ngưỡng chọn trên val, `--no-test`, `--json`
  - `scripts/eval_cluster_pred.py` — như trên nhưng với box face/body do DETECTOR dự đoán (cache `data/pred/`); cách đo A/B
  - `scripts/eval_fused.py`, `scripts/eval_fused_learned.py` — ghép face+body (khoảng cách đặt tay / MLP học trên val)
  - `scripts/run_characters.py` — chạy trên chương thật (ảnh/thư mục/.cbz) -> `out/<tên>/characters/` (characters.txt/json, sheet_XX.jpg, `--draw`)
- Speaker attribution (gán thoại cho thực thể/nhân vật nói):
  - `data/dialog/` — Manga109Dialog (XML, CC BY 4.0, tải bằng sparse-clone `manga109/public-annotations`; KHÔNG commit)
  - `scripts/build_speaker_labels.py` — ánh xạ nhãn Dialog (annotations 2021) sang annotations 2026 -> `data/speaker_labels.json`
  - `src/speaker.py` — thực thể (body + mặt), đặc trưng cặp (text, ứng viên), baseline nearest, `SpeakerScorer` (MLP listwise), `attribute()`
  - `scripts/eval_speaker.py` (box GT, `--model`, `--by-title`, bảng độ tin cậy), `scripts/eval_speaker_pred.py` (box detector, cách A/B),
    `scripts/train_speaker.py` (`--context` = giai đoạn 2 thử nghiệm, KHÔNG dùng), `scripts/train_speaker_vis.py` (thử nghiệm CNN hướng, kết quả âm tính) -> `data/speaker/model.pt` (gitignored)
  - `export_koma.py --characters out/<tên>/characters/characters.json` ghi `[NVk]` vào ghi chú (`note`) của từng ô
- `app.py` — demo Gradio (mặc định `weights/best.pt`, chỉnh conf/imgsz, bật/tắt OCR, không lưu ảnh)
- `space/` + `scripts/make_space.py` — đóng gói Hugging Face Space vào `build/space/` (gitignored); đẩy lên bằng
  `hf upload <user>/manga-lens build/space . --repo-type space`. CPU mô phỏng (máy local): detect 3.7 s (imgsz 1024)
  / 1.8 s (640), thêm OCR 23 box -> 22.6 s/trang. TRƯỚC KHI public: xác nhận điều khoản Manga109 có cho phát hành
  trọng số huấn luyện từ nó không (chưa xác nhận); demo không kèm ví dụ ảnh truyện.
- `src/reading_order.py` — XY-cut đệ quy RTL + gán text vào panel (+ `kendall_tau`)
- `src/ocr.py`, `src/pipeline.py`, `app.py` — OCR, pipeline end-to-end, demo Gradio
- `tests/test_basic.py` — chạy `python tests/test_basic.py` (không cần dữ liệu), hiện 14/14 pass

## Môi trường (Windows)
- venv ở `.venv/` (Python 3.13); chạy bằng `.venv/Scripts/python.exe`. torch hiện là bản CUDA (2.14.1+cu130, cuda=True), gradio 6.30.
- GPU: MX570 A, chỉ 2 GB VRAM -> train mặc định (yolo11s, imgsz 1024, batch 8) nhiều khả năng
  không đủ; cân nhắc yolo11n / batch nhỏ / imgsz thấp, hoặc train trên Colab/Kaggle.
- Console Windows là cp1252: chạy script có `print` tiếng Việt phải đặt `PYTHONUTF8=1`.
- Manga109 để NGOÀI repo: `D:\MINI APP\dataset\Manga109_released_2026_05_21` (4.1 GB), dùng `--root` trỏ tới đó.
  Đã convert với `--copy` -> `data/yolo/` (train 8325 / val 1057 / test 1220 ảnh; train có
  79319 frame, 127783 text, 91321 face, 120841 body). `data/` đã trong `.gitignore`.
  LỆCH SỐ CHƯA LÀM RÕ: `data/yolo/labels/train` hiện có 93.698 dòng face (khớp XML 2026: 93.699), không phải 91.321 như ghi
  ở trên; chưa biết `weights/best.pt` (detector) được train trên bản convert nào. Val/test khớp XML (10.910 / 13.984 face).
- Dữ liệu re-ID (đều trong `data/`, gitignored): `characters.json`, `crops/`, `emb/` (embedding thô), `pred/` (box detector dự
  đoán `det_{val,test}.json` + embedding crop dự đoán + JSON kết quả), `reid/{f_b,b_a}/best.pt` (trọng số fine-tune, tải từ
  Kaggle; `runs_kaggle/` cũng gitignored).

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
- ĐÃ chạy thật (Kaggle + local): re-ID nhân vật (xem mục "Nhận dạng nhân vật") và `run_characters.py` trên chương thật (chỉ
  nhận xét sơ bộ "tạm ổn"; không có nhãn nên chưa đo).
- CHƯA làm: mask SAM, bảng so sánh model (yolo11n/m), đo end-to-end gộp thứ tự + OCR trên toàn bộ trang, speaker attribution dùng thị giác (đuôi bóng thoại).

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
| **qua panel (pipeline: `panel_tol=0.05` + dự phòng theo cạnh trên)** | **0.991** | **28/30 (93.3%)** |
| XY-cut thẳng trên text (không panel) | 0.940 | 18/30 (60%) |
| thẳng, tắt tách trang đôi | 0.606 | 1/30 |
| baseline (y rồi phải->trái) | 0.410 | 0/30 |
Lịch sử sửa (đều đo trên các nhãn này): (1) dung sai XY-cut 1% quá chặt vì viền panel thật chồng/lệch 30-60 px
-> `order_page(panel_tol=0.05)`: 20/30 -> 28/30; (2) khi XY-cut không cắt được (panel chồng lấn/borderless), dự phòng
cũ sắp theo TÂM y làm panel trái đứng trước panel phải cùng mép trên -> dự phòng mới: gom hàng theo CẠNH TRÊN
(dung sai tol) rồi phải->trái: thoại tau 0.986 -> 0.991, panel 15/16 -> 16/16, KHÔNG xấu đi ở đâu; với dự phòng
mới thì tol 0.01 hay 0.05 cho cùng kết quả ở thoại (panel_tol vẫn có lợi ở nhãn panel: 87.5% -> 100%).
LƯU Ý: tham số chọn trên chính các trang này (in-sample); leave-one-title-out cho kết quả tương tự; mẫu nhỏ.
Còn sai: WarewareHaOniDearu/090 (layout lệch hàng, nhãn đọc theo cột; tau ~0.7) -> layout nhập nhằng.
Với box DETECTOR dự đoán (yolo11s conf 0.25, `scripts/eval_order_pred.py`, cùng 30 trang; tau/đúng tính trên text
ghép được với GT, IoU>=0.5): panel recall 98.9% / precision 95.9% (279/291 dự đoán); text recall 98.1%
(411/419), FP 12 (2.8%).
| panel \| text | Tau mean | Đúng hoàn toàn |
|---|---|---|
| GT \| GT (tham chiếu) | 0.991 | 28/30 |
| dự đoán \| GT | 0.991 | 28/30 |
| GT \| dự đoán | 0.991 | 29/30 |
| **dự đoán \| dự đoán (pipeline thật)** | **0.991** | **29/30 (97%)** |
| dự đoán, thẳng (không panel) | 0.942 | 19/30 (63%) |
Truyện thật (chương số hoá 1280x1790, 66 trang, KHÔNG có nhãn): detector hay trả các "panel" chồng lên nhau rất lớn
(vd trang 3: x 469-1280 chồng x 477-1244 chồng x 1059-1280) nên XY-cut không có đường cắt sạch -> dự phòng. Với dự phòng
cũ có 12 cặp thoại "trái đọc trước phải cùng hàng" ở 8 trang; dự phòng mới còn 4 cặp ở 2 trang. Đây chỉ là chỉ số
gián tiếp (chưa có nhãn thật cho truyện này); cần người xem lại bằng mắt.
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

### Nhận dạng nhân vật (re-ID): gom mặt/thân theo nhân vật TRONG một bộ truyện
Nhãn: mỗi `<face>`/`<body>` trong XML có `character="<id>"` (phủ 100%: 118.593 face, 157.234 body). Id chỉ có nghĩa trong
từng bộ (0/3460 id trùng giữa các bộ) => chỉ đo được gom trong một bộ, không có danh tính xuyên bộ. `<text>` KHÔNG có
nhãn người nói. Nhóm "Other/Others/モブ…" (~9-10% face) là nhân vật nền gộp nhiều người: bị loại khỏi train và đo.
Mặt nhỏ (cạnh ngắn trung vị ~58 px). 98,9% mặt nằm trong một body (body nhỏ nhất bao nó) cùng nhân vật; 24,6% body không có mặt.
Test chỉ 11 bộ (360 nhân vật, 233 người có >=5 mặt): số đo dao động mạnh giữa các bộ (BCubed F1 face 0,42-0,81).

Mô hình: DINOv2-small (CLS + mean patch) + head 256 chiều, SupCon (32 nhân vật x 4 crop/batch), train 87 bộ train, chọn trên val.
Face `f_b` (mở cả 12 block, lr 1e-5, 10 epoch; thắng `f_a`/`f_c` theo val), body `b_a` (mở 4 block, lr 2e-5). Crop: face 126x126,
body 224x112, lề 10%. Đo (eval_cluster.py): agglomerative average-linkage trên cosine, cắt theo ngưỡng chọn trên val; trung bình
theo bộ; bỏ Other; chỉ nhân vật có >=2 box. Xáo nhãn ngẫu nhiên cho BCubed F1 ~0,20.
| Test, 11 bộ | Face | Body |
|---|---|---|
| DINOv2 thô, box GT: BCubed F1 / ARI / R@1 / số cụm dự đoán (thật) | 0,395 / 0,131 / 0,715 / 104 (26) | 0,336 / 0,030 / 0,467 / 75 (28) |
| **Fine-tune, box GT**: BCubed F1 / ARI / R@1 / mAP / cụm | **0,636** / 0,554 / 0,842 / 0,652 / 37 (26) | **0,549** / 0,430 / 0,766 / 0,542 / 39 (28) |
| Fine-tune, box DETECTOR, cách A (chỉ box ghép được GT) BCubed F1 / ARI | 0,658 / 0,573 | 0,565 / 0,493 |
| Fine-tune, box DETECTOR, **cách B (end-to-end)** BCubed F1 / pairwise F1 / ARI | **0,590** / 0,533 / 0,472 | **0,526** / 0,484 / 0,421 |
| Recall / precision detector (mặt/thân hợp lệ, IoU>=0,5) | 86,0% / ~88% | 89,8% / ~86% |
Cách A cô lập chất lượng embedding trên crop của detector (bỏ mặt bị sót nên mẫu dễ hơn, đừng đọc là "detector tốt hơn box GT").
Cách B tính box thừa (nhãn rác riêng lẻ) và mặt sót (cụm lẻ): hơi bi quan, "số cụm" của B bị thổi phồng bởi cụm lẻ.
Face fine-tune theo bộ (box detector, B): tốt nhất GinNoChimera 0,73, LoveHina 0,65, MiraiSan 0,72; tệ nhất WarewareHaOniDearu 0,44,
Hamlet 0,50, HeiseiJimen 0,47; Akuhamu 0,49 vì detector chỉ bắt 61% mặt (BCubed F1 0,62 với box GT). Vẫn CHIA THỪA cụm (~1,4x
số nhân vật thật ở box GT), riêng MiraiSan gộp thừa => ngưỡng cố định không tối ưu cho mọi bộ.
Trọng số lưu local: kết quả tái tạo được từ `data/reid/{f_b,b_a}/best.pt` (kiểm tra: face test BCubed F1 0,636 local vs 0,640 Kaggle).

Ghép face+body (`eval_fused.py`, `eval_fused_learned.py`; thực thể = một người trên một trang: mặt + body nhỏ nhất bao nó,
hoặc mặt/body đứng riêng; khoảng cách = trung bình có trọng số trên modality CẢ HAI cùng có, thiếu modality chung = 1,0).
Test, box detector, cách B, trung bình 11 bộ (cùng tập thực thể, so công bằng):
| Cấu hình | BCubed F1 | Pairwise F1 | ARI |
|---|---|---|---|
| Chỉ face | **0,593** | 0,464 | 0,386 |
| Chỉ body | 0,521 | 0,472 | 0,410 |
| Ghép 0,7/0,3 (đặt tay) | 0,573 | 0,535 | 0,471 |
| Ưu tiên mặt (D_face nếu cả hai có mặt, else D_body) | 0,543 | 0,481 | 0,426 |
| MLP học trên val (face+body) | 0,577 | **0,543** | **0,487** |
| MLP học + cùng-trang/khoảng cách trang | 0,588 | 0,532 | 0,480 |
KẾT LUẬN: ghép KHÔNG vượt chỉ-face theo BCubed F1 (chênh trung bình so với chỉ-face: -0,020 cho 0,7/0,3, -0,016 cho MLP,
-0,005 cho MLP+trang; sai số chuẩn ~0,018; tốt hơn ở 3/11, 3/11, 6/11 bộ) nhưng ARI/pairwise F1 cao hơn rõ; ghép giúp ở bộ detector bắt mặt kém (Akuhamu 0,50 -> 0,61). MLP học gần như bằng trọng số đặt tay
(AUC mức cặp 0,856 vs 0,848; thêm đặc trưng trang +0,002) => trọng số đặt tay đã gần tối ưu. AUC "chỉ face" 0,706 bị kéo thấp vì
~26% thực thể không có mặt (gán khoảng cách 1,0). BCubed khoan dung với cụm lẻ hơn pairwise/ARI nên hai thước đo bất đồng.
Khuyến nghị: gom theo MẶT làm chính; dùng ghép 0,7/0,3 nếu muốn bù trang detector không bắt mặt.

Chương thật: `run_characters.py <ảnh|thư mục|.cbz> [--mode face|fused] [--thr] [--sweep ...] [--draw]`. Ngưỡng mặc định (face 0,4;
fused 0,5) chọn trên val Manga109, KHÔNG kiểm chứng được trên truyện thật (không nhãn): dùng `--sweep`, xem `sheet_XX.jpg` để
phát hiện gộp nhầm/tách thừa. Mỗi thực thể trong characters.json có page, face_box, body_box, cluster (1-based, null = chưa gán).

### Speaker attribution (gán thoại cho người nói)
Nhãn: Manga109Dialog (ICME 2024, CC BY 4.0), 132.692 cặp (text_id, speaker_id) trên 109 bộ. Đã kiểm tra: dựa trên annotations
BẢN 2021.12.30 (147.887 text đúng bằng số text trong bài báo); `speaker_id` LUÔN là id <body> cùng trang (132.692/132.692). Id
face/body/frame giống hệt ở bản 2026 nhưng id <text> bị đánh số lại hoàn toàn (chỉ 43/147.887 trùng) -> ghép text 2021->2026 theo
IoU>=0,5 từng trang (khớp 1-1): 85,9% text khớp ở IoU 0,5 (94,7% ở 0,3). Trong cặp Dialog: train 14,5% / val 15,9% / test 11,6% bị loại vì
không ghép được text (nghĩa là mẫu đo lệch về text KHÔNG bị đổi box). 1.073 text có nhiều người nói (0,5-0,9%); lời dẫn truyện/tiêu đề/
người nói ngoài trang KHÔNG có nhãn -> "không có cặp" nghĩa là "không gán nhãn", không phải "không người nói". Phủ cả 11 bộ test (14.215 cặp).
Thực thể ứng viên = body (kèm mặt nằm trong) hoặc mặt đứng riêng; đúng ở mức BOX = chọn một body người nói của Dialog; mức NHÂN VẬT =
trùng id nhân vật (bỏ nhóm Other). Mô hình: MLP 2 lớp ẩn 128 trên 28 đặc trưng hình học (khoảng cách, cùng panel, hạng khoảng cách, kích thước,
vị trí tương đối, thứ tự đọc, mặt...), softmax trong top-20 ứng viên, train 87 bộ (89.118 thoại), chọn epoch trên val; không dùng ảnh.
| Đúng body / nhân vật (thoại có nhãn) | Val 11 bộ | TEST 11 bộ (chạm 1 lần) |
|---|---|---|
| Box GT: nearest toàn trang | 0,645 / 0,704 | (không chạy) |
| Box GT: nearest cùng panel | 0,699 / 0,726 | 0,792 / 0,818 |
| **Box GT: mô hình học** | **0,736** / 0,751 | **0,833** / 0,847 |
| Trần "người nói nằm cùng panel với text" | 0,825 | 0,896 |
| Box detector, nearest cùng panel, B (sót = sai) | 0,620 / 0,648 | 0,728 / 0,754 |
| **Box detector, mô hình học, B** | (không chạy) | **0,764** / 0,780 (A: 0,774 / 0,790) |
Detector bắt 98,7% thoại có nhãn (test). Test dễ hơn val (baseline 0,79 vs 0,70) nên đừng so tuyệt đối giữa hai cột; chênh mô hình-baseline:
+3,7 (val), +4,1 (test) điểm. Theo bộ (test, mô hình): 0,77 (ToutaMairimasu) -> 0,90 (Akuhamu); chỉ 11 bộ.
Lỗi còn lại (val/test): người nói NGOÀI panel của text 13,2%/6,8% (mô hình hầu như không bao giờ chọn ngoài panel); trong panel mà chọn nhầm
người khác 13,1%/9,9% (trung bình ~1,9 ứng viên cùng panel) -> cần tín hiệu thị giác (đuôi bóng thoại, hướng nhìn), hình học đã bão hoà.
Thử thị giác (`scripts/train_speaker_vis.py`, CNN nhỏ nhìn vùng 2,5x quanh box thoại dự đoán hướng tới người nói, train 2-fold theo bộ, thêm 4 đặc trưng
hướng vào MLP): hướng đúng trong 45° ở 76,1% thoại val (cos trung bình 0,745) nhưng val đúng body 0,739 so với 0,736 -> KHÔNG thêm thông tin ngoài
hình học (chưa chạy test, không đưa vào pipeline). Có thể CNN chỉ học lại ưu tiên hình học; thử đuôi bóng thoại thật cần mask bong bóng (SAM) hoặc nhãn hướng.
Thử giai đoạn 2 (thêm xác suất của câu thoại liền trước/sau theo thứ tự đọc): val 0,736 -> 0,740, nằm trong sai số -> không dùng.
Độ tin cậy (val, mô hình): p>=0,5 phủ 87,8% chính xác 0,781; p>=0,6 phủ 75,5% / 0,816; p>=0,7 phủ 63,1% / 0,844; p>=0,9 phủ 23,8% / 0,913.
`export_koma.py --speaker-min-prob` mặc định 0,6. Ghép với re-ID (thoại -> thực thể -> cụm `[NVk]`): thử trên 16 trang LoveHina_vol01 (test, 1 bộ):
gợi ý phủ 70,3% thoại có nhãn, đúng nhân vật 0,830 trên phần có gợi ý (ánh xạ cụm->nhân vật GT bằng đa số, hơi lạc quan; một nhân vật bị tách
thành 2 cụm). MẪU RẤT NHỎ, chưa đo trên cả 11 bộ. Chương thật (18+) chưa chạy gợi ý người nói.

### Chạy trên Kaggle
- Dữ liệu tải từ HF gated `hal-utokyo/Manga109` (cần token Read trong Secrets `HF_TOKEN`),
  convert ngay trên máy Kaggle; không upload dataset. `data.yaml` phải trỏ path Linux.
- File văn bản tải từ Kaggle có thể bị bọc markdown (```yaml, dấu `"` quanh từng dòng csv) và
  đổi đuôi (.csv->.txt); `.pt` có thể bị đổi thành .zip (chỉ cần đổi đuôi lại, đừng giải nén).
- Output notebook cũ gắn làm input ở `/kaggle/input/notebooks/<user>/<notebook>/`
  (chứa `yolo/` và `out/runs/detect/runs/<name>/weights/best.pt`).
- `kaggle_reid.ipynb`: Run All một lần (tải Manga109 ~4,1 GB, clone repo, tự chọn cấu hình face theo val, test một lần,
  rồi body); hàm `sh()` dừng notebook khi lệnh lỗi. Cell tải/giải nén chạy lại được. Notebook clone từ GitHub nên script
  phải đã PUSH trước. Crop cache vào `/kaggle/temp` nếu có, không thì `/kaggle/working` (giới hạn 20 GB; crop body train ~9 GB,
  notebook xoá cache face trước khi làm body). Output tải về: `.pt` và `.npz` bị đổi đuôi thành `.zip` -> đổi tên lại, đừng
  giải nén; đừng tải cả thư mục Output (có `crops/` nhiều GB). `transformers` Kaggle cũ đặt tên khoá attention khác bản local
  (`attention.attention.query` vs `attention.q_proj`): `train_reid.load_ckpt` đã đổi tên khoá.
- Smoke test trong notebook với ít bộ cần `P <= số nhân vật` và tổng bước >= 20 (đã sửa trong `train_reid.py`).

## Phát hiện từ dữ liệu thật
- Mỗi "trang" Manga109 là TRANG ĐÔI (ảnh ~1654x1170, hai trang liền nhau, đọc phải->trái).
- Thứ tự box trong XML KHÔNG phải thứ tự đọc (LoveHina p20: nhãn panel nhảy lung tung)
  -> không dùng làm GT reading order, phải tự gán nhãn ~50 trang.
- `reading_order()` có tham số `spread` (tự bật khi ảnh rộng hơn cao): tách trang đôi tại rãnh
  giữa, đọc hết trang trước rồi mới sang trang sau; nếu có panel tràn qua rãnh thì không tách.
  Khi gán nhãn/đo Kendall tau cần quyết định tính trên trang đôi hay từng trang đơn.
- Lớp: 0 frame, 1 text, 2 face, 3 body. `text` = bbox CHỮ, không phải đường viền bong bóng.
- `<face>`/`<body>` có `character` (id nhân vật theo bộ), `<text>` không có nhãn người nói: GT speaker phải lấy từ nguồn
  khác (Manga109Dialog, github.com/liyingxuan1012/Manga109Dialog, ~132.692 cặp thoại-người nói; giấy phép và việc khớp id với
  annotations 2026 CHƯA kiểm tra) hoặc tự gán nhãn.

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
7. (Một phần xong, xem mục "Speaker attribution"; còn: đuôi bóng thoại/thị giác, đo end-to-end với re-ID trên 11 bộ, tự gán ~30 trang kiểm tra nhãn Dialog) Speaker attribution: khảo sát nhãn (Manga109Dialog hay tự gán ~30-50 trang test), baseline "nhân vật gần nhất trong
   cùng panel", đo accuracy với box GT/dự đoán, rồi nối vào `export_koma.py` (gợi ý "nhân vật N" cho từng thoại).
8. (Mới) Re-ID: chưa đo trên truyện thật có nhãn; chưa thử ngưỡng thích nghi theo bộ (ngưỡng cố định chia thừa/gộp thừa tuỳ bộ);
   chưa làm rõ lệch số face train (91.321 vs 93.698).
