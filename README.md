# manga-lens

Phân tích một trang manga: phát hiện **panel / text / mặt / thân**, sắp **thứ tự đọc** (phải sang trái,
có xử lý trang đôi) và **OCR thoại** tiếng Nhật. Dự án cá nhân để luyện thị giác máy tính.
Dữ liệu: [Manga109](http://www.manga109.org/) (chỉ dùng cho học thuật). Phạm vi "Mid": chưa gán lời thoại
cho từng nhân vật.

```
ảnh trang -> YOLO11s (frame/text/face/body) -> XY-cut (panel, rồi thoại trong panel) -> manga-ocr -> JSON + ảnh vẽ
```

## Kết quả

Tất cả số liệu đo trên các bộ truyện **tập test** (chia theo bộ truyện, không chia theo trang, để tránh rò
rỉ nhân vật/nét vẽ). Huấn luyện: `yolo11s`, 50 epoch, imgsz 1024, batch 16, một GPU T4 (Kaggle).

### 1. Detector (1220 ảnh test)

| Lớp | mAP50 | mAP50-95 |
|---|---|---|
| frame (panel) | 0.993 | 0.970 |
| text (chữ) | 0.990 | 0.889 |
| face | 0.896 | 0.457 |
| body | 0.920 | 0.690 |
| **Tất cả** | **0.949** | **0.752** |

Precision 0.930, recall 0.904. Suy luận trên T4, imgsz 1024: 15.8 ms (mạng) + 1.9 (tiền xử lý) + 3.7
(hậu xử lý) mỗi ảnh. mAP50-95 test (0.752) khớp val (0.746), nên không có dấu hiệu overfit.
`face` và `body` thấp ở mAP50-95: model tìm được nhưng box không khít (nhãn không nhất quán, mặt nhỏ,
thân bị che).

| Ma trận nhầm lẫn (tập val) | Đường precision-recall (tập val) |
|---|---|
| ![](docs/confusion_matrix_val.png) | ![](docs/pr_curve_val.png) |

*Hai biểu đồ đo trên tập **val** ở ngưỡng conf rất thấp (0.001), nên cột "background" bị thổi lên và
không phản ánh hoạt động ở conf 0.25.*

### 2. OCR (manga-ocr, 200 trang test ngẫu nhiên, 2987 text box)

| Nhóm | CER theo box | Đúng hoàn toàn |
|---|---|---|
| Box GT (giới hạn trên của OCR) | 0.034 (0.039 theo ký tự) | 84.3% |
| Box dự đoán (ghép được với GT, IoU ≥ 0.5) | 0.027 | 84.4% |
| **End-to-end** (GT bị detector sót tính CER 1.0) | **0.055** | 82.1% |

Detector bắt được 97.2% text (conf 0.25), 125 box thừa trên 3028 box dự đoán.
`manga-ocr` luôn xuất ký tự toàn chiều rộng (`．`, `！`) còn nhãn dùng `…`, `・`: nếu so từng ký tự, CER
là 0.156, nhưng phần lớn là khác cách viết dấu câu. Mặc định script chuẩn hóa dấu câu
(`--strict` để so từng ký tự).

### 3. Thứ tự đọc (30 trang tự gán nhãn, đọc trang phải trước)

| Panel \| text | Kendall tau | Trang đúng hoàn toàn |
|---|---|---|
| GT \| GT | 0.986 | 28/30 |
| **dự đoán \| dự đoán (pipeline thật)** | **0.987** | **29/30** |
| XY-cut thẳng trên text, không dùng panel | 0.942 | 19/30 |
| baseline: trên xuống, phải sang trái | 0.410 | 0/30 |

Hai điểm mấu chốt: (1) **tách trang đôi** là bắt buộc (tắt đi, tau còn 0.606); (2) viền panel thật hay
chồng/lệch nhau 30-60 px nên dung sai XY-cut mặc định 1% không cắt được. Dùng dung sai riêng
`panel_tol = 0.05` cho panel đưa tỉ lệ trang đúng từ 20/30 lên 28/30 (box GT).

### 4. Chuỗi cả pipeline (ước lượng)
detector tìm ~98% text → thứ tự thoại đúng ở 29/30 trang → OCR sai ~3-4% ký tự trên box đúng.

## Phân tích lỗi

- **Detector bỏ sót text (84/2987 box):** 52% chỉ có conf thấp (0.01-0.25), 23% bị gộp (Manga109 chia box theo
  cột, detector gộp nhiều cột thành một), 15% hoàn toàn không thấy (0.4% tổng). Hạ ngưỡng conf (0.15) thu về
  13 box nhưng thêm 54 box thừa; nhận thêm box conf thấp ở "chỗ trống" cho 27 box đúng và 129 box thừa. Cả hai
  đều không đáng, nên giữ conf 0.25.
- **OCR:** lỗi thật tập trung ở (i) chú thích dài trên box rất hẹp và dài (CER 0.8-0.9), (ii) ảo giác chữ trên box
  chỉ có dấu chấm, (iii) chữ viết tay / cách điệu / SFX, (iv) nhiều dòng và chữ lặp, (v) vài nhãn GT nghi nhiễu.
- **Thứ tự đọc:** trang sai duy nhất là layout lệch hàng mà người gán nhãn đọc theo cột (nhập nhằng).

## Giới hạn

- Mẫu nhỏ: OCR 200 trang, thứ tự đọc 30 trang. Các số liệu thứ tự đọc được chọn tham số (`panel_tol`) trên chính
  các trang này nên hơi lạc quan (leave-one-title-out: 28/30, không có dấu hiệu overfit rõ).
- Thứ tự đọc chỉ tính trên text ghép được với GT; text bị sót và box thừa không nằm trong tau.
- Chỉ thử một model (`yolo11s`); chưa có bảng so sánh `yolo11n/m`, `rtdetr-l`. Chưa làm mask bong bóng (SAM).
- Mới chạy demo trên máy cá nhân; chưa đưa lên Hugging Face Spaces.

## Chạy lại

```bash
pip install -r requirements.txt
python tests/test_basic.py                       # kiểm tra nhanh, không cần dữ liệu

# Dữ liệu: xin Manga109 tại manga109.org (học thuật), giải nén, rồi:
python scripts/convert_manga109.py --root /path/Manga109 --out data/yolo --copy

# Train: notebook kaggle_train.ipynb (tải Manga109 từ Hugging Face rồi train trên Kaggle), hoặc local:
python scripts/train.py --model yolo11s.pt --epochs 50 --imgsz 1024 --batch 16

# Đánh giá
python scripts/eval_det.py --weights weights/best.pt
python scripts/eval_ocr.py --root /path/Manga109 --titles data/yolo/test_titles.txt --weights weights/best.pt
python scripts/eval_order_pred.py --root /path/Manga109 --weights weights/best.pt   # cần nhãn thứ tự đọc

# Demo Gradio
python app.py --weights weights/best.pt
```

Tự gán nhãn thứ tự đọc: `python scripts/label_order.py --root /path/Manga109` (bấm box thoại theo thứ tự đọc),
rồi `python scripts/eval_order.py --mode text`.

## Cấu trúc

| Đường dẫn | Việc |
|---|---|
| `src/pipeline.py`, `app.py` | pipeline end-to-end, demo Gradio |
| `src/reading_order.py` | XY-cut đệ quy phải→trái, tách trang đôi, gán text vào panel, Kendall tau |
| `src/ocr.py` | manga-ocr, CER, ghép box, chẩn đoán box bị sót |
| `scripts/` | convert, train, các script đánh giá, công cụ gán nhãn thứ tự đọc |
| `kaggle_train.ipynb` | train trên Kaggle |

## Lưu ý

- Luôn chia train/val/test **theo bộ truyện**. Không bật lật ngang (`fliplr`) khi train: lật làm sai hướng đọc.
- Repo chỉ chứa code, tài liệu và biểu đồ số liệu. Không commit ảnh/dataset/trọng số, không đăng trang truyện
  có bản quyền (README, demo) và tuân thủ điều khoản của Manga109.
