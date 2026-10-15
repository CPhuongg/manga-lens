"""Fine-tune detector 4 lớp (frame/text/face/body).

    python scripts/train.py --model yolo11s.pt --epochs 50 --batch 8
Đổi --model sang yolo11n.pt / yolo11m.pt hoặc rtdetr-l.pt để làm bảng so sánh.
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/yolo/data.yaml")
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--name", default=None)
ap.add_argument("--device", default=None)
ap.add_argument("--time", type=float, default=None,
                help="giới hạn giờ huấn luyện; Ultralytics tự chỉnh số epoch + lịch lr để KẾT THÚC đúng hạn "
                     "(ghi đè --epochs). Dùng cho model lớn để khỏi vượt 12 giờ/phiên Kaggle")
ap.add_argument("--fraction", type=float, default=1.0, help="dùng một phần dữ liệu train (chạy thử nhanh)")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

YOLO(a.model).train(
    data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
    device=a.device, project="runs", name=a.name or a.model.split(".")[0],
    time=a.time, fraction=a.fraction, workers=a.workers,
    # manga là ảnh đen trắng: bỏ augment màu, giữ hình học vừa phải
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, fliplr=0.0,  # KHÔNG lật ngang: đổi hướng đọc & bố cục
    mosaic=0.5, scale=0.4, degrees=0.0,
)
