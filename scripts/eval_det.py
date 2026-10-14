"""mAP theo từng lớp trên tập test (chia theo bộ truyện).

    python scripts/eval_det.py --weights runs/yolo11s/weights/best.pt
"""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", default="data/yolo/data.yaml")
ap.add_argument("--split", default="test")
ap.add_argument("--imgsz", type=int, default=1024)
a = ap.parse_args()

m = YOLO(a.weights)
r = m.val(data=a.data, split=a.split, imgsz=a.imgsz)
print(f"\n{'class':8s} {'mAP50':>7s} {'mAP50-95':>9s}")
for i, name in m.names.items():
    print(f"{name:8s} {r.box.ap50[i]:7.3f} {r.box.maps[i]:9.3f}")
print(f"{'ALL':8s} {r.box.map50:7.3f} {r.box.map:9.3f}")
