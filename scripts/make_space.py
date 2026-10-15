"""Gom thư mục triển khai Hugging Face Space vào build/space/ (không commit, đã trong .gitignore).

    python scripts/make_space.py --weights weights/best.pt
    hf upload <user>/manga-lens build/space . --repo-type space     # cần `hf auth login`
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "best.pt")
ap.add_argument("--out", type=Path, default=ROOT / "build" / "space")
a = ap.parse_args()

if not a.weights.exists():
    raise SystemExit(f"Không thấy weights: {a.weights}")
if a.out.exists():
    shutil.rmtree(a.out)
(a.out / "weights").mkdir(parents=True)
shutil.copy2(ROOT / "app.py", a.out / "app.py")
shutil.copy2(ROOT / "space" / "README.md", a.out / "README.md")
shutil.copy2(ROOT / "space" / "requirements.txt", a.out / "requirements.txt")
shutil.copytree(ROOT / "src", a.out / "src", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2(a.weights, a.out / "weights" / "best.pt")
for p in sorted(a.out.rglob("*")):
    if p.is_file():
        print(f"{p.relative_to(a.out)}  ({p.stat().st_size / 1e6:.2f} MB)")
print("-> sẵn sàng:", a.out)
