"""Count free and occupied parking spaces in images with a trained YOLOv8 model.

    python -m pklot.predict --weights runs/yolo_parking_occupancy/weights/best.pt --source lot.jpg
    python -m pklot.predict --weights best.pt --source photos/ --out predictions/

For each image it writes an annotated copy to --out and prints the free/occupied
counts; a summary.csv with one row per image is written alongside.
"""
import argparse
import csv
from pathlib import Path

from .common import PROJECT_ROOT, pick_device
from .occupancy import classify, count_spaces

COLORS = {"free": (60, 190, 60), "occupied": (50, 50, 230)}  # BGR


def annotate(result, occupancy):
    """Frame with a green box per free space, a red box per occupied one, and a count banner.

    Ultralytics' own result.plot() writes a label on every box, which is unreadable in a
    100-space lot.
    """
    import cv2

    image = result.orig_img.copy()
    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        kind = classify(result.names[int(cls)])
        if kind:
            x0, y0, x1, y1 = (int(v) for v in box)
            cv2.rectangle(image, (x0, y0), (x1, y1), COLORS[kind], 2)
    cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(image, f"{occupancy.free} free / {occupancy.occupied} occupied", (12, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="trained YOLOv8 weights (best.pt or best.onnx)")
    p.add_argument("--source", required=True, help="image file, folder of images, or video")
    p.add_argument("--out", default=str(PROJECT_ROOT / "predictions"))
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=352, help="inference size (training used 352)")
    p.add_argument("--device", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import cv2
    from ultralytics import YOLO

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    device = args.device if args.device is not None else pick_device()

    rows = []
    for result in model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz,
                                device=device, stream=True, verbose=False):
        name = Path(result.path).name
        occupancy = count_spaces(result.names[int(c)] for c in result.boxes.cls)
        cv2.imwrite(str(out_dir / name), annotate(result, occupancy))
        print(f"{name}: {occupancy}")
        rows.append({"image": name, "free": occupancy.free, "occupied": occupancy.occupied,
                     "total": occupancy.total, "occupancy_rate": round(occupancy.rate, 4)})

    if rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"{len(rows)} image(s) processed -> {out_dir}")


if __name__ == "__main__":
    main()
