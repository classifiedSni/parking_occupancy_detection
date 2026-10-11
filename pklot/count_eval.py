"""How accurate are the free / occupied COUNTS, not just the boxes?

    python -m pklot.count_eval --weights runs/yolo_parking_occupancy/weights/best.pt --split test

mAP says how well individual spaces are found. What a parking sign needs is the number of
free spaces in a frame, so this compares the predicted counts with the labelled counts for
every image of a split and reports the error per parking lot.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .common import DATA_DIR, pick_device
from .occupancy import count_spaces

CHUNK = 16  # images per predict() call


def true_counts(label_file: Path):
    """(free, occupied) from a YOLO label file (class 0 = empty, 1 = occupied)."""
    classes = [int(line.split()[0]) for line in label_file.read_text().splitlines() if line.strip()]
    return classes.count(0), classes.count(1)


def summarize(rows):
    """rows: [(lot, true_free, true_occ, pred_free, pred_occ)] -> error statistics."""
    def stats(subset):
        free_err = [abs(r[3] - r[1]) for r in subset]
        occ_err = [abs(r[4] - r[2]) for r in subset]
        return {
            "images": len(subset),
            "mean_abs_error_free": round(mean(free_err), 3),
            "mean_abs_error_occupied": round(mean(occ_err), 3),
            "exact_free_count_pct": round(100 * sum(e == 0 for e in free_err) / len(subset), 1),
            "free_within_2_pct": round(100 * sum(e <= 2 for e in free_err) / len(subset), 1),
            "mean_true_spaces_per_image": round(mean(r[1] + r[2] for r in subset), 1),
        }
    by_lot = defaultdict(list)
    for row in rows:
        by_lot[row[0]].append(row)
    return {"overall": stats(rows), **{lot: stats(r) for lot, r in sorted(by_lot.items())}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--data-dir", default=str(DATA_DIR / "pklot"), help="folder written by pklot.prepare_data")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=352)
    p.add_argument("--device", default=None)
    p.add_argument("--out", help="write the statistics to this json file")
    args = p.parse_args(argv)
    from ultralytics import YOLO

    images = sorted((Path(args.data_dir) / args.split / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images in {args.data_dir}/{args.split}/images - run pklot.prepare_data first.")
    model = YOLO(args.weights)
    device = args.device if args.device is not None else pick_device()

    rows = []
    # Small explicit chunks. Handing Ultralytics one long list of paths makes it one giant
    # batch, whose NMS step hits a time limit and silently drops detections.
    for start in range(0, len(images), CHUNK):
        chunk = images[start:start + CHUNK]
        for result in model.predict(source=[str(i) for i in chunk], conf=args.conf, imgsz=args.imgsz,
                                    device=device, verbose=False):
            image = Path(result.path)
            predicted = count_spaces(result.names[int(c)] for c in result.boxes.cls)
            free, occupied = true_counts(image.parents[1] / "labels" / f"{image.stem}.txt")
            rows.append((image.name.split("_")[0], free, occupied, predicted.free, predicted.occupied))
        if (start // CHUNK) % 10 == 9:
            print(f"  {len(rows)}/{len(images)}", flush=True)

    report = summarize(rows)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps({"split": args.split, "conf": args.conf, **report}, indent=2))


if __name__ == "__main__":
    main()
