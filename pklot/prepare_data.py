"""Download the official PKLot dataset and convert it to YOLO and COCO format.

    python -m pklot.prepare_data                     # download (4.9 GB), convert
    python -m pklot.prepare_data --tar PKLot.tar.gz  # use a tarball you already have

Source: Almeida et al., "PKLot - A robust dataset for parking lot classification",
Expert Systems with Applications, 2015 (UFPR / PUCPR, CC BY 4.0).
https://web.inf.ufpr.br/vri/databases/parking-lot-database/  -  no account or key needed.

Output (default data/pklot):
    {train,valid,test}/images/*.jpg          full 1280x720 frames
    {train,valid,test}/labels/*.txt          YOLO labels   (0 = space-empty, 1 = space-occupied)
    {train,valid,test}/_annotations.coco.json  the same boxes as COCO (category ids 1 and 2)
    data.yaml, splits.json

The split is by capture DATE within each parking lot, not by random frame. PKLot is
time-lapse footage from three fixed cameras (a frame every 5 minutes), so a random
split puts near-identical frames in train and validation and inflates the score.
"""
import argparse
import json
import random
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from math import cos, radians, sin
from pathlib import Path

import requests
import yaml
from PIL import Image

from .common import DATA_DIR

URL = "https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz"
CLASSES = ("space-empty", "space-occupied")  # YOLO class ids 0 and 1; COCO ids 1 and 2
SPLITS = ("train", "valid", "test")
FRACTIONS = (0.7, 0.2, 0.1)
MEMBER = re.compile(r"^PKLot/PKLot/(?P<lot>[A-Za-z0-9]+)/(?P<weather>[A-Za-z]+)/"
                    r"(?P<date>[0-9-]+)/(?P<name>[^/]+)\.(?P<ext>jpg|xml)$")


def download(url: str, dest: Path) -> Path:
    """Download with resume; skip if the file is already complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = int(requests.head(url, allow_redirects=True, timeout=30).headers["Content-Length"])
    have = dest.stat().st_size if dest.exists() else 0
    if have == total:
        print(f"{dest} already downloaded ({total / 1e9:.2f} GB).")
        return dest
    if have > total:
        dest.unlink()
        have = 0
    print(f"Downloading {total / 1e9:.2f} GB to {dest} (resuming at {have / 1e9:.2f} GB) ...")
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        response.raise_for_status()
        resumed = have and response.status_code == 206
        done, next_report = (have if resumed else 0), 0
        with open(dest, "ab" if resumed else "wb") as out:
            for chunk in response.iter_content(chunk_size=1 << 20):
                out.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    print(f"  {done / total:.0%}", flush=True)
                    next_report = done + total // 10
    if dest.stat().st_size != total:
        raise SystemExit("Download is incomplete - run the command again to resume.")
    return dest


def extract_frames(tar_path: Path, staging: Path) -> int:
    """Pull the full frames and their XML out of the tarball (skips the cropped-space copy)."""
    count = 0
    with tarfile.open(tar_path, "r|gz") as archive:  # stream: one pass through the 4.9 GB
        for member in archive:
            match = MEMBER.match(member.name)
            if not (member.isfile() and match):
                continue
            target = staging / match["lot"] / match["weather"] / match["date"] / f"{match['name']}.{match['ext']}"
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            count += match["ext"] == "jpg"
            if count and count % 2000 == 0 and match["ext"] == "jpg":
                print(f"  extracted {count} frames", flush=True)
    return count


def _box_from_space(space):
    """Axis-aligned (x0, y0, x1, y1) in pixels for one <space>, from its contour or rotated rect."""
    points = [(float(p.get("x")), float(p.get("y"))) for p in space.findall("./contour/point")]
    if not points:
        rect = space.find("rotatedRect")
        if rect is None:
            return None
        cx, cy = (float(rect.find("center").get(k)) for k in ("x", "y"))
        w, h = (float(rect.find("size").get(k)) for k in ("w", "h"))
        angle = radians(float(rect.find("angle").get("d")))
        points = [(cx + dx * cos(angle) - dy * sin(angle), cy + dx * sin(angle) + dy * cos(angle))
                  for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def read_spaces(xml_path: Path, width: int, height: int):
    """[(class_id, x0, y0, x1, y1)] for every space with a known occupancy, clipped to the frame."""
    boxes = []
    for space in ET.parse(xml_path).getroot().iter("space"):
        occupied = space.get("occupied")
        box = _box_from_space(space)
        if occupied not in ("0", "1") or box is None:
            continue  # occupancy unknown
        x0, y0 = max(0.0, box[0]), max(0.0, box[1])
        x1, y1 = min(float(width), box[2]), min(float(height), box[3])
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            boxes.append((int(occupied), x0, y0, x1, y1))
    return boxes


def assign_splits(dates_by_lot, seed=0):
    """{(lot, date): split}. Whole days go to one split; every lot appears in every split."""
    rng = random.Random(seed)
    assignment = {}
    for lot, dates in sorted(dates_by_lot.items()):
        dates = sorted(dates)
        rng.shuffle(dates)
        total = sum(n for _, n in dates)
        running, edges = 0, [total * FRACTIONS[0], total * (FRACTIONS[0] + FRACTIONS[1])]
        for date, n in dates:
            split = SPLITS[0] if running < edges[0] else SPLITS[1] if running < edges[1] else SPLITS[2]
            assignment[(lot, date)] = split
            running += n
    return assignment


def convert(staging: Path, out_dir: Path) -> dict:
    """Move frames into split folders and write YOLO labels + COCO json. Returns summary stats."""
    frames = defaultdict(list)  # (lot, date) -> [jpg paths]
    for jpg in sorted(staging.glob("*/*/*/*.jpg")):
        frames[(jpg.parents[2].name, jpg.parent.name)].append(jpg)
    if not frames:
        raise SystemExit(f"No frames found under {staging}.")

    dates_by_lot = defaultdict(list)
    for (lot, date), files in frames.items():
        dates_by_lot[lot].append((date, len(files)))
    assignment = assign_splits(dates_by_lot)

    coco = {s: {"images": [], "annotations": [], "categories":
                [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]} for s in SPLITS}
    stats = {s: {"images": 0, "boxes": [0, 0]} for s in SPLITS}
    for split in SPLITS:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for (lot, date), files in sorted(frames.items()):
        split = assignment[(lot, date)]
        for jpg in files:
            xml = jpg.with_suffix(".xml")
            if not xml.exists():
                continue
            with Image.open(jpg) as image:
                width, height = image.size
            boxes = read_spaces(xml, width, height)
            name = f"{lot}_{date}_{jpg.stem}.jpg"
            shutil.move(str(jpg), out_dir / split / "images" / name)
            lines, image_id = [], len(coco[split]["images"]) + 1
            coco[split]["images"].append({"id": image_id, "file_name": name, "width": width, "height": height})
            for cls, x0, y0, x1, y1 in boxes:
                lines.append(f"{cls} {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} "
                             f"{(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}")
                coco[split]["annotations"].append({
                    "id": len(coco[split]["annotations"]) + 1, "image_id": image_id, "category_id": cls + 1,
                    "bbox": [round(x0, 2), round(y0, 2), round(x1 - x0, 2), round(y1 - y0, 2)],
                    "area": round((x1 - x0) * (y1 - y0), 2), "iscrowd": 0})
                stats[split]["boxes"][cls] += 1
            (out_dir / split / "labels" / name.replace(".jpg", ".txt")).write_text("\n".join(lines))
            stats[split]["images"] += 1

    for split in SPLITS:
        (out_dir / split / "_annotations.coco.json").write_text(json.dumps(coco[split]))
    (out_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": len(CLASSES), "names": list(CLASSES)}, sort_keys=False))
    (out_dir / "splits.json").write_text(json.dumps(
        {s: sorted(f"{lot}/{date}" for (lot, date), v in assignment.items() if v == s) for s in SPLITS}, indent=1))
    return stats


def prepare(out_dir: Path = None, tar_path: Path = None) -> Path:
    out_dir = Path(out_dir or DATA_DIR / "pklot")
    if (out_dir / "data.yaml").exists():
        print(f"Dataset already prepared at {out_dir}.")
        return out_dir
    tar_path = Path(tar_path) if tar_path else download(URL, DATA_DIR / "PKLot.tar.gz")
    staging = out_dir.parent / "_pklot_staging"
    if staging.exists():
        shutil.rmtree(staging)
    print("Extracting frames ...")
    print(f"{extract_frames(tar_path, staging)} frames extracted. Converting ...")
    stats = convert(staging, out_dir)
    shutil.rmtree(staging, ignore_errors=True)
    for split, s in stats.items():
        print(f"  {split:5s} {s['images']:6d} images   {s['boxes'][0]:7d} empty   {s['boxes'][1]:7d} occupied")
    return out_dir


def get_dataset(data_dir=None) -> Path:
    """Folder to train on: --data-dir if given, otherwise download + convert on first use."""
    return Path(data_dir) if data_dir else prepare()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", help="output folder (default: data/pklot)")
    p.add_argument("--tar", help="path to an already-downloaded PKLot.tar.gz")
    args = p.parse_args(argv)
    print(f"Ready: {prepare(args.out, args.tar)}")


if __name__ == "__main__":
    main()
