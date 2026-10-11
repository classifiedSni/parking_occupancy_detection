"""Train and validate YOLOv8 on PKLot (the model the report selected).

    python -m pklot.train_yolo                          # recipe from the report: yolov8x, 50 epochs
    python -m pklot.train_yolo --model yolov8s.pt --epochs 30 --batch 16 --cache ram   # laptop-sized

The first run downloads and prepares the PKLot dataset (see pklot/prepare_data.py).
    python -m pklot.train_yolo --save-dir "/content/drive/MyDrive/Pklot/weights"

--save-dir copies the finished run (best.pt, best.onnx, metrics) somewhere that
survives the session. The original notebook skipped this and the weights were lost.
"""
import argparse
import json
import shutil
from pathlib import Path

import yaml

from .common import PROJECT_ROOT, pick_device
from .prepare_data import get_dataset


def write_data_yaml(dataset_dir: Path) -> Path:
    """Write a copy of data.yaml that points at this machine's absolute dataset path."""
    dataset_dir = Path(dataset_dir).resolve()
    config = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    config["path"] = str(dataset_dir)
    config["train"] = "train/images"
    config["val"] = "valid/images"
    if (dataset_dir / "test" / "images").is_dir():
        config["test"] = "test/images"
    else:
        config.pop("test", None)
    out = dataset_dir / "data.resolved.yaml"
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolov8x.pt", help="pretrained checkpoint or .yaml (default: yolov8x.pt)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=352)
    p.add_argument("--name", default="yolo_parking_occupancy")
    p.add_argument("--device", default=None, help="0, mps or cpu (default: auto-detect)")
    p.add_argument("--data-dir", help="prepared dataset folder (default: data/pklot, built by pklot.prepare_data)")
    p.add_argument("--cache", choices=["ram", "disk"], help="cache resized images (much faster epochs; needs RAM)")
    p.add_argument("--save-dir", help="copy the finished run here (e.g. a Google Drive folder)")
    p.add_argument("--no-onnx", action="store_true", help="skip the ONNX export")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from ultralytics import YOLO  # heavy import, keep it out of --help

    data_yaml = write_data_yaml(get_dataset(args.data_dir))
    device = args.device if args.device is not None else pick_device()
    print(f"Training {args.model} on device={device}")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        project=str(PROJECT_ROOT / "runs"),
        name=args.name,
        exist_ok=False,
        cache=args.cache or False,
    )
    run_dir = Path(model.trainer.save_dir)

    # best.pt on the validation days, then on the held-out test days (never seen during training)
    summary = {}
    for split in ("val", "test"):
        if split == "test" and "test" not in yaml.safe_load(data_yaml.read_text()):
            continue
        metrics = model.val(split=split, project=str(PROJECT_ROOT / "runs"), name=f"{args.name}_{split}", exist_ok=True)
        summary[split] = {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "per_class_mAP50-95": {metrics.names[i]: float(metrics.box.maps[i]) for i in metrics.box.ap_class_index},
        }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_onnx:
        model.export(format="onnx")

    if args.save_dir:
        destination = Path(args.save_dir) / run_dir.name
        shutil.copytree(run_dir, destination, dirs_exist_ok=True)
        print(f"Run copied to {destination}")
    print(f"Weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
