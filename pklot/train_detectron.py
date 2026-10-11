"""Train and evaluate Faster R-CNN or RetinaNet on PKLot with Detectron2.

    python -m pklot.train_detectron --model faster_rcnn
    python -m pklot.train_detectron --model retinanet
    python -m pklot.train_detectron --model retinanet --eval-only --weights output/retinanet/model_final.pth
    python -m pklot.train_detectron --model faster_rcnn --predict lot.jpg --weights output/faster_rcnn/model_final.pth

Needs a CUDA GPU and Detectron2 (see README) - run it on Colab, not on a Mac.

Fix vs. the original notebooks: they set NUM_CLASSES = 2, but their dataset export had
three categories (an empty "spaces" parent plus space-empty and space-occupied), so
"space-occupied" fell outside the output layer and scored AP 0.000. The class count is
now read from the dataset itself.
"""
import argparse
import json
import math
import shutil
from pathlib import Path

from .common import PROJECT_ROOT
from .prepare_data import get_dataset
from .occupancy import count_spaces

ZOO = {
    "faster_rcnn": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "retinanet": "COCO-Detection/retinanet_R_50_FPN_3x.yaml",
}
# Category order written by pklot.prepare_data (COCO ids 1 and 2). Training reads the real list
# from the dataset; this constant is only used for --predict so inference needs no dataset on disk.
PKLOT_CLASSES = ("space-empty", "space-occupied")
# Settings of the runs whose results are in docs/RESULTS.md
DEFAULTS = {
    "faster_rcnn": {"max_iter": 10000, "ims_per_batch": 5},
    "retinanet": {"max_iter": 5000, "ims_per_batch": 20},
}


def register_datasets(coco_dir: Path):
    """Register the splits with Detectron2; returns the class names it derived."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    for name, split in (("pklot_train", "train"), ("pklot_test", "test")):
        if name not in DatasetCatalog.list():
            register_coco_instances(name, {}, str(coco_dir / split / "_annotations.coco.json"),
                                    str(coco_dir / split / "images"))
    DatasetCatalog.get("pklot_train")  # loading it is what fills in thing_classes
    return list(MetadataCatalog.get("pklot_train").thing_classes)


def build_cfg(args, num_classes: int):
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    zoo_file = ZOO[args.model]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(zoo_file))
    cfg.MODEL.WEIGHTS = args.weights or model_zoo.get_checkpoint_url(zoo_file)
    cfg.DATASETS.TRAIN = ("pklot_train",)
    cfg.DATASETS.TEST = ("pklot_test",)
    cfg.DATALOADER.NUM_WORKERS = 2

    defaults = DEFAULTS[args.model]
    cfg.SOLVER.MAX_ITER = args.max_iter or defaults["max_iter"]
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch or defaults["ims_per_batch"]
    cfg.SOLVER.BASE_LR = args.lr
    # Learning-rate decay. The original set cfg.INPUT.GAMMA, which Detectron2 ignores,
    # and left STEPS beyond MAX_ITER, so the rate never actually decayed.
    cfg.SOLVER.GAMMA = args.gamma
    cfg.SOLVER.STEPS = (int(0.75 * cfg.SOLVER.MAX_ITER),)

    if args.model == "faster_rcnn":
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.score_thresh

    cfg.OUTPUT_DIR = str(PROJECT_ROOT / "output" / args.model)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.MODEL.DEVICE == "cpu":
        print("WARNING: no CUDA GPU found - Detectron2 training on CPU is impractically slow.")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return cfg


def _json_safe(value):
    """COCO eval reports NaN for classes with no instances; turn that into null."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def evaluate(cfg, model, out_dir: Path):
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    evaluator = COCOEvaluator("pklot_test", tasks=("bbox",), distributed=False,
                              output_dir=str(out_dir / "eval"))
    loader = build_detection_test_loader(cfg, "pklot_test")
    results = inference_on_dataset(model, loader, evaluator)
    (out_dir / "metrics.json").write_text(json.dumps(_json_safe(results), indent=2))
    return results


def predict(cfg, class_names, source: Path, out_dir: Path):
    """Draw detections for one image or a folder and print free/occupied counts."""
    import cv2
    from detectron2.data import MetadataCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer

    predictor = DefaultPredictor(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in ([source] if source.is_file() else source.iterdir())
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Unable to read {path}")
            continue
        instances = predictor(image)["instances"].to("cpu")
        names = [class_names[i] for i in instances.pred_classes.tolist()]
        print(f"{path.name}: {count_spaces(names)}")
        drawn = Visualizer(image[:, :, ::-1], MetadataCatalog.get("pklot_train"), scale=1.2)
        cv2.imwrite(str(out_dir / path.name), drawn.draw_instance_predictions(instances).get_image()[:, :, ::-1])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=ZOO, required=True)
    p.add_argument("--max-iter", type=int, help="default: 10000 faster_rcnn, 5000 retinanet")
    p.add_argument("--ims-per-batch", type=int, help="default: 5 faster_rcnn, 20 retinanet")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--gamma", type=float, default=0.05, help="learning-rate decay factor")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--data-dir", help="prepared dataset folder (default: data/pklot, built by pklot.prepare_data)")
    p.add_argument("--weights", help="checkpoint to load (for --eval-only / --predict)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--predict", help="image or folder to run detection on (needs --weights)")
    p.add_argument("--save-dir", help="copy the output folder here (e.g. a Google Drive folder)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.eval_only or args.predict) and not args.weights:
        raise SystemExit("--eval-only and --predict need --weights path/to/model_final.pth")

    if args.predict:
        # Inference only needs the class names, not the multi-GB dataset.
        from detectron2.data import MetadataCatalog
        MetadataCatalog.get("pklot_train").thing_classes = list(PKLOT_CLASSES)
        cfg = build_cfg(args, num_classes=len(PKLOT_CLASSES))
        predict(cfg, PKLOT_CLASSES, Path(args.predict), PROJECT_ROOT / "predictions" / args.model)
        return

    class_names = register_datasets(get_dataset(args.data_dir))
    print(f"Classes ({len(class_names)}): {class_names}")
    cfg = build_cfg(args, num_classes=len(class_names))
    out_dir = Path(cfg.OUTPUT_DIR)

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.engine import DefaultTrainer

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    else:
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        model = trainer.model

    print(evaluate(cfg, model, out_dir))

    if args.save_dir and not args.eval_only:
        destination = Path(args.save_dir) / args.model
        shutil.copytree(out_dir, destination, dirs_exist_ok=True)
        print(f"Output copied to {destination}")
    print(f"Weights: {out_dir / 'model_final.pth'}")


if __name__ == "__main__":
    main()
