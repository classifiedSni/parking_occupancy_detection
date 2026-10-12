# Image-Based Parking Occupancy Detection (PKLot)

Looks at a photo of a parking lot and says which spaces are **free** and which are
**occupied**, then counts them:

```
lot.jpg: 51 free / 49 occupied of 100 spaces (49% full)
```

![A PKLot test frame: green boxes are free spaces, red are occupied](docs/example_prediction.jpg)

*A frame from days the model never saw (PKLot, CC BY 4.0): 51 free / 49 occupied, which is exactly
what the labels say.*

## Results

YOLOv8s trained on the official data, split by day, evaluated on 9 held-out days (1,167 frames):

| | |
|---|---|
| Detection, mAP50 / mAP50-95 | **0.995** / **0.906** |
| Free-space count exactly right | **87.5 %** of frames |
| Free-space count within +/-2 | **97.2 %** of frames |
| Mean error in free count | 0.52 spaces (frames average 59 spaces) |

The two Detectron2 baselines, trained on the same split, score lower: **Faster R-CNN 79.4** and
**RetinaNet 71.3** mAP50-95 against YOLOv8s's 90.6 (the original experiments' 40.6 and 37.9 came from
a class-count bug, now fixed).

Same three cameras as training; a new lot would need fine-tuning. Full tables, the original
experiments' numbers, and the caveats are in [docs/RESULTS.md](docs/RESULTS.md).

The project compares three object detectors on the [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/)
dataset - **YOLOv8**, **Faster R-CNN** and **RetinaNet** (the last two via Detectron2) - and
uses YOLOv8 for inference. It started as a university course project; this repository is the
rebuilt, cleaned-up version of it. See [docs/RESULTS.md](docs/RESULTS.md) for the numbers and
how far to trust them.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1. get the data: downloads the official 4.9 GB PKLot archive, converts it (about 5 minutes)
.venv/bin/python -m pklot.prepare_data

# 2. train (the recipe behind the shipped weights: 3 hours on an Apple M4; CUDA also works)
.venv/bin/python -m pklot.train_yolo --model yolov8s.pt --epochs 30 --batch 16 --cache ram

# 3. count spaces in your own photos - works right away with the shipped weights
.venv/bin/python -m pklot.predict --weights weights/yolov8s_pklot.pt --source lot.jpg

# 4. how accurate are the counts on the held-out test days?
.venv/bin/python -m pklot.count_eval --weights weights/yolov8s_pklot.pt --split test
```

Step 1 also runs automatically the first time you train. `predict` writes annotated images
and a `summary.csv` to `predictions/`. A trained model ships in `weights/yolov8s_pklot.pt`
(22 MB), so steps 1-2 are only needed to retrain.

## Data

The official PKLot release from its authors - no account, no API key. 12,416 labelled
1280x720 frames from three fixed cameras (PUCPR, UFPR04, UFPR05) in sun, cloud and rain,
about 694,000 parking spaces labelled occupied or empty.

`pklot.prepare_data` converts it to YOLO and COCO format and splits it **by capture date**:

| split | days | images | empty spaces | occupied spaces |
|---|---|---|---|---|
| train | 71 | 8,875 | 242,023 | 252,667 |
| valid | 19 | 2,374 | 86,097 | 43,703 |
| test | 9 | 1,167 | 29,858 | 39,407 |

Why by date: frames are taken every five minutes from a fixed camera, so a random split puts
near-identical images in both train and validation and makes the score look better than it is.
Whole days stay together here, and every lot appears in every split. The exact day lists are
written to `data/pklot/splits.json`.

## Models

| | script | notes |
|---|---|---|
| YOLOv8 | `pklot/train_yolo.py` | the model used for inference; trains on a laptop GPU |
| Faster R-CNN, RetinaNet | `pklot/train_detectron.py` | Detectron2, needs a CUDA GPU - use Colab or Kaggle |

```bash
.venv/bin/python -m pklot.train_yolo --help
python -m pklot.train_detectron --model faster_rcnn     # on a CUDA machine with Detectron2 installed
python -m pklot.train_detectron --model retinanet
```

The original recipe was `yolov8x`, 50 epochs, batch 20, 352 px (the script defaults), about an
hour on an A100. Detectron2 install: `pip install 'git+https://github.com/facebookresearch/detectron2.git'`
after PyTorch.

**Colab:** `notebooks/colab_train.ipynb` trains all three models and copies the weights to Google
Drive when each run finishes.

## Repository layout

```
pklot/
  prepare_data.py     download + convert PKLot to YOLO/COCO, split by date
  train_yolo.py       train + validate YOLOv8 (valid and test splits), copy results somewhere durable
  predict.py          count free/occupied spaces in images with trained YOLOv8 weights
  count_eval.py       frame-by-frame counting error against the labels, per parking lot
  train_detectron.py  Faster R-CNN / RetinaNet: train, evaluate, predict
  occupancy.py        free/occupied counting, independent of the detector
notebooks/colab_train.ipynb   Colab runner for all three models
weights/yolov8s_pklot.pt      the trained model (YOLOv8s, 22 MB)
docs/RESULTS.md               metrics, caveats, and what the original experiments got wrong
tests/                        python -m unittest discover -s tests
```

## What changed from the original experiments

1. **Class-count bug fixed.** The Detectron2 notebooks set `NUM_CLASSES = 2` against a three-category
   dataset, so `space-occupied` could never be learned (AP 0.000 in both runs). The class count is now
   read from the dataset.
2. **Learning-rate decay actually happens.** The original set `cfg.INPUT.GAMMA`, which Detectron2 ignores.
3. **Weights are saved.** The original "save to Drive" cells copied the wrong folders and the trained
   weights vanished with the Colab session; `--save-dir` now copies the real run.
4. **No credentials in code, and no expiring links.** Data comes from the official archive, not a
   per-user Roboflow export.
5. **Leak-free evaluation.** Date-based split instead of a random one, plus a separate held-out test split.
6. Occupancy counting is a tested function, not ad-hoc notebook code.

## Limitations

- The model is trained on three fixed cameras at 352 px. Expect to fine-tune it for a different lot,
  camera angle, or night-time footage (PKLot has daylight images only).
- Faster R-CNN and RetinaNet were retrained once each on a free Kaggle T4 (no repeats); their weights
  are not in this repository. The recipes are in [docs/RESULTS.md](docs/RESULTS.md).
- The report's "future directions" (reservations, license-plate recognition, EV charging) were never
  implemented; this repository covers the detection work only.

## Data attribution

PKLot: P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., A. L. Koerich,
"PKLot - A robust dataset for parking lot classification", *Expert Systems with Applications*, 2015.
Licensed CC BY 4.0. Pretrained detectors come from [Ultralytics](https://github.com/ultralytics/ultralytics)
(YOLOv8, AGPL-3.0 - the shipped weights are fine-tuned from YOLOv8s and carry that licence) and
[Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0).
