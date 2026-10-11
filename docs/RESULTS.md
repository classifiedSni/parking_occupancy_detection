# Results

Two parts: the numbers recovered from the original experiments (below), and a retrain of
YOLOv8 with this repository's code on the official PKLot data (last section).

## Original experiments (recovered)

Read out of the saved outputs of the original Colab notebooks. The trained weights
themselves were **not** recovered, so these numbers could not be re-checked.

## Datasets (two different Roboflow exports of PKLot)

| | Detectron2 (Faster R-CNN, RetinaNet) | YOLOv8 |
|---|---|---|
| Format | COCO JSON | YOLO txt + data.yaml |
| Categories | `spaces` (empty parent, 0 instances), `space-empty`, `space-occupied` | `space-empty`, `space-occupied` |
| Train | 8,502 images - 265,908 empty / 231,948 occupied | 8,690 images (189 without labels) |
| Evaluated on | `test`: 1,242 images - 36,584 empty / 34,100 occupied | `valid`: 2,483 images - 143,316 instances |

## Models

| Model | Setup | Result on held-out split |
|---|---|---|
| **YOLOv8x** | A100 40 GB, Ultralytics 8.2.17, 50 epochs, batch 20, 352 px, 1.07 h. best.pt 137 MB, ONNX 260 MB, 1.4 ms/image | P 0.999 - R 0.999 - **mAP50 0.995** - **mAP50-95 0.994** (identical for both classes) |
| **Faster R-CNN** R50-FPN | Colab T4, 10,000 iter, batch 5, lr 0.001, ~2.5 h | **AP 40.6** - AP50 48.2 - AP75 47.5 - APs 37.7 - APm 41.9 - APl 45.0 |
| **RetinaNet** R50-FPN | Colab L4, 5,000 iter, batch 20, lr 0.001, 2 h 21 min | **AP 37.9** - AP50 44.8 - AP75 44.1 - APs 35.6 - APm 39.1 - APl 42.3 |

Per-category AP (bbox):

| | space-empty | space-occupied |
|---|---|---|
| Faster R-CNN | 81.3 | **0.0** |
| RetinaNet | 75.8 | **0.0** |

## The Detectron2 numbers are distorted by a bug

Both Detectron2 notebooks set `NUM_CLASSES = 2`, but the COCO export has **three** categories.
Detectron2 maps them to indices 0, 1, 2, so `space-occupied` (index 2) fell outside the
output layer and could never be learned. The overall AP is exactly half the
`space-empty` AP in both runs, which is the signature of one real class scoring zero
(81.263 / 2 = 40.63; 75.777 / 2 = 37.89).

The report attributes the 0.00 occupied AP to "a need for further optimization"; it was a
configuration error. `pklot/train_detectron.py` now reads the class count from the dataset.

**Consequence for the report's conclusion:** the "YOLOv8 99% vs Detectron2 45%" comparison
is overstated. The corrected Detectron2 models will very likely score far higher than 45 -
retraining is needed to know by how much. YOLOv8 is still plausibly the best choice
(single-stage, 1.4 ms/image), but the size of the gap is unmeasured.

## Other caveats

- **The comparison is not like-for-like**: different exports, different splits, different
  evaluation sets, and YOLOv8's metric covers two real classes while the Detectron2 metric
  was dragged down by the broken one.
- **YOLOv8's ~0.995 is probably optimistic.** PKLot is time-lapse footage from three fixed
  cameras; a random train/valid split puts near-identical frames on both sides. How the
  Roboflow split was made isn't recorded, but a day-held-out evaluation is a stricter test -
  that is what the retrain below uses.
- The Faster R-CNN notebook's *config cell* (batch 20, 20,000 iter) was edited after the run
  whose output it holds; the saved log shows batch 5. The numbers above use the batch-5 run.

## Numbers that differ between documents

| | Report (2024) | Saved notebook output |
|---|---|---|
| Faster R-CNN AP50 | 47.985 | 48.231 |
| Faster R-CNN APs / APm | 35.219 / 36.939 | 37.735 / 41.860 |
| RetinaNet AP / AP50 / AP75 | 37.89 / 44.85 / 44.10 | 37.888 / 44.848 / 44.101 (match) |
| YOLOv8 mAP50-95 | not stated (slides: 99.4) | 0.994 |

The report's Faster R-CNN figures come from a run whose output was not saved in the notebooks.
The 2023 midterm deck quotes an earlier Faster R-CNN run (AP50 93.1, APs 54.9, APm 60.1) with
a different setup; that run is not recoverable either.

## Retrained on the official data (this repository)

Run with this repository's code on the official PKLot archive, split **by capture date**
(train 71 days / valid 19 / test 9; see the README). The test days were never used for
training or for choosing anything.

Setup: YOLOv8s, COCO-pretrained, 30 epochs, batch 16, 352 px, one Apple M4 laptop GPU (MPS),
3.16 hours, Ultralytics 8.4.175 / PyTorch 2.14.1. The trained weights ship in
`weights/yolov8s_pklot.pt` (22 MB).

### Detection metrics

| split | precision | recall | mAP50 | mAP50-95 | space-empty mAP50-95 | space-occupied mAP50-95 |
|---|---|---|---|---|---|---|
| valid (19 days, 2,374 images) | 0.980 | 0.993 | 0.993 | 0.924 | 0.940 | 0.909 |
| **test** (9 days, 1,167 images) | 0.996 | 0.996 | **0.995** | **0.906** | 0.934 | 0.879 |

mAP50 reached 0.957 after one epoch and 0.99 by epoch 20. mAP50-95 was still creeping up over the last
epochs (0.918 at epoch 25, 0.924 at epoch 29, the saved checkpoint), so more epochs or a larger model
would gain a little.

### Counting accuracy (what a "free spaces" sign actually needs)

`python -m pklot.count_eval --weights weights/yolov8s_pklot.pt --split test` compares predicted
free/occupied counts with the labelled counts, frame by frame, at confidence 0.5:

| test frames | frames | spaces per frame | mean error, free count | free count exactly right | free count within +/-2 |
|---|---|---|---|---|---|
| all | 1,167 | 59 | 0.52 | 87.5 % | 97.2 % |
| PUCPR | 441 | 100 | 1.18 | 86.2 % | 92.5 % |
| UFPR04 | 320 | 28 | 0.10 | 90.3 % | 100 % |
| UFPR05 | 406 | 40 | 0.14 | 86.7 % | 100 % |

The mean error on the *occupied* count is 1.10 overall (PUCPR 2.70). PUCPR, the largest lot, is the
hardest: its spaces are the smallest and densest (about 13 x 14 px at 352 px wide, against about
28 x 19 px in the other two lots). Training or running at a larger image size was not evaluated.

### How to read this

- **Not comparable to the original YOLOv8x number (mAP50-95 0.994).** That run used a different
  (random) split, a much larger model (`yolov8x`, 68 M parameters against 11 M) and a different export. How much of the
  0.994 -> 0.906 gap is the stricter split and how much is the smaller model is not known; only a
  `yolov8x` run on this split would separate them. The date split is the more trustworthy of the two.
- **Same three cameras.** The test days are new days, not new cameras or lots. These numbers say
  nothing about a lot the model has not seen; expect to fine-tune for one.
- **The Detectron2 models were not retrained** (no CUDA GPU), so the corrected Faster R-CNN /
  RetinaNet comparison is still open.
- **A pitfall worth knowing.** Passing one long list of image paths to Ultralytics `predict()` turns
  it into a single giant batch whose NMS step times out and silently drops detections; the first
  version of `count_eval` did exactly that and reported counts that were 25 spaces off. It now
  processes 16 frames at a time, and single-image predictions agree with the labels.
