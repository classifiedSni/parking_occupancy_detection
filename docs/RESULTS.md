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

*Added when the run finishes.*
