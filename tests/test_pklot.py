"""Run with:  python -m unittest discover -s tests -v"""
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

from PIL import Image  # noqa: E402

from pklot import prepare_data  # noqa: E402
from pklot.occupancy import Occupancy, classify, count_spaces  # noqa: E402
from pklot.train_yolo import write_data_yaml  # noqa: E402


class OccupancyTests(unittest.TestCase):
    def test_classify_pklot_class_names(self):
        self.assertEqual(classify("space-empty"), "free")
        self.assertEqual(classify("space-occupied"), "occupied")
        self.assertIsNone(classify("spaces"))  # empty parent category in the COCO export

    def test_count_spaces(self):
        names = ["space-empty", "space-occupied", "space-occupied", "space-empty", "space-empty"]
        result = count_spaces(names)
        self.assertEqual((result.free, result.occupied, result.total), (3, 2, 5))
        self.assertAlmostEqual(result.rate, 0.4)

    def test_count_spaces_accepts_generator_and_empty(self):
        self.assertEqual(count_spaces(n for n in ["space-empty"]), Occupancy(free=1, occupied=0))
        empty = count_spaces([])
        self.assertEqual(empty.total, 0)
        self.assertEqual(empty.rate, 0.0)

    def test_str(self):
        self.assertEqual(str(Occupancy(free=3, occupied=1)), "3 free / 1 occupied of 4 spaces (25% full)")


class DataYamlTests(unittest.TestCase):
    def test_paths_are_rewritten_and_names_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.yaml").write_text(yaml.safe_dump({
                "train": "../train/images", "val": "../valid/images", "test": "../test/images",
                "nc": 2, "names": ["space-empty", "space-occupied"]}))
            (root / "train" / "images").mkdir(parents=True)
            written = yaml.safe_load(write_data_yaml(root).read_text())
            self.assertEqual(written["path"], str(root.resolve()))
            self.assertEqual((written["train"], written["val"]), ("train/images", "valid/images"))
            self.assertNotIn("test", written)  # no test/images folder present
            self.assertEqual(written["names"], ["space-empty", "space-occupied"])


XML = """<parking id="demo">
  <space id="1" occupied="1"><rotatedRect><center x="100" y="50"/><size w="40" h="20"/><angle d="0"/></rotatedRect>
    <contour><point x="80" y="40"/><point x="120" y="40"/><point x="120" y="60"/><point x="80" y="60"/></contour></space>
  <space id="2" occupied="0"><rotatedRect><center x="300" y="200"/><size w="60" h="20"/><angle d="90"/></rotatedRect></space>
  <space id="3"><rotatedRect><center x="10" y="10"/><size w="10" h="10"/><angle d="0"/></rotatedRect></space>
</parking>"""


class PrepareDataTests(unittest.TestCase):
    def test_read_spaces_contour_rotated_rect_and_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "a.xml"
            xml.write_text(XML)
            boxes = prepare_data.read_spaces(xml, 1280, 720)
        self.assertEqual(len(boxes), 2)  # space 3 has no occupancy label -> dropped
        occupied, empty = boxes
        self.assertEqual(occupied, (1, 80.0, 40.0, 120.0, 60.0))  # from <contour>
        cls, x0, y0, x1, y1 = empty  # no contour: 60x20 rect rotated 90 degrees -> 20 wide, 60 tall
        self.assertEqual(cls, 0)
        self.assertAlmostEqual(x1 - x0, 20.0)
        self.assertAlmostEqual(y1 - y0, 60.0)

    def test_split_keeps_each_day_in_one_split_and_every_lot_in_every_split(self):
        dates = {lot: [(f"2012-09-{d:02d}", 50) for d in range(1, 21)] for lot in ("A", "B")}
        assignment = prepare_data.assign_splits(dates)
        self.assertEqual(len(assignment), 40)
        for lot in ("A", "B"):
            self.assertEqual({v for (l, _), v in assignment.items() if l == lot}, set(prepare_data.SPLITS))
        counts = {s: sum(v == s for v in assignment.values()) for s in prepare_data.SPLITS}
        self.assertGreater(counts["train"], counts["valid"])
        self.assertGreater(counts["valid"], counts["test"])

    def test_convert_writes_yolo_and_coco(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, out = Path(tmp) / "staging", Path(tmp) / "out"
            for day in range(1, 11):
                folder = staging / "LOT" / "Sunny" / f"2012-09-{day:02d}"
                folder.mkdir(parents=True)
                Image.new("RGB", (1280, 720)).save(folder / "f.jpg")
                (folder / "f.xml").write_text(XML)
            stats = prepare_data.convert(staging, out)
            self.assertEqual(sum(s["images"] for s in stats.values()), 10)
            self.assertEqual(sum(s["boxes"][1] for s in stats.values()), 10)  # one occupied per frame
            train_label = next((out / "train" / "labels").glob("*.txt")).read_text().splitlines()
            self.assertEqual(train_label[0].split()[0], "1")
            self.assertAlmostEqual(float(train_label[0].split()[1]), 100 / 1280, places=4)
            coco = json.loads((out / "train" / "_annotations.coco.json").read_text())
            self.assertEqual([c["id"] for c in coco["categories"]], [1, 2])
            self.assertEqual(len(coco["annotations"]), 2 * len(coco["images"]))
            self.assertTrue((out / "data.yaml").exists())


if __name__ == "__main__":
    unittest.main()
