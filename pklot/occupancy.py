"""Turn detections into parking-lot occupancy numbers.

Framework independent on purpose: YOLOv8 and Detectron2 both just hand over a
list of predicted class names, so the free/occupied counting lives here once.
"""
from dataclasses import dataclass

FREE_WORDS = ("empty", "vacant", "free")
OCCUPIED_WORDS = ("occup",)


@dataclass(frozen=True)
class Occupancy:
    free: int
    occupied: int

    @property
    def total(self) -> int:
        return self.free + self.occupied

    @property
    def rate(self) -> float:
        """Fraction of detected spaces that are occupied (0.0 if none detected)."""
        return self.occupied / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (f"{self.free} free / {self.occupied} occupied "
                f"of {self.total} spaces ({self.rate:.0%} full)")


def classify(class_name: str):
    """Map a dataset class name to "free", "occupied" or None.

    PKLot's COCO export also carries a "spaces" parent category with no
    instances; it maps to None and is ignored.
    """
    name = class_name.lower()
    if any(word in name for word in FREE_WORDS):
        return "free"
    if any(word in name for word in OCCUPIED_WORDS):
        return "occupied"
    return None


def count_spaces(class_names) -> Occupancy:
    """Count free and occupied spaces from one image's predicted class names."""
    labels = [classify(name) for name in class_names]
    return Occupancy(free=labels.count("free"), occupied=labels.count("occupied"))
