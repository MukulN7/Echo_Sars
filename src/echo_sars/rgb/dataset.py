import json
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional


class FLIRDataset:
    """
    Loader for FLIR ADAS v2 datasets (RGB and Thermal splits).
    Parses COCO annotation json and indexes images and ground truth person instances.
    """

    def __init__(self, split_dir: str | Path):
        self.split_dir = Path(split_dir)
        self.coco_path = self.split_dir / "coco.json"
        self.images: Dict[int, Dict[str, Any]] = {}
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        self.person_category_id: Optional[int] = None
        self._load_annotations()

    def _load_annotations(self) -> None:
        if not self.coco_path.exists():
            raise FileNotFoundError(f"FLIR COCO annotation not found at {self.coco_path}")

        with open(self.coco_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Map categories to find 'person'
        for cat in data.get("categories", []):
            if cat.get("name") == "person":
                self.person_category_id = cat.get("id")
                break

        for img in data.get("images", []):
            img_id = img["id"]
            self.images[img_id] = img
            self.annotations_by_image[img_id] = []

        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id in self.annotations_by_image:
                self.annotations_by_image[img_id].append(ann)

    def get_image_info(self, image_id: int) -> Dict[str, Any]:
        return self.images[image_id]

    def get_image_path(self, image_id: int) -> Path:
        img_info = self.images[image_id]
        return self.split_dir / img_info["file_name"]

    def get_person_annotations(self, image_id: int) -> List[Dict[str, Any]]:
        if self.person_category_id is None:
            return []
        return [
            ann for ann in self.annotations_by_image.get(image_id, [])
            if ann.get("category_id") == self.person_category_id
        ]

    def iter_images(self) -> Generator[Dict[str, Any], None, None]:
        for img_id, info in self.images.items():
            yield info

    def __len__(self) -> int:
        return len(self.images)

