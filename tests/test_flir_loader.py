from pathlib import Path
import pytest
from echo_sars.rgb.dataset import FLIRDataset
from echo_sars.config import FLIR_ROOT


def test_flir_dataset_loading():
    val_thermal = FLIR_ROOT / "images_thermal_val"
    if not val_thermal.exists():
        pytest.skip(f"FLIR thermal directory not found: {val_thermal}")

    dataset = FLIRDataset(val_thermal)
    assert len(dataset) > 0
    assert dataset.person_category_id is not None
    assert dataset.person_category_id == 1  # Standard FLIR ADAS category for person

    # Verify first image access
    first_info = next(dataset.iter_images())
    assert "id" in first_info
    assert "file_name" in first_info
    img_path = dataset.get_image_path(first_info["id"])
    assert img_path.exists()


def test_missing_dataset_raises_error():
    with pytest.raises(FileNotFoundError):
        FLIRDataset(Path("non_existent_directory_flir"))

