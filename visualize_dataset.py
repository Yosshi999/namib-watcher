import shutil
from pathlib import Path
import os


if __name__ == "__main__":
    SAVE_DIR = Path("/namib")
    RAW_DIR = SAVE_DIR / "raw"
    DB_DIR = SAVE_DIR / "db"

    os.environ["FIFTYONE_DATABASE_DIR"] = str(DB_DIR)
    import fiftyone as fo
    from fiftyone import ViewField as F

    assert "namib" in fo.list_datasets()

    dataset = fo.load_dataset("namib")
    session = fo.launch_app(dataset)

    bbox = F("bounding_box")
    bbox_area = bbox[2] * bbox[3]
    stage = fo.FilterLabels("moving_average", bbox_area > 0.004)
    view = dataset.add_stage(stage)

    session.view = view
    session.wait()
