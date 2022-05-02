import shutil
from pathlib import Path

import cv2
import fiftyone as fo
from tqdm import tqdm
from moving_average import Annotator

if __name__ == "__main__":
    SAVE_DIR = Path("/namib")
    RAW_DIR = SAVE_DIR / "raw"
    DB_DIR = SAVE_DIR / "db"

    database_dir = fo.config.database_dir

    fns = sorted(RAW_DIR.iterdir())
    HEIGHT, WIDTH, _ = cv2.imread(str(fns[0])).shape
    annotator = Annotator((WIDTH, HEIGHT))

    samples = []
    for fn in tqdm(fns):
        frame = cv2.imread(str(fn))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annots, (thresh_frame, retval, anomaly_rate) = annotator(gray_frame)

        sample = fo.Sample(filepath=str(fn))
        detections = []
        for x1,y1,x2,y2 in annots:
            top_left_x_rel = x1 / WIDTH
            top_left_y_rel = y1 / HEIGHT
            width_rel = (x2 - x1) / WIDTH
            height_rel = (y2 - y1) / HEIGHT
            detections.append(
                fo.Detection(
                    bounding_box=[top_left_x_rel, top_left_y_rel, width_rel, height_rel],
                ))
        sample["moving_average"] = fo.Detections(detections=detections)
        samples.append(sample)

    dataset = fo.Dataset("namib")
    dataset.add_samples(samples)

    dataset.persistent = True
    fo.core.odm.sync_database()
    shutil.copytree(database_dir, str(DB_DIR))