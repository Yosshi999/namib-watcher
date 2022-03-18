from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']

SAVE_DIR = Path("/namib")
RAW_DIR = SAVE_DIR / "raw"
DETECT_DIR = SAVE_DIR / "annotated"
H5_DIR = SAVE_DIR / "proba.h5"
PLOT_FN = SAVE_DIR / "plot.png"
VIDEO_FN = SAVE_DIR / "plot.mp4"

h5 = h5py.File(str(H5_DIR), "r")
fns = sorted(RAW_DIR.iterdir())
MEM = 60 * 3

fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
writer = cv2.VideoWriter(str(VIDEO_FN), fourcc, 10.0, (1088, 640+100))

data = []
queue = []
scores = []
for fn in fns[:MEM]:
    feat = h5[fn.stem][:]
    queue.append(len(feat))
    data.append(feat)
    scores.append(np.nan)

data = np.concatenate(data, axis=0)

for fn in tqdm(fns[MEM:]):
    feat = h5[fn.stem][:].reshape(-1, len(coco_names))
    sz = queue.pop(0)
    queue.append(len(feat))
    data = np.concatenate([data[sz:], feat], axis=0)
    if len(feat) > 0:
        score = 0
        for f in feat:
            dist = jensenshannon(data, f)
            score += sum(sorted(dist)[:5])
        scores.append(score / len(feat))
    else:
        scores.append(np.nan)

plt.plot(scores)
plt.savefig(str(PLOT_FN))

ymin = np.nanmin(scores)
ymax = np.nanmax(scores)
pts = []
for i,y in enumerate(scores):
    if np.isfinite(y):
        pts.append([i / len(scores), (y - ymin) / (ymax - ymin)])
pts = np.array(pts)

whole_img = np.zeros((640+100, 1088, 3), np.uint8)
plot_img = np.zeros((100, 1088, 3), np.uint8)
for i, fn in enumerate(tqdm(fns)):
    annot_img = cv2.imread(str(DETECT_DIR / (fn.name)))
    whole_img[:640] = annot_img
    plot_img[:] = [255, 255, 255]
    cv2.polylines(plot_img, [(pts * np.array([1088, 100])).astype(np.int32)], False, color=(255,0,0), thickness=1)
    cv2.line(plot_img, [int(i / len(scores) * 1088), 0], [int(i / len(scores) * 1088), 100], (0, 0, 255), thickness=2)
    whole_img[640:] = plot_img

    writer.write(whole_img)