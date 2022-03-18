import time
from pathlib import Path
import json

import pafy
import cv2
import numpy as np
import onnxruntime
from tqdm import tqdm
import h5py

CONF_THRES = 0.25
IOU_THRES = 0.45
coco_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush']
class_agonistic_nms = False

SAVE_DIR = Path("/namib")
RAW_DIR = SAVE_DIR / "raw"
H5_DIR = SAVE_DIR / "proba.h5"

def overlaps(xyxy):
    """xyxy: (num_boxes, 4)"""
    x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
    y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
    x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
    y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    intersect_w = np.maximum(0, x2 - x1)
    intersect_h = np.maximum(0, y2 - y1)
    intersection = intersect_w * intersect_h
    union = area[:, None] + area[None, :] - intersection
    iou = intersection / union
    return iou

def analyze_pred(dets, th, tw):
    positive_mask = dets[:, 4] > CONF_THRES
    dets = dets[positive_mask]

    probas = dets[:, 5:].copy()
    dets[:, 5:] = dets[:, 5:] * dets[:, 4:5] # conf = obj_conf * cls_conf
    labels = np.argmax(dets[:, 5:], axis=1)
    confs = dets[np.arange(len(labels)), 5 + labels]
    boxes_xywh = dets[:, :4] # (center x, center y, width, height)
    boxes_xyxy = np.concatenate([
        boxes_xywh[:, :2] - boxes_xywh[:, 2:] / 2,
        boxes_xywh[:, :2] + boxes_xywh[:, 2:] / 2
    ], axis=1)

    offset = labels * max(th, tw) # for category-aware nms
    iou = overlaps(boxes_xyxy if class_agonistic_nms else (boxes_xyxy + offset[:,None]))
    selected_indices = []
    deleted_indices = set()

    for i in np.argsort(confs)[:-30:-1]:
        if int(i) in deleted_indices:
            continue
        selected_indices.append(i)
        to_delete = np.where(iou[i] > IOU_THRES)[0]
        iou[to_delete, :] = 0
        iou[:, to_delete] = 0
        for d in to_delete:
            deleted_indices.add(int(d))
    selected_boxes = boxes_xyxy[selected_indices]
    selected_labels= labels[selected_indices]
    selected_confs = confs[selected_indices]
    return np.concatenate([selected_boxes, selected_confs[:, None], selected_labels[:, None]], axis=1), probas[selected_indices]

session = onnxruntime.InferenceSession("./yolov5s.onnx")
print(list(map(lambda x: x.name, session.get_outputs())))

fns = sorted(RAW_DIR.iterdir())
h5 = h5py.File(str(H5_DIR), "w")
for fn in tqdm(fns):
    frame = cv2.imread(str(fn))

    h, w = frame.shape[:2]
    th = 640
    tw = int(th * w / h); tw -= tw % 64

    # (batch, channel, height, width)
    frame = cv2.resize(frame, (tw, th))
    batch = np.ascontiguousarray(frame[None, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)

    # (batch, num_boxes, 5 + num_classes)
    dets = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: batch})[0]
    assert dets.shape[2] == 5 + len(coco_names)
    dets = dets[0]

    objects, probas = analyze_pred(dets, th, tw) # (num_boxes, (xyxy, conf, label))

    h5.create_dataset(fn.stem, data=probas)