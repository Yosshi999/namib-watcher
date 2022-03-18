import time
from pathlib import Path
import json

import pafy
import cv2
import numpy as np
import onnxruntime

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
class_agonistic_nms = True

SAVE_DIR = Path("/namib")
RAW_DIR = SAVE_DIR / "raw"
DETECT_DIR = SAVE_DIR / "annotated"
JSON_DIR = SAVE_DIR / "json"

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
    return np.concatenate([selected_boxes, selected_confs[:, None], selected_labels[:, None]], axis=1)

url = "https://www.youtube.com/watch?v=ydYDqZQpim8"
video = pafy.new(url)
best = video.getbest(preftype="mp4")

session = onnxruntime.InferenceSession("./yolov5s.onnx")
print(list(map(lambda x: x.name, session.get_outputs())))

print("fetching", best.url, "@", best.resolution)
capture = cv2.VideoCapture(best.url)
frame_idx = 0

while True:
    start = time.time()
    epoch = start
    logmes = "[%04d] " % frame_idx

    capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    grabbed, frame = capture.read()
    if not grabbed:
        print("failed to grab frame")
        break
    cv2.imwrite(str(RAW_DIR / f"{frame_idx:04d}.png"), frame)

    nepoch = time.time()
    logmes += f"fetch: {nepoch - epoch:.3f}s "
    epoch = nepoch

    h, w = frame.shape[:2]
    th = 640
    tw = int(th * w / h); tw -= tw % 64
    # tw = 640

    # (batch, channel, height, width)
    frame = cv2.resize(frame, (tw, th))
    batch = np.ascontiguousarray(frame[None, :, :, ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255.0)

    nepoch = time.time()
    logmes += f"preprocess: {nepoch - epoch:.3f}s "
    epoch = nepoch

    # (batch, num_boxes, 5 + num_classes)
    dets = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: batch})[0]
    assert dets.shape[2] == 5 + len(coco_names)
    dets = dets[0]

    nepoch = time.time()
    logmes += f"forward: {nepoch - epoch:.3f}s "
    epoch = nepoch

    objects = analyze_pred(dets, th, tw) # (num_boxes, (xyxy, conf, label))

    nepoch = time.time()
    logmes += f"postprocess: {nepoch - epoch:.3f}s ({len(objects):2d} objects) "
    epoch = nepoch

    annot = []
    for obj in objects:
        cv2.rectangle(frame, (int(obj[0]), int(obj[1])), (int(obj[2]), int(obj[3])), (0, 255, 0), 2)
        cv2.putText(frame, coco_names[int(obj[5])] + f" {obj[4]:02.0%}", (int(obj[0]), int(obj[1])-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), thickness=2)
        annot.append({
            "x1": int(obj[0]),
            "y1": int(obj[1]),
            "x2": int(obj[2]),
            "y2": int(obj[3]),
            "cat_id": int(obj[5]),
            "cat_name": coco_names[int(obj[5])],
            "confidence": float(obj[4]),
            "image_width": frame.shape[1],
            "image_height": frame.shape[0],
            "unixtime": time.time(),
        })
    with (JSON_DIR / f"{frame_idx:04d}.json").open("w") as f:
        json.dump(annot, f)

    end = time.time()
    logmes += f"all: {end - start:.3f}s"
    print(logmes)

    cv2.imwrite(str(DETECT_DIR / f"{frame_idx:04d}.png"), frame)

    time.sleep(10)
    frame_idx += 1