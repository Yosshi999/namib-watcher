from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ALPHA = 0.5
kernel = np.ones((5,5), np.uint8)
MASKED_UPDATE = True
NONUI_TOP_PCT = 0.24

SAVE_DIR = Path("/namib")
RAW_DIR = SAVE_DIR / "raw"
VIDEO_FN = SAVE_DIR / "moving_average.mp4"

fns = sorted(RAW_DIR.iterdir())
HEIGHT, WIDTH, _ = cv2.imread(str(fns[0])).shape
print(f"image shape: {WIDTH}x{HEIGHT}")
O_HEIGHT = HEIGHT // 2
O_WIDTH = WIDTH // 2
NONUI_TOP = int(NONUI_TOP_PCT * HEIGHT)

fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
writer = cv2.VideoWriter(str(VIDEO_FN), fourcc, 10.0, (WIDTH, HEIGHT))

whole_img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

thresh_frame = np.zeros((HEIGHT, WIDTH), np.uint8)
avg = None
for fn in tqdm(fns):
    frame = cv2.imread(str(fn))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    whole_img[:O_HEIGHT, :O_WIDTH, :] = cv2.resize(frame, (O_WIDTH, O_HEIGHT))
    whole_img[:O_HEIGHT, O_WIDTH:, :] = cv2.resize(gray_frame, (O_WIDTH, O_HEIGHT))[:,:,None]
    gray = gray_frame[NONUI_TOP:, :]

    if avg is None:
        avg = gray.copy().astype("float")
    # y <- (1 - alpha) * y + alpha * x if mask != 0
    if MASKED_UPDATE:
        new_avg = cv2.accumulateWeighted(gray, avg.copy(), ALPHA)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(new_avg))
    else:
        cv2.accumulateWeighted(gray, avg, ALPHA)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    retval, thresh = cv2.threshold(cv2.GaussianBlur(delta, (5, 5), 0), 10, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    thresh_frame[:NONUI_TOP, :] = 0
    thresh_frame[NONUI_TOP:, :] = thresh
    thresh_info_frame = cv2.putText(cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR), f"{retval:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
    whole_img[O_HEIGHT:, :O_WIDTH, :] = cv2.resize(thresh_info_frame, (O_WIDTH, O_HEIGHT))

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annot_frame = frame.copy()
    if MASKED_UPDATE:
        mask = np.ones_like(thresh)
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        annot_frame = cv2.rectangle(annot_frame, (x, y + NONUI_TOP), (x+w, y + NONUI_TOP + h), (0, 255, 0), 3)
        if MASKED_UPDATE:
            mask = cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
    if MASKED_UPDATE:
        avg = cv2.accumulateWeighted(gray, avg, ALPHA, mask)
    # avg = cv2.accumulateWeighted(gray, avg, ALPHA/10)
    whole_img[O_HEIGHT:, O_WIDTH:, :] = cv2.resize(annot_frame, (O_WIDTH, O_HEIGHT))
    writer.write(whole_img)