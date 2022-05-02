from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm

class Annotator:
    def __init__(self, image_wh: Tuple[int, int], alpha=0.5, kernel_size=5, masked_update=True, nonui_top_pct=0.24, reset_pct=0.1):
        self.width, self.height = image_wh
        self.alpha = alpha
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.masked_update = masked_update
        self.nonui_top = int(self.height * nonui_top_pct)
        self.reset_pct = reset_pct

        self.thresh_frame = np.zeros((self.height, self.width), np.uint8)
        self.init()

    def init(self):
        self.avg = None

    def __call__(self, gray_frame):
        force_update = False

        gray = gray_frame[self.nonui_top:, :]
        if self.avg is None:
            self.avg = gray.copy().astype("float")
        # y <- (1 - alpha) * y + alpha * x if mask != 0
        if self.masked_update:
            new_avg = cv2.accumulateWeighted(gray, self.avg.copy(), self.alpha)
            delta = cv2.absdiff(gray, cv2.convertScaleAbs(new_avg))
        else:
            cv2.accumulateWeighted(gray, self.avg, self.alpha)
            delta = cv2.absdiff(gray, cv2.convertScaleAbs(self.avg))

        retval, thresh = cv2.threshold(cv2.GaussianBlur(delta, (5, 5), 0), 10, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, self.kernel, iterations=1)
        thresh = cv2.dilate(thresh, self.kernel, iterations=1)

        anomaly_rate = (thresh > 0).sum() / thresh.size
        if anomaly_rate > self.reset_pct:
            thresh[:] = 0
            force_update = True
        self.thresh_frame[:self.nonui_top, :] = 0
        self.thresh_frame[self.nonui_top:, :] = thresh

        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annots = [] # list of xyxy_abs

        if self.masked_update:
            mask = np.ones_like(thresh)
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            annots.append((x, y + self.nonui_top, x+w, y + self.nonui_top + h))

            if self.masked_update:
                mask = cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)
        if self.masked_update:
            if force_update:
                self.avg = cv2.accumulateWeighted(gray, self.avg, self.alpha)
            else:
                self.avg = cv2.accumulateWeighted(gray, self.avg, self.alpha, mask)
                self.avg = cv2.accumulateWeighted(gray, self.avg, self.alpha/10)

        return annots, (self.thresh_frame, retval, anomaly_rate)

if __name__ == "__main__":
    SAVE_DIR = Path("/namib")
    RAW_DIR = SAVE_DIR / "raw"
    VIDEO_FN = SAVE_DIR / "moving_average.mp4"

    fns = sorted(RAW_DIR.iterdir())
    HEIGHT, WIDTH, _ = cv2.imread(str(fns[0])).shape
    print(f"image shape: {WIDTH}x{HEIGHT}")
    O_HEIGHT = HEIGHT // 2
    O_WIDTH = WIDTH // 2

    annotator = Annotator((WIDTH, HEIGHT))

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(str(VIDEO_FN), fourcc, 10.0, (WIDTH, HEIGHT))

    whole_img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    for fn in tqdm(fns):
        frame = cv2.imread(str(fn))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        whole_img[:O_HEIGHT, :O_WIDTH, :] = cv2.resize(frame, (O_WIDTH, O_HEIGHT))
        whole_img[:O_HEIGHT, O_WIDTH:, :] = cv2.resize(gray_frame, (O_WIDTH, O_HEIGHT))[:,:,None]

        annots, (thresh_frame, retval, anomaly_rate) = annotator(gray_frame)

        thresh_info_frame = cv2.putText(cv2.cvtColor(thresh_frame, cv2.COLOR_GRAY2BGR), f"{retval:.2f}, {anomaly_rate:.2f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        whole_img[O_HEIGHT:, :O_WIDTH, :] = cv2.resize(thresh_info_frame, (O_WIDTH, O_HEIGHT))

        annot_frame = frame.copy()
        for x1,y1,x2,y2 in annots:
            annot_frame = cv2.rectangle(annot_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        whole_img[O_HEIGHT:, O_WIDTH:, :] = cv2.resize(annot_frame, (O_WIDTH, O_HEIGHT))
        writer.write(whole_img)
