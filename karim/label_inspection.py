import cv2
import glob

label_paths = glob.glob("datasets/raw/labels/train2/*.txt")

areas = []

for lp in label_paths:
    with open(lp, "r") as f:
        for line in f:
            cls, x, y, w, h = map(float, line.split())

            # YOLO format → pixel size (assume 1024x1024)
            img_size = 1024
            w_px = w * img_size
            h_px = h * img_size

            areas.append(w_px * h_px)

print("Min area:", min(areas))
print("Max area:", max(areas))
print("Avg area:", sum(areas)/len(areas))