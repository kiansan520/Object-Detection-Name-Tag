import os
import sys
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g. best.pt)')
parser.add_argument('--source', required=True, help='Path to video file (e.g. video.mp4)')
parser.add_argument('--thresh', type=float, default=0.5, help='Minimum confidence threshold (default: 0.5)')
parser.add_argument('--resolution', default=None, help='Resolution in WxH format (e.g. 640x480)')
parser.add_argument('--record', action='store_true', help='Save output video as demo1.avi')
args = parser.parse_args()

# Check model file
if not os.path.exists(args.model):
    print('ERROR: Model file not found.')
    sys.exit(1)

# Check video file
if not os.path.isfile(args.source):
    print('ERROR: Video file not found.')
    sys.exit(1)

# Load YOLO model
model = YOLO(args.model)
labels = model.names

# Parse resolution
resize = False
if args.resolution:
    try:
        resW, resH = map(int, args.resolution.lower().split('x'))
        resize = True
    except:
        print("Invalid resolution format. Use WxH, e.g., 640x480")
        sys.exit(1)

# Open video file
cap = cv2.VideoCapture(args.source)
if not cap.isOpened():
    print('ERROR: Failed to open video.')
    sys.exit(1)

# Set up recorder if needed
if args.record:
    if not resize:
        print('Recording requires --resolution to be set.')
        sys.exit(1)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# Colors for bounding boxes
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

# Process video
fps_list = []
while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        print('End of video.')
        break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    obj_count = 0
    for det in detections:
        conf = det.conf.item()
        if conf < args.thresh:
            continue
        obj_count += 1

        x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy().squeeze())
        cls_id = int(det.cls.item())

        # Label includes class, confidence, and top-left corner coordinates
        label = f"{labels[cls_id]} {conf:.2f} ({x1},{y1})"
        color = colors[cls_id % len(colors)]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    # Calculate FPS
    fps = 1 / (time.time() - start)
    fps_list.append(fps)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {obj_count}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow('YOLO Detection', frame)

    if args.record:
        recorder.write(frame)

    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
# Clean up
cap.release()
if args.record:
    recorder.release()
cv2.destroyAllWindows()
print(f"Average FPS: {np.mean(fps_list):.2f}")
