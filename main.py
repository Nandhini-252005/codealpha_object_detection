import cv2 # type: ignore
from ultralytics import YOLO # type: ignore
import numpy as np # type: ignore

# Load the YOLOv8 model (make sure yolov8n.pt is in the same folder)
model = YOLO("yolov8n.pt")

# Load the video file (change the filename if needed)
cap = cv2.VideoCapture("video.mp4")

# Define video writer to save output
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0,
                      (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        print("✅ Video completed.")
        break

    # Perform detection and tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Draw results
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.int().cpu().numpy()
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    # Write and display frame
    out.write(frame)
    cv2.imshow("YOLOv8 Object Detection and Tracking", frame)

    # Press Q to quit early
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
