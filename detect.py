import cv2
import os
import shutil
from ultralytics import YOLO

# paths - update data.yaml to point to your downloaded dataset
DATA_YAML = "data.yaml"
WEIGHTS = "best.pt"

# train only if weights dont exist yet
if not os.path.exists(WEIGHTS):
    print("No weights found, starting training...")
    base = YOLO("yolov8n.pt")
    base.train(
        data=DATA_YAML,
        epochs=3,
        imgsz=640,
        device="cpu",  # change to 0 if you have a GPU
        project=".",
        name="runs",
    )
    shutil.copy("runs/weights/best.pt", WEIGHTS)
    print("Training done")

# load weights
model = YOLO(WEIGHTS)
CLASS_NAMES = model.names

# green for gear present, red for missing
COLORS = {
    "hard-hat":    (0, 255, 0),
    "vest":        (0, 200, 255),
    "no-hard-hat": (0, 0, 255),
    "no-vest":     (0, 0, 200),
    "person":      (200, 200, 200),
}

# open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting detection... Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    results = model(frame, conf=0.4, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = CLASS_NAMES[cls_id]
        color = COLORS.get(label.lower(), (255, 255, 0))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # make sure text doesnt go above the frame
        text_y = y1 - 8 if y1 - 8 > 10 else y1 + 20
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # show inference speed
    fps = results.speed.get("inference", 0)
    cv2.putText(frame, f"Inference: {fps:.1f}ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Safety Gear Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
