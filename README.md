# Real-Time Safety Gear Detection System

A computer vision system that detects hard hats and safety vests on people using a live webcam feed. Built for the IoT & Embedded Systems intern assessment.

---

## Use Case

Construction sites and electrical grid stations require workers to wear PPE (Personal Protective Equipment) at all times. This system monitors a live camera feed and flags whether people in frame are wearing a hard hat and/or safety vest in real time.

The use case is directly relevant to electrical utility environments — during my internship at IESCO (Islamabad Electric Supply Company), PPE compliance is a daily safety requirement at 132 KV grid stations.

---

## Hardware

**Target hardware:** Raspberry Pi 4 or equivalent single-board computer with camera module.

**Substitution used:** Laptop with built-in webcam (ASUS/HP), running inference locally in real time via CPU.

No dedicated edge hardware was available during development. The system runs entirely on-device with no cloud dependency and processes frames in real time, satisfying the edge inference requirement. The same code runs on a Raspberry Pi 4 with no changes other than setting `device="cpu"` in the training config, which is already the default.

---

## Model

- **Base model:** YOLOv8n (nano) — chosen for its low latency on CPU
- **Fine-tuned on:** Hard Hat and Safety Vest dataset from Roboflow Universe
- **Classes detected:** `hard-hat`, `no-hard-hat`, `vest`, `no-vest`, `person`
- **Training:** 20 epochs, image size 640x640, CPU

YOLOv8n was picked over larger variants because inference speed matters more than raw accuracy in a real-time live feed scenario. On a laptop CPU it runs at roughly 80–150ms per frame, which is acceptable for a safety monitoring use case.

---

## Dataset

**Source:** [Hard Hat - Safety Vest Dataset on Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/hard-hat-sample-2kga5)

- Annotated images of workers in construction/industrial settings
- Labels: hard-hat, no-hard-hat, vest, no-vest, person
- Format: YOLOv8 (YOLO txt annotations + data.yaml)
- Split: train / valid / test folders included

---

## Project Structure

```
safety-gear-detection/
├── detect.py          # main script: trains model then runs live detection
├── requirements.txt   # Python dependencies
└── README.md
```

> **Note on weights:** Model weights (`best.pt`) are generated locally during training and are not included in the repo due to file size. See setup instructions below — training runs automatically on first launch.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Samra761/safety-gear-detection.git
cd safety-gear-detection
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python
```

Python 3.8 or higher required.

### 3. Download the dataset

Download the dataset in YOLOv8 format from Roboflow:
[Hard Hat - Safety Vest Dataset](https://universe.roboflow.com/roboflow-universe-projects/hard-hat-sample-2kga5)

Extract it and note the path to `data.yaml`. Update the path in `detect.py` if needed:

```python
data=r"path\to\hard-hat - safety-vest.v1i.yolo26\data.yaml"
```

### 4. Run

```bash
python detect.py
```

On first run, the script trains YOLOv8n on the dataset (20–40 min on CPU). After training completes, the webcam feed opens automatically with live detections.

Press `q` to quit.

---

## How It Works

1. YOLOv8n is fine-tuned on the safety gear dataset for 20 epochs
2. Trained weights are saved to `Desktop/safety_gear/weights/best.pt`
3. OpenCV captures frames from the webcam at 1280x720
4. Each frame is passed to the YOLO model for inference
5. Bounding boxes are drawn with color coding:
   - **Green** → hard hat detected
   - **Orange** → vest detected
   - **Red** → no hard hat / no vest
6. Inference time is displayed on the feed in milliseconds

---

## Requirements

```
ultralytics
opencv-python
```

---

## Design Decisions and Trade-offs

| Decision | Reason |
|---|---|
| YOLOv8n over larger variants | Faster inference on CPU; accuracy trade-off acceptable for live demo |
| Fine-tuning vs pretrained only | Fine-tuning on domain-specific data improves detection of safety gear vs general COCO classes |
| Laptop webcam over Pi | No Pi available; laptop satisfies real-time edge inference requirement |
| 20 epochs | Sufficient for convergence on this dataset size without overfitting |
| conf=0.4 threshold | Reduces false positives while keeping sensitivity reasonable |

---

## Author

**Samra Mehmood**  
[GitHub](https://github.com/Samra761) · [LinkedIn](https://linkedin.com/in/samra-mehmood)
