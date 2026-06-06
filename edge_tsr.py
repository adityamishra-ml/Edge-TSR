import cv2
import time
import psutil
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO
from PIL import Image
from collections import deque
import numpy as np

# ===============================
# CONFIG
# ===============================
VIDEO_PATH = "test-set/13.mp4" 
DETECTOR_PATH = "detector-all.pt"
CLASSIFIER_PATH = "resnet50-fold-0.pth"
DATASET_TRAIN_DIR = "/home/aditya/Desktop/RPi/Classify/dataset/fold-0/train"
DEVICE = "cuda"

DETECT_EVERY = 1
CONF_THRESH = 0.5
PADDING = 0.20

RESULTS_SAVE_PATH = "1_inference_results.json"
PRED_SAVE_PATH = "1_predictions.json"
EVAL_FRAMES_PATH = "1_evaluated_frames.json"

device = torch.device(DEVICE)

# ===============================
# LOAD MODELS
# ===============================
yolo = YOLO(DETECTOR_PATH)

classes = sorted([
    d for d in os.listdir(DATASET_TRAIN_DIR)
    if os.path.isdir(os.path.join(DATASET_TRAIN_DIR, d))
])

class_to_idx = {cls: i for i, cls in enumerate(classes)}
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

classifier = models.resnet50(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, len(classes))
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
classifier = classifier.to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])

# ===============================
# VIDEO
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened()

frame_id = 0
total_frames = 0

active_boxes = []
all_predictions = []
evaluated_frames = []

track_memory = {}

detector_times = []
classifier_times = []
tracker_times = []
fps_list = []

cpu_usages = []
cpu_temps = []
ram_usages = []

start_time = time.time()

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except:
        return None

# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % DETECT_EVERY == 0:

        evaluated_frames.append(frame_id)

        t0 = time.time()

        results = yolo.track(
            frame,
            imgsz=640,
            conf=CONF_THRESH,
            tracker="bytetrack.yaml",
            persist=True,
            device=DEVICE,
            verbose=False
        )[0]

        detector_times.append(time.time() - t0)

        new_boxes = []
        frame_preds = []

        if results.boxes is not None:

            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            ids = results.boxes.id

            if ids is not None:
                ids = ids.int().cpu().tolist()
            else:
                ids = [-1] * len(boxes)

            h, w, _ = frame.shape

            for i, box in enumerate(boxes):

                conf = float(scores[i])
                if conf < CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box)
                track_id = ids[i]
                width = x2 - x1
                height = y2 - y1 

                if (x2 - x1) < 15 or (y2 - y1) < 15:
                    continue

                dx = int((x2 - x1) * PADDING)
                dy = int((y2 - y1) * PADDING)

                x1p = max(0, x1 - dx)
                y1p = max(0, y1 - dy)
                x2p = min(w, x2 + dx)
                y2p = min(h, y2 + dy)

                crop = frame[y1p:y2p, x1p:x2p]
                if crop.size == 0:
                    continue

                t_cls = time.time()

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                inp = transform(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = classifier(inp)
                    probs = torch.softmax(logits, dim=1)
                    conf_cls, pred = torch.max(probs, dim=1)

                pred = pred.item()
                conf_cls = conf_cls.item()

                classifier_times.append(time.time() - t_cls)

                label = idx_to_class[pred]

                if track_id not in track_memory:
                    track_memory[track_id] = {
                        "labels": deque(maxlen=5),
                        "locked_label": None,
                        "lock_conf": 0.0
                    }

                mem = track_memory[track_id]

                mem["labels"].append((label, conf_cls))

                labels = [l for l, c in mem["labels"]]
                counts = {l: labels.count(l) for l in set(labels)}

                majority_label = max(counts, key=counts.get)
                majority_count = counts[majority_label]

                majority_confs = [c for l, c in mem["labels"] if l == majority_label]
                avg_conf = sum(majority_confs) / max(1, len(majority_confs))

                if mem["locked_label"] is not None:

                    if mem["locked_label"] == "unknown":

                        if (
                            majority_label != "unknown" and
                            majority_count >= 2
                        ):
                            mem["locked_label"] = majority_label
                            mem["lock_conf"] = avg_conf

                        final_label = mem["locked_label"]

                    else:

                        if majority_label == "unknown":
                            final_label = mem["locked_label"]

                        else:
                            if (
                                majority_label != mem["locked_label"] and
                                majority_count >= 3 and
                                avg_conf > mem["lock_conf"] + 0.05
                            ):
                                mem["locked_label"] = majority_label
                                mem["lock_conf"] = avg_conf

                            final_label = mem["locked_label"]

                else:

                    if (
                        majority_label != "unknown" and
                        majority_count >= 3 and
                        avg_conf > 0.6
                    ):
                        mem["locked_label"] = majority_label
                        mem["lock_conf"] = avg_conf

                    final_label = majority_label

                if final_label == "advertisement" and (width < 50 or height < 50):
                    final_label = "unknown"

                new_boxes.append((x1, y1, x2, y2, final_label, conf))

                frame_preds.append({
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": conf,
                    "cls_conf": conf_cls,
                    "score": conf * conf_cls,
                    "cls_label": final_label,
                    "track_id": track_id
                })

        active_boxes = new_boxes

        all_predictions.append({
            "frame_id": frame_id,
            "detections": frame_preds
        })

    # DRAW
    t0 = time.time()
    for x1, y1, x2, y2, label, conf in active_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    tracker_times.append(time.time() - t0)

    current_time = time.time()
    fps = total_frames / (current_time - start_time + 1e-6)
    fps_list.append(fps)

    cpu_usages.append(psutil.cpu_percent(interval=None))
    ram = psutil.virtual_memory()
    ram_usages.append(ram.percent)
    cpu_temps.append(get_cpu_temp())

    cv2.putText(frame, f"FPS: {fps:.2f}", (20,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    total_frames += 1
    frame_id += 1

cap.release()
cv2.destroyAllWindows()
end_time = time.time()

# ===============================
# SAVE
# ===============================
with open(PRED_SAVE_PATH, "w") as f:
    json.dump(all_predictions, f, indent=4)

with open(EVAL_FRAMES_PATH, "w") as f:
    json.dump(evaluated_frames, f)

print("Saved predictions + evaluated frames")

# ===============================
# Metrics
# ===============================
total_time = end_time - start_time
fps = total_frames / total_time

avg_det = np.mean(detector_times) if detector_times else 0
avg_track = np.mean(tracker_times) if tracker_times else 0
avg_cls = np.mean(classifier_times) if classifier_times else 0

latencies = detector_times + tracker_times + classifier_times

valid_cpu_temps = [t for t in cpu_temps if t is not None]

results_dict = {
    "total_frames": total_frames,
    "total_runtime_sec": total_time,
    "fps": fps,
    "fps_std": float(np.std(fps_list)),
    "avg_detector_latency_ms": avg_det*1000,
    "avg_tracker_latency_ms": avg_track*1000,
    "avg_classifier_latency_ms": avg_cls*1000,
    "latency_std_ms": float(np.std(latencies)*1000),
    "cpu_usage_percent": float(np.mean(cpu_usages)),
    "cpu_temperature_c": float(np.mean(valid_cpu_temps)) if valid_cpu_temps else None,
    "max_cpu_temperature_c": float(np.max(valid_cpu_temps)) if valid_cpu_temps else None,
    "ram_usage_percent": float(np.mean(ram_usages)),
    "max_ram_usage_percent": float(np.max(ram_usages)),
    "num_classifications": len(classifier_times)
}

with open(RESULTS_SAVE_PATH, "w") as f:
    json.dump(results_dict, f, indent=4)

print("\n===============================")
print("Full Pipeline Results ")
print("===============================")
for k, v in results_dict.items():
    print(f"{k}: {v}")
print("===============================\n")