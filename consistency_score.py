import json
import argparse
from collections import defaultdict

# ===============================
# ARGPARSE
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, default="ablation/no_memory_predictions.json",
                    help="Path to predictions JSON")
parser.add_argument("--conf_thresh", type=float, default=0.5,
                    help="Confidence threshold")
parser.add_argument("--min_track_len", type=int, default=5,
                    help="Minimum track length to consider")

args = parser.parse_args()

PRED_PATH = args.pred_path
CONF_THRESH = args.conf_thresh
MIN_TRACK_LEN = args.min_track_len

# ===============================
# LOAD
# ===============================
with open(PRED_PATH, "r") as f:
    preds = json.load(f)

# ===============================
# BUILD TRACKS
# ===============================
tracks = defaultdict(list)

for frame in preds:
    frame_id = frame["frame_id"]

    for det in frame.get("detections", []):
        score = det.get("score", 0.0)

        if score < CONF_THRESH:
            continue

        track_id = det.get("track_id", None)
        label = det.get("cls_label", None)

        if track_id is None or track_id == -1:
            continue

        tracks[track_id].append((frame_id, label))

# ===============================
# COMPUTE METRICS
# ===============================
total_same = 0
total_transitions = 0
valid_tracks = 0

for tid, seq in tracks.items():

    # sort by frame
    seq = sorted(seq, key=lambda x: x[0])

    if len(seq) < MIN_TRACK_LEN:
        continue

    valid_tracks += 1

    for i in range(1, len(seq)):
        prev_label = seq[i-1][1]
        curr_label = seq[i][1]

        total_transitions += 1

        if prev_label == curr_label:
            total_same += 1

# ===============================
# RESULTS
# ===============================
if total_transitions == 0:
    consistency = 0.0
    flip_rate = 0.0
else:
    consistency = total_same / total_transitions
    flip_rate = 1.0 - consistency

print("\n===============================")
print("Temporal Stability Metrics")
print("===============================")
print(f"Valid tracks: {valid_tracks}")
print(f"Total transitions: {total_transitions}")
print(f"Consistency: {consistency:.4f} ({consistency*100:.2f}%)")
print(f"Flip Rate:  {flip_rate:.4f} ({flip_rate*100:.2f}%)")
print("===============================\n")