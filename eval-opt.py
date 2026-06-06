import json
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
import argparse

# ===============================
# ARGPARSE
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ignore_classes",
    nargs="+",
    default=None,
    help="List of class names to ignore in evaluation"
)
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.5,
    help="Confidence threshold for filtering predictions (default: 0.5)"
)
parser.add_argument(
    "--pred_path",
    type=str,
    default="predictions.json"
)
parser.add_argument(
    "--xml_path",
    type=str,
    default="test-set/13-mp4.xml"
)
parser.add_argument(
    "--eval_frames_path",
    type=str,
    default="evaluated_frames.json"
)
args = parser.parse_args()

IGNORE_CLASSES  = set(args.ignore_classes) if args.ignore_classes else set()
CONF_THRESH     = args.conf_thresh
PRED_PATH       = args.pred_path
XML_PATH        = args.xml_path
EVAL_FRAMES_PATH = args.eval_frames_path

IOU_THRESHOLDS  = np.arange(0.5, 1.0, 0.05).round(2)

# ===============================
# LOAD
# ===============================
preds       = json.load(open(PRED_PATH))
eval_frames = set(json.load(open(EVAL_FRAMES_PATH)))

print(f"Loaded {len(preds)} prediction frames, {len(eval_frames)} eval frames")

# ===============================
# IoU
# ===============================
def iou(a, b):
    x1 = max(a[0], b[0]);  y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);  y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

# ===============================
# LOAD GT
# ===============================
def load_gt(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = defaultdict(list)

    for track in root.findall("track"):
        label = track.attrib["label"]
        if label in IGNORE_CLASSES:
            continue
        for box in track.findall("box"):
            # Skip occluded boxes if the attribute exists
            if box.attrib.get("outside", "0") == "1":
                continue
            f = int(box.attrib["frame"])
            if f not in eval_frames:
                continue
            gt[f].append({
                "bbox": [
                    float(box.attrib["xtl"]),
                    float(box.attrib["ytl"]),
                    float(box.attrib["xbr"]),
                    float(box.attrib["ybr"])
                ],
                "label": label
            })
    return gt

gt = load_gt(XML_PATH)

total_gt_boxes = sum(len(v) for v in gt.values())
print(f"Total GT boxes across eval frames: {total_gt_boxes}")

# ===============================
# MATCH — location + label (for detection mAP)
# ===============================
def match_with_label(gtb, prb, thr):
    """
    Match predictions to GT requiring both IoU >= thr AND same class label.
    Used for detection mAP (joint localisation + classification).
    """
    used = set()
    matches = []
    for i, p in enumerate(prb):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gtb):
            if j in used:
                continue
            if p["cls_label"] != g["label"]:
                continue
            val = iou(p["bbox"], g["bbox"])
            if val > best_iou:
                best_iou, best_j = val, j
        if best_iou >= thr:
            used.add(best_j)
            matches.append((i, best_j))
    return matches

# ===============================
# MATCH — location only (for classification metrics)
# ===============================
def match_location_only(gtb, prb, thr):
    """
    Match predictions to GT by IoU >= thr only, ignoring class labels.
    Used for classification metrics so misclassified-but-localised
    detections are properly counted as TP_loc with wrong label.
    """
    used = set()
    matches = []
    for i, p in enumerate(prb):
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gtb):
            if j in used:
                continue
            val = iou(p["bbox"], g["bbox"])
            if val > best_iou:
                best_iou, best_j = val, j
        if best_iou >= thr:
            used.add(best_j)
            matches.append((i, best_j))
    return matches

# ===============================
# DETECTION mAP  (proper PR-curve area)
# ===============================
def compute_map():
    """
    Proper mAP:
      - For each IoU threshold, collect ALL detections across ALL frames
        sorted by descending score.
      - Walk down the ranked list, marking each detection TP or FP.
      - Compute area under the PR curve (101-point interpolation).
      - Average AP across IoU thresholds for mAP@50:95.

    NOTE: matching uses label equality, so this measures joint
    detection + classification performance (standard for COCO-style mAP).
    """
    all_aps = []

    for thr in IOU_THRESHOLDS:
        ranked = []       # (score, is_tp)
        total_gt_count = 0

        for p in preds:
            f = p["frame_id"]
            if f not in eval_frames:
                continue

            gtb = gt.get(f, [])
            total_gt_count += len(gtb)

            prb = [
                d for d in p["detections"]
                if d.get("score", 0.0) >= CONF_THRESH
                and d.get("cls_label") not in IGNORE_CLASSES
            ]
            prb = sorted(prb, key=lambda x: x.get("score", 0.0), reverse=True)

            matches = match_with_label(gtb, prb, thr)
            matched_pred_indices = {pi for pi, _ in matches}

            for i, d in enumerate(prb):
                ranked.append((d.get("score", 0.0), i in matched_pred_indices))

        if total_gt_count == 0 or len(ranked) == 0:
            all_aps.append(0.0)
            continue

        # Sort all detections by descending confidence
        ranked.sort(key=lambda x: -x[0])

        tp_cum, fp_cum = 0, 0
        precisions, recalls = [], []

        for _, is_tp in ranked:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            precisions.append(tp_cum / (tp_cum + fp_cum))
            recalls.append(tp_cum / total_gt_count)

        # 101-point interpolation (COCO standard)
        ap = 0.0
        for r_thresh in np.linspace(0, 1, 101):
            prec_at_r = [p for p, r in zip(precisions, recalls) if r >= r_thresh]
            ap += (max(prec_at_r) if prec_at_r else 0.0)
        ap /= 101.0
        all_aps.append(ap)

    map5095 = float(np.mean(all_aps))
    map50   = float(all_aps[0])
    return map5095, map50, all_aps

# ===============================
# CLASSIFICATION METRICS
# ===============================
def compute_classification():
    cls_tp      = defaultdict(int)
    cls_fp      = defaultdict(int)
    cls_fn      = defaultdict(int)
    cls_support = defaultdict(int)
    total_gt_count = 0

    # confusion[true_label][pred_label] = count
    confusion = defaultdict(lambda: defaultdict(int))

    for p in preds:
        f = p["frame_id"]
        if f not in eval_frames:
            continue

        gtb = [g for g in gt.get(f, []) if g["label"] not in IGNORE_CLASSES]
        total_gt_count += len(gtb)

        prb = [
            d for d in p["detections"]
            if d.get("score", 0.0) >= CONF_THRESH
            and d.get("cls_label") not in IGNORE_CLASSES
        ]
        prb = sorted(prb, key=lambda x: x.get("score", 0.0), reverse=True)

        # Count support from GT side
        for g in gtb:
            cls_support[g["label"]] += 1

        # Match by location only — then check labels separately
        matches = match_location_only(gtb, prb, 0.5)

        matched_gt  = set()
        matched_pr  = set()

        for pi, gi in matches:
            gt_label   = gtb[gi]["label"]
            pred_label = prb[pi]["cls_label"]

            matched_gt.add(gi)
            matched_pr.add(pi)

            confusion[gt_label][pred_label] += 1

            if gt_label == pred_label:
                cls_tp[gt_label] += 1
            else:
                # Correct location, wrong label
                cls_fp[pred_label] += 1
                cls_fn[gt_label]   += 1

        # Unmatched GT → missed detections (FN)
        for gi, g in enumerate(gtb):
            if gi not in matched_gt:
                cls_fn[g["label"]] += 1
                confusion[g["label"]]["__missed__"] += 1

        # Unmatched predictions → false alarms (FP)
        for pi, pbox in enumerate(prb):
            if pi not in matched_pr:
                cls_fp[pbox["cls_label"]] += 1
                confusion["__ghost__"][pbox["cls_label"]] += 1

    # -------------------------------------------------------
    # Per-class metrics
    # -------------------------------------------------------
    all_labels = sorted(
        set(list(cls_tp.keys()) + list(cls_fp.keys()) + list(cls_fn.keys()))
    )

    per_class_prec  = []
    per_class_rec   = []
    per_class_f1    = []

    print("\nClassification Metrics (per class):")
    print("{:>25} {:>10} {:>10} {:>10} {:>10}".format(
        "CLASS", "SUPPORT", "PREC", "RECALL", "F1"
    ))
    print("-" * 67)

    for cls in all_labels:
        denom_prec = cls_tp[cls] + cls_fp[cls]
        denom_rec  = cls_tp[cls] + cls_fn[cls]

        prec = cls_tp[cls] / denom_prec if denom_prec > 0 else float("nan")
        rec  = cls_tp[cls] / denom_rec  if denom_rec  > 0 else float("nan")

        if not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = float("nan")

        support_str = str(cls_support.get(cls, 0))
        prec_str    = f"{prec:.3f}" if not np.isnan(prec) else "  N/A"
        rec_str     = f"{rec:.3f}"  if not np.isnan(rec)  else "  N/A"
        f1_str      = f"{f1:.3f}"   if not np.isnan(f1)   else "  N/A"

        print("{:>25} {:>10} {:>10} {:>10} {:>10}".format(
            cls, support_str, prec_str, rec_str, f1_str
        ))

        per_class_prec.append(prec)
        per_class_rec.append(rec)
        per_class_f1.append(f1)

    # -------------------------------------------------------
    # Macro average (unweighted, NaN-safe)
    # -------------------------------------------------------
    macro_prec = float(np.nanmean(per_class_prec))
    macro_rec  = float(np.nanmean(per_class_rec))
    macro_f1   = float(np.nanmean(per_class_f1))

    # -------------------------------------------------------
    # Weighted average (weighted by support)
    # -------------------------------------------------------
    total_support = sum(cls_support.values())
    weighted_prec, weighted_rec, weighted_f1 = 0.0, 0.0, 0.0

    for cls, prec, rec, f1 in zip(all_labels, per_class_prec, per_class_rec, per_class_f1):
        w = cls_support.get(cls, 0) / total_support if total_support > 0 else 0
        if not np.isnan(prec): weighted_prec += w * prec
        if not np.isnan(rec):  weighted_rec  += w * rec
        if not np.isnan(f1):   weighted_f1   += w * f1

    # -------------------------------------------------------
    # Accuracy = TP / total_GT_boxes
    # (out of all GT objects, how many were correctly localised AND labelled)
    # -------------------------------------------------------
    total_tp = sum(cls_tp.values())
    accuracy = total_tp / total_gt_count if total_gt_count > 0 else 0.0

    # -------------------------------------------------------
    # Confusion matrix (only real classes, not __missed__/__ghost__)
    # -------------------------------------------------------
    real_classes = sorted(cls_support.keys())

    print("\nConfusion Matrix  (rows = GT label, cols = Predicted label):")
    col_width = max(12, max((len(c) for c in real_classes), default=8) + 2)
    header = " " * 26 + "".join(c[:col_width-1].rjust(col_width) for c in real_classes)
    header += "  | MISSED"
    print(header)
    print("-" * len(header))

    for true_cls in real_classes:
        row = true_cls.rjust(25) + " "
        for pred_cls in real_classes:
            row += str(confusion[true_cls].get(pred_cls, 0)).rjust(col_width)
        row += "  | " + str(confusion[true_cls].get("__missed__", 0))
        print(row)

    # Ghost row (predictions with no matching GT box)
    ghost_row = "  (false alarms)          "
    for pred_cls in real_classes:
        ghost_row += str(confusion["__ghost__"].get(pred_cls, 0)).rjust(col_width)
    print(ghost_row)

    return {
        "macro_prec":    macro_prec,
        "macro_rec":     macro_rec,
        "macro_f1":      macro_f1,
        "weighted_prec": weighted_prec,
        "weighted_rec":  weighted_rec,
        "weighted_f1":   weighted_f1,
        "accuracy":      accuracy,
    }

# ===============================
# RUN
# ===============================
map5095, map50, per_iou_aps = compute_map()
cls_metrics = compute_classification()

print("\n" + "=" * 40)
print("Detection Metrics  (joint loc + cls)")
print("=" * 40)
print(f"  mAP@50:          {map50:.4f}")
print(f"  mAP@50:95:       {map5095:.4f}")
print("\n  AP per IoU threshold:")
for thr, ap in zip(IOU_THRESHOLDS, per_iou_aps):
    print(f"    IoU={thr:.2f}  →  AP={ap:.4f}")

print("\n" + "=" * 40)
print("Classification Metrics")
print("=" * 40)
print(f"  Accuracy (TP / total GT):   {cls_metrics['accuracy']:.4f}")
print()
print(f"  Macro   Prec: {cls_metrics['macro_prec']:.4f}   "
      f"Rec: {cls_metrics['macro_rec']:.4f}   "
      f"F1: {cls_metrics['macro_f1']:.4f}")
print(f"  Weighted Prec: {cls_metrics['weighted_prec']:.4f}   "
      f"Rec: {cls_metrics['weighted_rec']:.4f}   "
      f"F1: {cls_metrics['weighted_f1']:.4f}")
print("=" * 40 + "\n")
