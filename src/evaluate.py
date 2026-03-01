"""
evaluate.py — Clinical metrics for fracture detection
Computes: Sensitivity, Specificity, AUC, AP, F1
Usage:
    python src/evaluate.py --checkpoint results/checkpoints/best.ckpt \
                           --dataset_root "data/bone fracture.v2-release.coco"
"""
import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import DetrImageProcessor

# Optional sklearn for AUC
try:
    from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] sklearn not installed. AUC disabled. Run: pip install scikit-learn")


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_model(model, dataloader, image_processor, iou_threshold=0.5, score_threshold=0.5, device="cpu"):
    """
    Run inference on dataloader and compute clinical metrics.

    Returns dict with keys:
        sensitivity, specificity, precision, f1, auc, ap, tp, fp, fn, tn
    """
    model.eval()
    model.to(device)

    all_scores = []   # detection confidence scores
    all_labels = []   # ground truth: 1=fracture present, 0=no fracture
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            targets = batch["labels"]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # Post-process
            target_sizes = torch.stack([torch.tensor(t["orig_size"]) for t in targets])
            results = image_processor.post_process_object_detection(
                outputs, threshold=score_threshold, target_sizes=target_sizes
            )

            for result, target in zip(results, targets):
                pred_boxes = result["boxes"].cpu().numpy()
                pred_scores = result["scores"].cpu().numpy()
                gt_boxes = target["boxes"].cpu().numpy()

                has_gt = len(gt_boxes) > 0
                has_pred = len(pred_boxes) > 0

                # Image-level score = max detection confidence
                img_score = float(pred_scores.max()) if has_pred else 0.0
                all_scores.append(img_score)
                all_labels.append(1 if has_gt else 0)

                if has_gt and has_pred:
                    # Check if any pred matches a GT box
                    matched = False
                    for pb in pred_boxes:
                        for gb in gt_boxes:
                            if compute_iou(pb, gb) >= iou_threshold:
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        tp += 1
                    else:
                        fp += 1
                        fn += 1
                elif has_gt and not has_pred:
                    fn += 1
                elif not has_gt and has_pred:
                    fp += 1
                else:
                    tn += 1

    # Clinical metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / true positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # true negative rate
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    auc = 0.0
    ap  = 0.0
    if SKLEARN_AVAILABLE and len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
        ap  = average_precision_score(all_labels, all_scores)

    metrics = {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision":   round(precision, 4),
        "f1":          round(f1, 4),
        "auc":         round(auc, 4),
        "ap":          round(ap, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    return metrics


def print_metrics_table(metrics: dict):
    """Pretty-print clinical metrics table."""
    print("\n" + "="*50)
    print("  FRACTURE DETECTION — CLINICAL METRICS")
    print("="*50)
    print(f"  Sensitivity (Recall)  : {metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity           : {metrics['specificity']*100:.2f}%")
    print(f"  Precision             : {metrics['precision']*100:.2f}%")
    print(f"  F1 Score              : {metrics['f1']*100:.2f}%")
    print(f"  AUC-ROC               : {metrics['auc']:.4f}")
    print(f"  Average Precision (AP): {metrics['ap']:.4f}")
    print("-"*50)
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str,
                        default="data/bone fracture.v2-release.coco")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="results/metrics.json")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset import build_dataloaders
    from model import DETRFractureDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    loaders, train_ds, _, _ = build_dataloaders(args.dataset_root, image_processor)

    categories = train_ds.coco.cats
    id2label = {k: v["name"] for k, v in categories.items()}
    label2id = {v: k for k, v in id2label.items()}

    model = DETRFractureDetector.load_from_checkpoint(
        args.checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    loader = loaders[args.split]
    metrics = evaluate_model(model.model, loader, image_processor,
                             iou_threshold=args.iou_threshold,
                             score_threshold=args.score_threshold,
                             device=device)

    print_metrics_table(metrics)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
