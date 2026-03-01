"""
predict.py — Run fracture detection inference on a single image or folder
Usage:
    python src/predict.py --image path/to/xray.jpg
    python src/predict.py --image path/to/xray.dcm --checkpoint results/checkpoints/best.ckpt
    python src/predict.py --folder data/test_images/ --output results/predictions/
"""
import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection


def predict_single(image_path: str, model, image_processor,
                   score_threshold: float = 0.5,
                   id2label: dict = None,
                   device: str = "cpu"):
    """
    Run inference on a single image (.jpg / .png / .dcm).
    Returns PIL image with annotated boxes, and raw results dict.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset import load_image

    pil_image = load_image(image_path)
    encoding = image_processor(images=pil_image, return_tensors="pt")
    pixel_values = encoding["pixel_values"].to(device)
    pixel_mask = encoding.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    target_sizes = torch.tensor([[pil_image.height, pil_image.width]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]

    # Draw boxes on image
    img_np = np.array(pil_image.convert("RGB"))
    annotated = img_np.copy()

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        label_name = id2label.get(label.item(), str(label.item())) if id2label else "fracture"
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label_name}: {score:.2f}"
        cv2.putText(annotated, text, (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if len(results["boxes"]) == 0:
        cv2.putText(annotated, "No fracture detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return Image.fromarray(annotated), results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--folder", type=str, default=None, help="Folder of images")
    parser.add_argument("--checkpoint", type=str, default="facebook/detr-resnet-50")
    parser.add_argument("--output", type=str, default="results/predictions")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(args.checkpoint).to(device)

    os.makedirs(args.output, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".dcm"}

    if args.image:
        paths = [args.image]
    elif args.folder:
        paths = [os.path.join(args.folder, f) for f in os.listdir(args.folder)
                 if os.path.splitext(f)[1].lower() in supported]
    else:
        print("Provide --image or --folder")
        return

    for path in paths:
        annotated_img, results = predict_single(
            path, model, image_processor,
            score_threshold=args.score_threshold,
            device=device,
        )
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output, f"{base}_pred.jpg")
        annotated_img.save(out_path)
        n = len(results["boxes"])
        print(f"{os.path.basename(path)}: {n} detection(s) → {out_path}")


if __name__ == "__main__":
    main()
