"""
gradcam.py — GradCAM visualization for DETR fracture detection
Generates heatmaps showing which regions the model focuses on.
Usage:
    python src/gradcam.py --image path/to/xray.jpg --checkpoint results/checkpoints/best.ckpt
    python src/gradcam.py --image path/to/xray.dcm --checkpoint results/checkpoints/best.ckpt
"""
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from transformers import DetrImageProcessor, DetrForObjectDetection


class DETRGradCAM:
    """
    GradCAM for DETR — hooks into the last ResNet backbone layer.
    Produces a saliency heatmap over the input X-ray.
    """

    def __init__(self, model: DetrForObjectDetection, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self._gradients = None
        self._activations = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Hook into the last layer of the DETR backbone (layer4 of ResNet)."""
        # DETR backbone: model.model.backbone.conv_encoder.model.layer4
        target_layer = self.model.model.backbone.conv_encoder.model.layer4

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor, query_idx: int = 0):
        """
        Generate GradCAM heatmap.

        Args:
            pixel_values: preprocessed image tensor [1, 3, H, W]
            pixel_mask: attention mask [1, H, W]
            query_idx: DETR object query index to backprop from (default=0 = top detection)

        Returns:
            heatmap: numpy array [H, W] with values in [0, 1]
        """
        self.model.eval()
        pixel_values = pixel_values.to(self.device).requires_grad_(False)
        pixel_mask = pixel_mask.to(self.device)

        # Enable gradients on pixel_values for backprop through backbone
        pixel_values.requires_grad_(True)

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # Use the logit of the top query as the scalar to backprop
        # outputs.logits shape: [1, num_queries, num_classes+1]
        logits = outputs.logits  # [1, 100, num_classes+1]
        pred_scores = logits[0].softmax(-1)[:, :-1].max(-1).values  # [100]

        # Sort queries by confidence, pick top-k
        top_idx = pred_scores.argsort(descending=True)[query_idx]
        target_score = pred_scores[top_idx]

        self.model.zero_grad()
        target_score.backward(retain_graph=True)

        if self._gradients is None or self._activations is None:
            raise RuntimeError("Hooks did not fire. Check model architecture.")

        # Pool gradients across channels
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self._activations).sum(dim=1).squeeze(0)  # [h, w]
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy()

    def overlay(self, original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5):
        """
        Overlay GradCAM heatmap on the original image.

        Args:
            original_image: RGB numpy array [H, W, 3]
            heatmap: float array [h, w] in [0, 1]
            alpha: blending factor

        Returns:
            overlaid image as numpy array [H, W, 3]
        """
        H, W = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (W, H))
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]  # [H, W, 3] RGB
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        original_float = original_image.astype(np.float32)
        heatmap_float = heatmap_colored.astype(np.float32)
        overlaid = (1 - alpha) * original_float + alpha * heatmap_float
        return overlaid.astype(np.uint8)


def run_gradcam(image_path: str, model_or_checkpoint, image_processor,
                output_dir: str = "results", score_threshold: float = 0.3,
                id2label: dict = None):
    """
    End-to-end GradCAM pipeline for a single image.
    Saves: original + heatmap + overlay + detection boxes.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset import load_image

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load image
    pil_image = load_image(image_path)
    original_np = np.array(pil_image)

    # Preprocess
    encoding = image_processor(images=pil_image, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    pixel_mask = encoding.get("pixel_mask", torch.ones(1, pixel_values.shape[2], pixel_values.shape[3]))

    # Load model
    if isinstance(model_or_checkpoint, str):
        model = DetrForObjectDetection.from_pretrained(model_or_checkpoint)
    else:
        model = model_or_checkpoint

    model.eval()
    gradcam = DETRGradCAM(model, device=device)

    # Generate heatmap
    heatmap = gradcam.generate(pixel_values, pixel_mask, query_idx=0)
    overlaid = gradcam.overlay(original_np, heatmap, alpha=0.45)

    # Run inference for bounding boxes
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values.to(device),
            pixel_mask=pixel_mask.to(device)
        )
    target_sizes = torch.tensor([[pil_image.height, pil_image.width]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=score_threshold, target_sizes=target_sizes
    )[0]

    # Draw detections on overlaid image
    annotated = overlaid.copy()
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        label_name = id2label.get(label.item(), str(label.item())) if id2label else str(label.item())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"{label_name}: {score:.2f}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save outputs
    base = os.path.splitext(os.path.basename(image_path))[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_np)
    axes[0].set_title("Original X-Ray")
    axes[0].axis("off")

    axes[1].imshow(original_np, cmap="gray")
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5,
                   extent=[0, original_np.shape[1], original_np.shape[0], 0])
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(annotated)
    axes[2].set_title("Detections + GradCAM")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{base}_gradcam.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    gradcam.remove_hooks()
    print(f"GradCAM saved: {out_path}")
    print(f"Detections: {len(results['boxes'])} fractures found")
    return out_path, heatmap, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray (.jpg/.png/.dcm)")
    parser.add_argument("--checkpoint", type=str, default="facebook/detr-resnet-50",
                        help="Model checkpoint path or HuggingFace model ID")
    parser.add_argument("--output_dir", type=str, default="results/gradcam")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    args = parser.parse_args()

    image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained(args.checkpoint)

    run_gradcam(
        image_path=args.image,
        model_or_checkpoint=model,
        image_processor=image_processor,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()
