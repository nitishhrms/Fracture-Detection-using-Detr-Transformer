"""
export_onnx.py — Export DETR PyTorch model to ONNX format
Usage:
    python export/export_onnx.py --checkpoint results/checkpoints/best.ckpt \
                                  --output results/detr_fracture.onnx
"""
import argparse
import os
import sys
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class DETRONNXWrapper(torch.nn.Module):
    """
    Wraps DETR to produce flat outputs compatible with ONNX export.
    ONNX does not support dict outputs natively.
    """

    def __init__(self, model: DetrForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        # Return logits and pred_boxes as flat tensors
        return outputs.logits, outputs.pred_boxes


def export_to_onnx(
    model: DetrForObjectDetection,
    output_path: str,
    image_size: int = 800,
    opset_version: int = 14,
    use_fp16: bool = False,
):
    """
    Export DETR model to ONNX.

    Args:
        model: DETR model (PyTorch)
        output_path: path to save .onnx file
        image_size: input image size for dummy input
        opset_version: ONNX opset (14 recommended for transformers)
        use_fp16: whether to export in FP16

    Returns:
        output_path
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model.eval()
    if use_fp16:
        model = model.half()

    wrapper = DETRONNXWrapper(model)
    wrapper.eval()

    dtype = torch.float16 if use_fp16 else torch.float32
    dummy_pixel_values = torch.zeros(1, 3, image_size, image_size, dtype=dtype)
    dummy_pixel_mask = torch.ones(1, image_size, image_size, dtype=dtype)

    print(f"Exporting to ONNX (opset={opset_version}, fp16={use_fp16})...")
    torch.onnx.export(
        wrapper,
        (dummy_pixel_values, dummy_pixel_mask),
        output_path,
        opset_version=opset_version,
        input_names=["pixel_values", "pixel_mask"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch_size", 2: "height", 3: "width"},
            "pixel_mask":   {0: "batch_size", 1: "height", 2: "width"},
            "logits":       {0: "batch_size"},
            "pred_boxes":   {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX model saved: {output_path}  ({size_mb:.1f} MB)")
    return output_path


def verify_onnx(onnx_path: str):
    """Basic ONNX model validation using onnx library."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("ONNX model check passed.")
    except ImportError:
        print("[WARN] onnx not installed — skipping validation. Run: pip install onnx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="facebook/detr-resnet-50",
                        help="PyTorch checkpoint or HuggingFace model ID")
    parser.add_argument("--output", type=str, default="results/detr_fracture_fp32.onnx")
    parser.add_argument("--image_size", type=int, default=800)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--fp16", action="store_true", help="Export FP16 model")
    args = parser.parse_args()

    # Load model
    if args.checkpoint.endswith(".ckpt"):
        from model import DETRFractureDetector
        lit_model = DETRFractureDetector.load_from_checkpoint(args.checkpoint)
        model = lit_model.model
    else:
        model = DetrForObjectDetection.from_pretrained(args.checkpoint)

    model.eval()

    # Export
    out_path = args.output.replace(".onnx", "_fp16.onnx") if args.fp16 else args.output
    export_to_onnx(model, out_path, args.image_size, args.opset, args.fp16)
    verify_onnx(out_path)


if __name__ == "__main__":
    main()
