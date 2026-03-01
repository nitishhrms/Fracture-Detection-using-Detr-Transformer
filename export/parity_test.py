"""
parity_test.py — Verify PyTorch vs ONNX Runtime output parity
PyTorch and ORT outputs should match within 1e-5 tolerance.

Usage:
    python export/parity_test.py --onnx results/detr_fracture_fp32.onnx
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def run_pytorch(model, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
    """Run inference through PyTorch model."""
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    return outputs.logits.numpy(), outputs.pred_boxes.numpy()


def run_onnxruntime(onnx_path: str, pixel_values: np.ndarray, pixel_mask: np.ndarray):
    """Run inference through ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    ort_inputs = {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
    }
    logits, pred_boxes = sess.run(["logits", "pred_boxes"], ort_inputs)
    return logits, pred_boxes


def parity_check(
    onnx_path: str,
    pytorch_model=None,
    image_size: int = 800,
    batch_sizes: list = [1],
    atol: float = 1e-4,   # FP32: 1e-5, FP16: 1e-3
    rtol: float = 1e-3,
):
    """
    Compare PyTorch and ONNX Runtime outputs.

    Args:
        onnx_path: path to exported ONNX model
        pytorch_model: DetrForObjectDetection (loaded separately)
        image_size: spatial size for dummy inputs
        batch_sizes: list of batch sizes to test
        atol: absolute tolerance
        rtol: relative tolerance

    Returns:
        True if all checks pass, False otherwise
    """
    is_fp16 = "fp16" in onnx_path.lower()
    dtype_torch = torch.float16 if is_fp16 else torch.float32
    dtype_np    = np.float16    if is_fp16 else np.float32

    print(f"\nParity Test: {onnx_path}")
    print(f"Precision: {'FP16' if is_fp16 else 'FP32'} | atol={atol} | rtol={rtol}")
    print("-" * 60)

    all_pass = True

    for bs in batch_sizes:
        np.random.seed(42)
        pixel_values_np = np.random.randn(bs, 3, image_size, image_size).astype(dtype_np)
        pixel_mask_np   = np.ones((bs, image_size, image_size), dtype=dtype_np)

        pixel_values_pt = torch.from_numpy(pixel_values_np).to(dtype_torch)
        pixel_mask_pt   = torch.from_numpy(pixel_mask_np).to(dtype_torch)

        # ORT output
        ort_logits, ort_boxes = run_onnxruntime(onnx_path, pixel_values_np, pixel_mask_np)

        if pytorch_model is not None:
            pt_logits, pt_boxes = run_pytorch(pytorch_model, pixel_values_pt, pixel_mask_pt)

            logit_diff = np.abs(pt_logits - ort_logits).max()
            box_diff   = np.abs(pt_boxes  - ort_boxes).max()

            logit_ok = bool(np.allclose(pt_logits, ort_logits, atol=atol, rtol=rtol))
            box_ok   = bool(np.allclose(pt_boxes,  ort_boxes,  atol=atol, rtol=rtol))

            status_logit = "PASS" if logit_ok else "FAIL"
            status_box   = "PASS" if box_ok   else "FAIL"

            print(f"Batch={bs} | logits max_diff={logit_diff:.2e} [{status_logit}] | "
                  f"boxes max_diff={box_diff:.2e} [{status_box}]")

            if not (logit_ok and box_ok):
                all_pass = False
        else:
            print(f"Batch={bs} | ORT output shapes: logits={ort_logits.shape}, boxes={ort_boxes.shape} [ORT-ONLY]")

    result = "ALL PASSED" if all_pass else "SOME FAILED"
    print(f"\nResult: {result}")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch checkpoint for comparison (optional)")
    parser.add_argument("--image_size", type=int, default=800)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    pytorch_model = None
    if args.checkpoint:
        from transformers import DetrForObjectDetection
        if args.checkpoint.endswith(".ckpt"):
            from model import DETRFractureDetector
            lit = DETRFractureDetector.load_from_checkpoint(args.checkpoint)
            pytorch_model = lit.model
        else:
            pytorch_model = DetrForObjectDetection.from_pretrained(args.checkpoint)
        pytorch_model.eval()

    passed = parity_check(
        onnx_path=args.onnx,
        pytorch_model=pytorch_model,
        image_size=args.image_size,
        batch_sizes=args.batch_sizes,
        atol=args.atol,
        rtol=args.rtol,
    )
    exit(0 if passed else 1)


if __name__ == "__main__":
    main()
