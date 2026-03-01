"""
benchmark.py — Latency benchmarking: PyTorch vs ONNX Runtime
Benchmarks batch sizes 1/4/8/16 at FP32 and FP16.
Outputs results/benchmark_results.csv and a chart.

Usage:
    python export/benchmark.py --onnx_fp32 results/detr_fracture_fp32.onnx \
                                --onnx_fp16 results/detr_fracture_fp16.onnx
"""
import argparse
import os
import time
import sys
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

WARMUP_RUNS = 5
BENCH_RUNS = 20
IMAGE_SIZE = 800


def bench_pytorch(model, batch_size: int, dtype=torch.float32, device="cpu", runs=BENCH_RUNS):
    """Benchmark PyTorch inference latency in milliseconds."""
    model.eval().to(device)
    if dtype == torch.float16:
        model = model.half()

    pv = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype, device=device)
    pm = torch.ones(batch_size, IMAGE_SIZE, IMAGE_SIZE, dtype=dtype, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(pixel_values=pv, pixel_mask=pm)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(pixel_values=pv, pixel_mask=pm)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)


def bench_onnxruntime(onnx_path: str, batch_size: int, dtype=np.float32, runs=BENCH_RUNS):
    """Benchmark ONNX Runtime inference latency in milliseconds."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    pv = np.random.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE).astype(dtype)
    pm = np.ones((batch_size, IMAGE_SIZE, IMAGE_SIZE), dtype=dtype)
    inputs = {"pixel_values": pv, "pixel_mask": pm}

    # Warmup
    for _ in range(WARMUP_RUNS):
        sess.run(None, inputs)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)


def run_benchmark(pytorch_model=None, onnx_fp32=None, onnx_fp16=None,
                  batch_sizes=(1, 4, 8, 16), output_dir="results"):
    """
    Run full benchmark across batch sizes and precisions.
    Saves CSV + chart to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    print(f"\nBenchmark | Device: {device.upper()} | Runs: {BENCH_RUNS} | Image: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("=" * 75)
    print(f"{'Backend':<20} {'Precision':<10} {'Batch':<8} {'Mean (ms)':<12} {'Std (ms)':<10}")
    print("-" * 75)

    for bs in batch_sizes:
        # PyTorch FP32
        if pytorch_model is not None:
            mean, std = bench_pytorch(pytorch_model, bs, torch.float32, device)
            print(f"{'PyTorch':<20} {'FP32':<10} {bs:<8} {mean:<12.2f} {std:<10.2f}")
            rows.append({"backend": "PyTorch", "precision": "FP32", "batch": bs,
                         "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

        # PyTorch FP16
        if pytorch_model is not None and device == "cuda":
            mean, std = bench_pytorch(pytorch_model, bs, torch.float16, device)
            print(f"{'PyTorch':<20} {'FP16':<10} {bs:<8} {mean:<12.2f} {std:<10.2f}")
            rows.append({"backend": "PyTorch", "precision": "FP16", "batch": bs,
                         "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

        # ORT FP32
        if onnx_fp32 and os.path.exists(onnx_fp32):
            mean, std = bench_onnxruntime(onnx_fp32, bs, np.float32)
            print(f"{'ORT':<20} {'FP32':<10} {bs:<8} {mean:<12.2f} {std:<10.2f}")
            rows.append({"backend": "ORT", "precision": "FP32", "batch": bs,
                         "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

        # ORT FP16
        if onnx_fp16 and os.path.exists(onnx_fp16):
            mean, std = bench_onnxruntime(onnx_fp16, bs, np.float16)
            print(f"{'ORT':<20} {'FP16':<10} {bs:<8} {mean:<12.2f} {std:<10.2f}")
            rows.append({"backend": "ORT", "precision": "FP16", "batch": bs,
                         "mean_ms": round(mean, 2), "std_ms": round(std, 2)})

    print("=" * 75)

    # Save CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "precision", "batch", "mean_ms", "std_ms"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")

    # Plot
    _plot_benchmark(rows, output_dir)
    return rows


def _plot_benchmark(rows, output_dir):
    """Generate latency comparison chart."""
    if not rows:
        return

    import pandas as pd
    try:
        df = pd.DataFrame(rows)
    except ImportError:
        print("[WARN] pandas not installed — skipping chart. Run: pip install pandas")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: batch=1 comparison bar chart
    bs1 = df[df["batch"] == 1].copy()
    labels = bs1["backend"] + " " + bs1["precision"]
    axes[0].bar(labels, bs1["mean_ms"], color=["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"])
    axes[0].set_title("Latency at Batch=1 (ms)")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xlabel("Backend + Precision")
    axes[0].tick_params(axis="x", rotation=15)

    # Right: latency vs batch size line chart
    for (backend, precision), group in df.groupby(["backend", "precision"]):
        axes[1].plot(group["batch"], group["mean_ms"],
                     marker="o", label=f"{backend} {precision}")
    axes[1].set_title("Latency vs Batch Size")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xlabel("Batch Size")
    axes[1].legend()
    axes[1].set_xticks([1, 4, 8, 16])

    plt.suptitle("DETR Fracture Detection — Inference Benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "benchmark_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart saved: {chart_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_fp32", type=str, default=None)
    parser.add_argument("--onnx_fp16", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="PyTorch checkpoint or HuggingFace model ID")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--output_dir", type=str, default="results")
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

    run_benchmark(
        pytorch_model=pytorch_model,
        onnx_fp32=args.onnx_fp32,
        onnx_fp16=args.onnx_fp16,
        batch_sizes=args.batch_sizes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
