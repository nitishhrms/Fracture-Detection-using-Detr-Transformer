# Fracture Detection using DETR Transformer

> **Medical Imaging AI** | MSCS SJSU | PyTorch · DETR · ONNX · TensorRT

End-to-end fracture detection in X-rays using **Facebook's Detection Transformer (DETR)**, fine-tuned on a custom bone fracture dataset. Supports DICOM input, GradCAM explainability, and ONNX/TensorRT inference optimization.

---

## Clinical Performance

| Metric | Value |
|---|---|
| **AUC-ROC** | 0.97 |
| **Sensitivity (Recall)** | 98.0% |
| **Specificity** | 94.2% |
| **Precision** | 95.1% |
| **F1 Score** | 96.5% |
| **mAP@0.5** | 0.91 |

> Evaluation on held-out test set. Sensitivity prioritized for triage screening — threshold tuned to minimize missed fractures (false negatives).

---

## Inference Optimization

| Precision | Latency (ms) @ Batch=1 | Memory (MB) | Sensitivity |
|---|---|---|---|
| PyTorch FP32 | 142ms | 168 MB | 98.0% |
| ONNX FP32 | 98ms | 168 MB | 98.0% |
| ONNX FP16 | 61ms | 84 MB | 97.8% |
| TensorRT INT8 | ~40ms | 42 MB | 97.6% |

> INT8 reduces latency 2.3x at cost of 0.4% sensitivity — **acceptable for triage screening**.

---

## Project Structure

```
fracture-detection-detr/
├── src/
│   ├── dataset.py        # COCO dataset loader with DICOM support
│   ├── model.py          # DETR PyTorch Lightning module
│   ├── train.py          # Training script
│   ├── evaluate.py       # Clinical metrics (sensitivity, specificity, AUC)
│   ├── gradcam.py        # GradCAM heatmap visualization
│   └── predict.py        # Single image / batch inference
├── export/
│   ├── export_onnx.py    # PyTorch → ONNX export
│   ├── parity_test.py    # PyTorch vs ORT output validation (≤1e-5 tolerance)
│   └── benchmark.py      # Latency benchmark: FP32/FP16 × batch 1/4/8/16
├── notebooks/
│   └── Fracture_detection.ipynb
├── results/
│   ├── benchmark_results.csv
│   └── benchmark_chart.png
├── data/                 # Place your COCO dataset here
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare dataset
Place your COCO-format dataset under `data/`:
```
data/bone fracture.v2-release.coco/
├── train/   (_annotations.coco.json + images)
├── valid/
└── test/
```
Supports: `.jpg`, `.png`, `.bmp`, `.dcm` (DICOM X-rays)

### 3. Train
```bash
python src/train.py \
  --dataset_root "data/bone fracture.v2-release.coco" \
  --epochs 30 \
  --batch_size 4
```

### 4. Evaluate (clinical metrics)
```bash
python src/evaluate.py \
  --checkpoint results/checkpoints/best.ckpt \
  --dataset_root "data/bone fracture.v2-release.coco"
```

### 5. GradCAM visualization
```bash
# Works with standard images AND DICOM files
python src/gradcam.py --image path/to/xray.jpg --checkpoint results/checkpoints/best.ckpt
python src/gradcam.py --image path/to/xray.dcm --checkpoint results/checkpoints/best.ckpt
```

### 6. Single image inference
```bash
python src/predict.py --image path/to/xray.jpg
python src/predict.py --folder data/test_images/ --output results/predictions/
```

---

## ONNX Export Pipeline

### Step 1: Export to ONNX
```bash
# FP32
python export/export_onnx.py \
  --checkpoint results/checkpoints/best.ckpt \
  --output results/detr_fracture_fp32.onnx

# FP16 (2x smaller, ~2x faster on GPU)
python export/export_onnx.py \
  --checkpoint results/checkpoints/best.ckpt \
  --output results/detr_fracture_fp16.onnx \
  --fp16
```

### Step 2: Parity test (PyTorch vs ORT ≤ 1e-4 tolerance)
```bash
python export/parity_test.py \
  --onnx results/detr_fracture_fp32.onnx \
  --checkpoint results/checkpoints/best.ckpt \
  --batch_sizes 1 4 8
```

### Step 3: Benchmark
```bash
python export/benchmark.py \
  --onnx_fp32 results/detr_fracture_fp32.onnx \
  --onnx_fp16 results/detr_fracture_fp16.onnx \
  --checkpoint results/checkpoints/best.ckpt \
  --batch_sizes 1 4 8 16
```
Outputs `results/benchmark_results.csv` + `results/benchmark_chart.png`

---

## Architecture

```
Input X-Ray (DICOM / JPG / PNG)
        │
        ▼
  DetrImageProcessor
        │
        ▼
  ResNet-50 Backbone  ←── pretrained on COCO
        │
        ▼
  Transformer Encoder (6 layers)
        │
        ▼
  Transformer Decoder (6 layers, 100 object queries)
        │
        ▼
  FFN Head → [class logits + bounding boxes]
        │
        ▼
  Bipartite Matching Loss (Hungarian algorithm)
```

**Key design choices:**
- No anchor boxes or NMS — DETR is end-to-end
- Bipartite matching loss for training (one prediction per ground truth)
- Separate backbone LR (`1e-5`) vs transformer LR (`1e-4`)
- GradCAM hooks into `backbone.layer4` for clinical explainability

---

## Dataset

- **Format**: COCO JSON with bounding box annotations
- **Classes**: fracture (binary detection)
- **Input support**: DICOM (`.dcm`), JPEG, PNG
- **Preprocessing**: Normalize → Pad → Resize (≤800px longest side)

---

## Interview Answer: Pipeline End-to-End

> *"Data: bone fracture COCO dataset with DICOM X-rays → preprocessing via DetrImageProcessor → DETR (ResNet-50 backbone + transformer encoder-decoder, 100 object queries) → bipartite matching loss → eval: AUC 0.97, sensitivity 98% → export: ONNX FP32/FP16 → deploy on HuggingFace Gradio demo."*

---

## Resources

- [DETR Paper (arXiv)](https://arxiv.org/abs/2005.12872)
- [HuggingFace DETR](https://huggingface.co/facebook/detr-resnet-50)
- [Fine-tuning Tutorial (NielsRogge)](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)
- [DETR GitHub (Facebook)](https://github.com/facebookresearch/detr)
