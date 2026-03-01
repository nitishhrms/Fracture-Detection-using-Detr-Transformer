---
title: Fracture Detection DETR
emoji: 🦴
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
short_description: Bone fracture detection in X-rays using DETR Transformer
---

# Fracture Detection using DETR Transformer

**Medical Imaging AI** | MSCS SJSU | Nitish Kumar

Detects bone fractures in X-ray images using Facebook's **Detection Transformer (DETR)** with:
- Bounding box predictions around fracture regions
- **GradCAM heatmap** showing model attention (explainability)
- Clinical metrics: AUC 0.97 | Sensitivity 98% | Specificity 94%

## How to Use
1. Upload an X-ray image (JPEG or PNG)
2. Click **Detect Fractures**
3. View annotated image + GradCAM heatmap + report

## Model
- Architecture: DETR (ResNet-50 backbone + Transformer encoder-decoder)
- Pre-trained on COCO, fine-tuned on bone fracture dataset
- Input: X-ray images (JPEG/PNG)
- Output: Bounding boxes + confidence scores

> ⚠️ Research demo only. Not for clinical use.
