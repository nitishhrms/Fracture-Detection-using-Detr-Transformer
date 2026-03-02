"""
test_all.py — Full end-to-end test using synthetic data (no real dataset needed)
Tests: imports, model forward pass, GradCAM, ONNX export, parity test, benchmark
Run: python test_all.py
"""
import os, sys, json, shutil, tempfile, traceback
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "export"))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = {}

def check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        results[name] = "PASS"
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc()
        results[name] = f"FAIL: {e}"

# ── helpers ──────────────────────────────────────────────────────────────────
def make_dummy_image(H=480, W=640):
    """Create a random RGB PIL image."""
    arr = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def make_dummy_coco_dataset(root):
    """Create a minimal COCO dataset with 4 fake images in train/valid/test."""
    from PIL import Image as PILImage
    import json

    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(root, split)
        os.makedirs(split_dir, exist_ok=True)

        images, annotations = [], []
        for i in range(4):
            fname = f"img_{i:03d}.jpg"
            img = PILImage.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            img.save(os.path.join(split_dir, fname))
            images.append({"id": i, "file_name": fname, "width": 640, "height": 480})
            annotations.append({
                "id": i, "image_id": i, "category_id": 1,
                "bbox": [50, 50, 100, 80], "area": 8000, "iscrowd": 0
            })

        coco_json = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "fracture", "supercategory": "bone"}],
        }
        with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
            json.dump(coco_json, f)

    return root

# ── setup ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  FRACTURE DETECTION — FULL TEST SUITE")
print("="*60)

TMP_DIR     = tempfile.mkdtemp()
DATASET_DIR = make_dummy_coco_dataset(TMP_DIR)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

model_obj = None
processor_obj = None

# ── 1. imports ───────────────────────────────────────────────────────────────
print("\n[1] IMPORTS")
def test_imports():
    import torch, transformers, pytorch_lightning, pydicom
    import cv2, supervision, sklearn, onnx, onnxruntime, pandas
    from dataset import CocoDetection, build_dataloaders, load_image
    from model import DETRFractureDetector
    from evaluate import evaluate_model, print_metrics_table, compute_iou
    from gradcam import DETRGradCAM, run_gradcam
    from predict import predict_single
    from export_onnx import export_to_onnx, verify_onnx
    from parity_test import parity_check
    from benchmark import run_benchmark

check("All imports", test_imports)

# ── 2. model load ─────────────────────────────────────────────────────────────
print("\n[2] MODEL LOAD")
from transformers import DetrImageProcessor, DetrForObjectDetection
from model import DETRFractureDetector

def test_model_load():
    global model_obj, processor_obj
    id2label = {1: "fracture"}
    label2id = {"fracture": 1}
    processor_obj = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model_obj = DETRFractureDetector(
        num_labels=1, id2label=id2label, label2id=label2id
    )
    total = sum(p.numel() for p in model_obj.parameters()) / 1e6
    assert total > 30, f"Expected >30M params, got {total:.1f}M"

check("DETR model loads (41M params)", test_model_load)

# ── 3. dataset ───────────────────────────────────────────────────────────────
print("\n[3] DATASET")
from dataset import build_dataloaders, load_image

def test_dataset_load():
    loaders, tr, va, te = build_dataloaders(
        DATASET_DIR, processor_obj, batch_size=2, num_workers=0
    )
    assert len(tr) == 4
    assert len(va) == 4
    pv, tgt = tr[0]
    assert pv.ndim == 3 and pv.shape[0] == 3

check("Dataset loads (4 train / 4 val / 4 test)", test_dataset_load)

def test_dicom_load():
    import pydicom
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian
    import pydicom.uid
    # Create minimal synthetic DICOM
    dcm_path = os.path.join(TMP_DIR, "test.dcm")
    ds = FileDataset(dcm_path, {}, file_meta=pydicom.dataset.FileMetaDataset(), preamble=b"\x00"*128)
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 128
    ds.Columns = 128
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.random.randint(0, 200, (128, 128), dtype=np.uint8).tobytes()
    pydicom.dcmwrite(dcm_path, ds)
    img = load_image(dcm_path)
    assert img.mode == "RGB"
    assert img.size == (128, 128)

check("DICOM loading (pydicom)", test_dicom_load)

# ── 4. forward pass ──────────────────────────────────────────────────────────
print("\n[4] MODEL FORWARD PASS")
def test_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pil_img = make_dummy_image()
    enc = processor_obj(images=pil_img, return_tensors="pt")
    pv  = enc["pixel_values"].to(device)
    pm  = enc.get("pixel_mask", torch.ones(1, pv.shape[2], pv.shape[3])).to(device)
    model_obj.model.to(device).eval()
    with torch.no_grad():
        out = model_obj.model(pixel_values=pv, pixel_mask=pm)
    assert out.logits.shape[0] == 1
    assert out.pred_boxes.shape[2] == 4

check(f"Forward pass ({'CUDA' if torch.cuda.is_available() else 'CPU'})", test_forward)

# ── 5. GradCAM ───────────────────────────────────────────────────────────────
print("\n[5] GRADCAM")
def test_gradcam():
    from gradcam import DETRGradCAM
    device = "cpu"   # GradCAM needs grad, use CPU for test
    m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    gcam = DETRGradCAM(m, device=device)
    pil_img = make_dummy_image(320, 320)
    enc = processor_obj(images=pil_img, return_tensors="pt")
    pv = enc["pixel_values"]
    pm = enc.get("pixel_mask", torch.ones(1, pv.shape[2], pv.shape[3]))
    heatmap = gcam.generate(pv, pm, query_idx=0)
    assert heatmap is not None
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    overlay = gcam.overlay(np.array(pil_img), heatmap)
    assert overlay.shape[:2] == (320, 320)
    gcam.remove_hooks()

check("GradCAM heatmap generation", test_gradcam)

def test_gradcam_save():
    from gradcam import run_gradcam
    tmp_img = os.path.join(TMP_DIR, "test_xray.jpg")
    make_dummy_image().save(tmp_img)
    m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    out_path, heatmap, _ = run_gradcam(
        image_path=tmp_img,
        model_or_checkpoint=m,
        image_processor=processor_obj,
        output_dir=os.path.join(RESULTS_DIR, "gradcam"),
        score_threshold=0.1,
    )
    assert os.path.exists(out_path)

check("GradCAM saves 3-panel PNG", test_gradcam_save)

# ── 6. predict ───────────────────────────────────────────────────────────────
print("\n[6] INFERENCE")
def test_predict():
    from predict import predict_single
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp_img = os.path.join(TMP_DIR, "test_xray2.jpg")
    make_dummy_image().save(tmp_img)
    m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    annotated, results = predict_single(tmp_img, m, processor_obj, device=device)
    assert isinstance(annotated, Image.Image)

check("Single image inference", test_predict)

# ── 7. evaluate (clinical metrics) ───────────────────────────────────────────
print("\n[7] CLINICAL METRICS")
def test_evaluate():
    from evaluate import evaluate_model, print_metrics_table
    from dataset import build_dataloaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaders, _, _, _ = build_dataloaders(DATASET_DIR, processor_obj, batch_size=2, num_workers=0)
    m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
    metrics = evaluate_model(m, loaders["test"], processor_obj, device=device)
    for key in ["sensitivity", "specificity", "precision", "f1", "auc"]:
        assert key in metrics, f"Missing metric: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} out of range"
    print_metrics_table(metrics)

check("Clinical metrics (sensitivity/specificity/AUC)", test_evaluate)

# ── 8. ONNX export ────────────────────────────────────────────────────────────
print("\n[8] ONNX EXPORT")
ONNX_FP32 = os.path.join(RESULTS_DIR, "test_detr_fp32.onnx")

def test_onnx_export():
    from export_onnx import export_to_onnx, verify_onnx
    m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    export_to_onnx(m, ONNX_FP32, image_size=224, opset_version=14, use_fp16=False)
    assert os.path.exists(ONNX_FP32)
    verify_onnx(ONNX_FP32)

check("ONNX FP32 export + verify", test_onnx_export)

# ── 9. parity test ────────────────────────────────────────────────────────────
print("\n[9] PARITY TEST (PyTorch vs ORT)")
def test_parity():
    from parity_test import parity_check
    if not os.path.exists(ONNX_FP32):
        raise FileNotFoundError("ONNX file missing — run ONNX export first")
    m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    passed = parity_check(
        onnx_path=ONNX_FP32,
        pytorch_model=m,
        image_size=224,
        batch_sizes=[1, 2],
        atol=1e-3,
    )
    assert passed, "Parity test failed — max diff exceeded tolerance"

check("PyTorch vs ORT parity (<=1e-3)", test_parity)

# ── 10. benchmark ─────────────────────────────────────────────────────────────
print("\n[10] BENCHMARK")
def test_benchmark():
    from benchmark import run_benchmark
    if not os.path.exists(ONNX_FP32):
        raise FileNotFoundError("ONNX file missing")
    m = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    rows = run_benchmark(
        pytorch_model=m,
        onnx_fp32=ONNX_FP32,
        batch_sizes=[1, 2],
        output_dir=RESULTS_DIR,
    )
    assert len(rows) > 0
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    assert os.path.exists(csv_path)

check("Benchmark (latency CSV + chart)", test_benchmark)

# ── summary ───────────────────────────────────────────────────────────────────
shutil.rmtree(TMP_DIR, ignore_errors=True)

passed = sum(1 for v in results.values() if v == "PASS")
total  = len(results)
print("\n" + "="*60)
print(f"  RESULTS: {passed}/{total} PASSED")
print("="*60)
for name, status in results.items():
    icon = "✓" if status == "PASS" else "✗"
    print(f"  {icon} {name}")
print("="*60 + "\n")

if passed < total:
    sys.exit(1)
