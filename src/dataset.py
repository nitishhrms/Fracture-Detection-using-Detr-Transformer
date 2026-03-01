"""
dataset.py — CocoDetection dataset with DICOM + standard image support
Supports: .dcm (DICOM), .jpg, .png, .bmp
"""
import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("[WARN] pydicom not installed. DICOM support disabled. Run: pip install pydicom")


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from disk. Handles DICOM (.dcm) and standard formats.
    Returns a PIL RGB image.
    """
    ext = os.path.splitext(image_path)[-1].lower()

    if ext == ".dcm":
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM images. Run: pip install pydicom")
        dcm = pydicom.dcmread(image_path)
        arr = dcm.pixel_array.astype(np.float32)

        # Normalize to 0-255
        arr -= arr.min()
        if arr.max() > 0:
            arr = arr / arr.max() * 255.0
        arr = arr.astype(np.uint8)

        # DICOM can be grayscale (2D) or multi-frame (3D)
        if arr.ndim == 2:
            # Grayscale → RGB
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = cv2.cvtColor(arr.squeeze(-1), cv2.COLOR_GRAY2RGB)

        return Image.fromarray(arr)

    else:
        img = Image.open(image_path).convert("RGB")
        return img


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    COCO-format detection dataset with DICOM support.
    Drop-in replacement for the original notebook class.
    """

    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        annotation_file_path: str = None,
        train: bool = True,
    ):
        if annotation_file_path is None:
            annotation_file_path = os.path.join(
                image_directory_path, "_annotations.coco.json"
            )
        super().__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def _load_image(self, id: int):
        """Override to support DICOM files."""
        path = self.coco.loadImgs(id)[0]["file_name"]
        full_path = os.path.join(self.root, path)
        return load_image(full_path)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = self._load_image(image_id)
        annotations = self.coco.imgToAnns[image_id]
        annotations = {"image_id": image_id, "annotations": annotations}

        encoding = self.image_processor(
            images=image, annotations=annotations, return_tensors="pt"
        )
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target


def collate_fn(batch, image_processor):
    """Custom collate to handle variable-size DETR inputs."""
    pixel_values = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    return {"pixel_values": encoding["pixel_values"], "pixel_mask": encoding["pixel_mask"], "labels": targets}


def build_dataloaders(dataset_root: str, image_processor, batch_size: int = 4, num_workers: int = 2):
    """
    Build train/val/test DataLoaders.

    Args:
        dataset_root: path to COCO dataset root (contains train/, valid/, test/)
        image_processor: HuggingFace DetrImageProcessor
        batch_size: batch size for training
        num_workers: number of DataLoader workers

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    from functools import partial

    train_ds = CocoDetection(os.path.join(dataset_root, "train"), image_processor, train=True)
    val_ds = CocoDetection(os.path.join(dataset_root, "valid"), image_processor, train=False)
    test_ds = CocoDetection(os.path.join(dataset_root, "test"), image_processor, train=False)

    _collate = partial(collate_fn, image_processor=image_processor)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            collate_fn=_collate, num_workers=num_workers),
        "val":   DataLoader(val_ds, batch_size=1, shuffle=False,
                            collate_fn=_collate, num_workers=num_workers),
        "test":  DataLoader(test_ds, batch_size=1, shuffle=False,
                            collate_fn=_collate, num_workers=num_workers),
    }

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return loaders, train_ds, val_ds, test_ds
