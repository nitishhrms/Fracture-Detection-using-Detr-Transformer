"""
train.py — Training script for DETR fracture detection
Usage:
    python src/train.py --dataset_root "data/bone fracture.v2-release.coco" --epochs 30
"""
import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import DetrImageProcessor

from dataset import build_dataloaders
from model import DETRFractureDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Train DETR for fracture detection")
    parser.add_argument("--dataset_root", type=str,
                        default="data/bone fracture.v2-release.coco",
                        help="Path to COCO dataset root")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pretrained", type=str, default="facebook/detr-resnet-50")
    return parser.parse_args()


def main():
    args = parse_args()

    # Image processor
    image_processor = DetrImageProcessor.from_pretrained(args.pretrained)

    # Dataloaders
    loaders, train_ds, val_ds, _ = build_dataloaders(
        args.dataset_root, image_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build label maps
    categories = train_ds.coco.cats
    id2label = {k: v["name"] for k, v in categories.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    print(f"Labels ({num_labels}): {id2label}")

    # Model
    model = DETRFractureDetector(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        pretrained=args.pretrained,
    )

    # Callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="detr-fracture-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Logger
    logger = TensorBoardLogger("results/logs", name="detr_fracture")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=5,
        accelerator="auto",
        devices=1,
    )

    trainer.fit(model, loaders["train"], loaders["val"])
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
