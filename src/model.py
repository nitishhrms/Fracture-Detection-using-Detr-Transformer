"""
model.py — DETR Lightning module for fracture detection
"""
import torch
import pytorch_lightning as pl
from transformers import DetrForObjectDetection


class DETRFractureDetector(pl.LightningModule):
    """
    PyTorch Lightning wrapper around HuggingFace DETR.
    Tracks train/val loss and logs per epoch.
    """

    def __init__(
        self,
        num_labels: int,
        id2label: dict,
        label2id: dict,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        pretrained: str = "facebook/detr-resnet-50",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DetrForObjectDetection.from_pretrained(
            pretrained,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        for k, v in loss_dict.items():
            self.log(f"val_{k}", v.item())
        return loss

    def configure_optimizers(self):
        # Separate backbone and transformer learning rates
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters()
                           if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if "backbone" in n and p.requires_grad],
                "lr": self.hparams.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
