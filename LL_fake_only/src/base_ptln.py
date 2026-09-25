import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from typing import Any, Optional
import torch.nn.functional as F
import numpy as np
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from utils.training_utils import CenterLoss, ASoftmaxLoss, ArcFaceLoss, SINCERE, CenterContrastiveLoss


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thresholds[idx]


class BasePTLN(LightningModule):
    def __init__(self, args, sync_dist=None, pipeline=None, test_dl=None):
        super().__init__()
        self.args = args
        self.pipeline = pipeline
        self.sync_dist = sync_dist
        self.mean_valid_loss = MeanMetric()

        ccl_real_w = getattr(self.args, 'ccl_real_w', 4.0)
        self.center_constrastive_loss = CenterContrastiveLoss(embed_dim=self.args.embed_dim, n_classes=2, m=0.7, s=30, lambda_c=2, real_w=ccl_real_w)
        self.arcface_loss = ArcFaceLoss(embed_dim=self.args.embed_dim, n_classes=self.args.num_label, m=3, s=30)
        for param in self.arcface_loss.parameters():
            param.requires_grad = True
        self.contrastive_loss = SINCERE()

        # ── Test dataloader for per-epoch inference ───────────────────────────
        self._test_dl = test_dl
        if test_dl is not None:
            self._real_class_idx = test_dl.dataset.label.get('real', 1)
        else:
            self._real_class_idx = 1

        # real label index in the TRAINING set (used for label_2class)
        # training set has (efs, fr, fs, fake, real) → real=4
        self._train_real_idx = 4

    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        label = batch['label']

        if self.args.num_label == 5:
            bonafide_head, softmax_head, constrastive_head = self.pipeline.forward_pipeline(audio)

            label_5class = label.long()
            # Use training set real index (always 4 for 5-class)
            label_2class = (label == self._train_real_idx).long()  # fake=0, real=1

            _, arcface_loss = self.arcface_loss.forward(x=bonafide_head, y=label_5class)

            center_constrastive_loss, _ = self.center_constrastive_loss.forward(
                x=bonafide_head,
                labels=label_2class
            )

            loss = (0.2 * arcface_loss) + (0.8 * center_constrastive_loss)

            # Optional: add weighted cross-entropy to 5-class training
            # NOTE: softmax_head is already softmax'd → use nll_loss(log(probs))
            ce_w = getattr(self.args, 'ce_w', 0.0)
            if ce_w > 0:
                ce_real_w = getattr(self.args, 'ce_real_w', 1.0)
                ce_class_w = torch.tensor(
                    [1.0, 1.0, 1.0, 1.0, ce_real_w],
                    device=softmax_head.device
                )
                log_probs = torch.log(softmax_head.clamp(min=1e-12))
                ce_loss = F.nll_loss(log_probs, label_5class, weight=ce_class_w)
                loss = loss + ce_w * ce_loss
                self.log("train/ce_loss", ce_loss.detach(),
                         on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

            self.log("train/center_constrastive_loss", center_constrastive_loss.detach(),
                     on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            self.log("train/arcface_loss", arcface_loss.detach(),
                     on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            self.log("train/total_loss", loss.detach(),
                     on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        else:
            model_pred = self.pipeline.forward_pipeline(audio)
            if isinstance(model_pred, tuple):
                model_pred = model_pred[0]
            loss = F.cross_entropy(model_pred, label)
            self.log("train/loss", loss.detach(),
                     on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch['audio']
        label = batch['label']

        if self.args.num_label == 5:
            bonafide_head, softmax_head, constrastive_head = self.pipeline.forward_pipeline(audio)
            label = torch.nn.functional.one_hot(label, num_classes=self.args.num_label).float()
            center_loss      = self.center_loss.forward(x=bonafide_head, y=label, id_=4)
            _, asoftmax_loss = self.asoftmax_loss.forward(x=softmax_head, y=label)
            contrastive_loss = self.contrastive_loss.forward(x=constrastive_head, y=label)
            loss = (0.6 * asoftmax_loss) + (0.2 * center_loss) + (0.2 * contrastive_loss)

            self.log("valid/center_loss",      center_loss.detach(),      on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            self.log("valid/asoftmax_loss",    asoftmax_loss.detach(),    on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            self.log("valid/contrastive_loss", contrastive_loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            self.log("valid/total_loss",       loss.detach(),             on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        else:
            model_pred = self.pipeline.forward_pipeline(audio)
            if isinstance(model_pred, tuple):
                model_pred = model_pred[0]
            loss = F.cross_entropy(model_pred, label)
            self.log("valid/loss", loss.detach(),
                     on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        return loss

    def _run_test_inference(self):
        if self._test_dl is None:
            return None, None, None, None, None

        device   = next(self.parameters()).device
        real_idx = self._real_class_idx  # from test dataset label mapping
        all_probs, all_labels = [], []

        self.pipeline.eval()
        self.center_constrastive_loss.eval()

        with torch.no_grad():
            for batch in self._test_dl:
                audio  = batch['audio'].to(device, dtype=torch.float32)
                labels = batch['label'].cpu().numpy()
                B      = audio.size(0)

                outputs       = self.pipeline.forward_pipeline(audio)
                bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs

                if self.args.num_label == 2:
                    probs = F.softmax(bonafide_head, dim=1).cpu().numpy()
                else:
                    dummy     = torch.zeros(B, dtype=torch.long, device=device)
                    _, logits = self.center_constrastive_loss.forward(x=bonafide_head, labels=dummy)
                    probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, 2]
                all_probs.append(probs)
                all_labels.append((labels == real_idx).astype(int))  # 1=real, 0=fake

        self.pipeline.train()
        self.center_constrastive_loss.train()

        probs_arr  = np.concatenate(all_probs,  axis=0)  # [N, 2]
        labels_arr = np.concatenate(all_labels, axis=0)  # [N]
        prob_real  = probs_arr[:, 1]                      # CCL real center score
        preds      = probs_arr.argmax(axis=1)
        acc        = accuracy_score(labels_arr, preds)

        # Per-class recall — i.e. accuracy within each class.
        real_mask = labels_arr == 1
        fake_mask = labels_arr == 0
        real_acc = float((preds[real_mask] == 1).mean()) if real_mask.any() else None
        fake_acc = float((preds[fake_mask] == 0).mean()) if fake_mask.any() else None

        if len(np.unique(labels_arr)) < 2:
            return None, None, acc, real_acc, fake_acc

        eer, _ = compute_eer(labels_arr, prob_real)
        auc    = roc_auc_score(labels_arr, prob_real)
        return eer, auc, acc, real_acc, fake_acc


    def on_train_epoch_end(self):
        eer, auc, acc, real_acc, fake_acc = self._run_test_inference()
        epoch = self.current_epoch

        if eer is not None:
            self.log("test/eer", eer,
                    on_step=False, on_epoch=True, prog_bar=True,
                    logger=True, sync_dist=self.sync_dist)
            self.log("test/auc", auc,
                    on_step=False, on_epoch=True, prog_bar=True,
                    logger=True, sync_dist=self.sync_dist)
        if acc is not None:
            self.log("test/accuracy", acc,
                    on_step=False, on_epoch=True, prog_bar=True,
                    logger=True, sync_dist=self.sync_dist)
        if real_acc is not None:
            self.log("test/real_acc", real_acc,
                    on_step=False, on_epoch=True, prog_bar=True,
                    logger=True, sync_dist=self.sync_dist)
        if fake_acc is not None:
            self.log("test/fake_acc", fake_acc,
                    on_step=False, on_epoch=True, prog_bar=True,
                    logger=True, sync_dist=self.sync_dist)

        # Console summary so it appears in stdout/log files alongside wandb.
        if self.trainer.is_global_zero:
            parts = [f"[Epoch {epoch}]"]
            if real_acc is not None: parts.append(f"real_acc={real_acc:.4f}")
            if fake_acc is not None: parts.append(f"fake_acc={fake_acc:.4f}")
            if acc is not None:      parts.append(f"acc={acc:.4f}")
            if eer is not None:      parts.append(f"eer={eer*100:.2f}%")
            if auc is not None:      parts.append(f"auc={auc:.4f}")
            print("  ".join(parts))

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.pipeline.parameters(),
    #         lr=self.args.learning_rate,
    #         weight_decay=self.args.adam_weight_decay,
    #         betas=(0.9, 0.999),
    #     )
    #     total_steps = self.trainer.estimated_stepping_batches
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=720,
    #         num_training_steps=total_steps,
    #     )
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    #     }

    def configure_optimizers(self):
        # Include all trainable parameters: pipeline + CCL centers + ArcFace weights
        all_params = list(self.pipeline.parameters()) + \
                     list(self.center_constrastive_loss.parameters()) + \
                     list(self.arcface_loss.parameters())
        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.adam_weight_decay,
            betas=(0.9, 0.999),
        )
        total_steps = self.trainer.estimated_stepping_batches
        # 1 epoch warmup
        steps_per_epoch = total_steps // self.trainer.max_epochs
        warmup_steps = steps_per_epoch * 0.7

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
    def save_checkpoint(self, filepath, weights_only: bool = False,
                        storage_options: Optional[Any] = None) -> None:
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")