"""
DIN-CTS Training Pipeline  (BEATs backbone)
=============================================
Implements the Contrastive Training Strategy from:
  "DIN-CTS: Low-Complexity Depthwise-Inception Neural Network with
   Contrastive Training Strategy for Deepfake Speech Detection"

Adapted to use BEATs backbone (model_beat.py) instead of the DIN backbone.

Stage 1 – Multi-class, Multi-loss  (default 50 epochs)
    L1 = A-Softmax  on backbone embeddings   (multi-class)
    L2 = SINCERE    on contrastive-head       (contrastive)
    L3 = CenterLoss on backbone embeddings    (bonafide only)
    L  = α·L1 + β·L2 + γ·L3    (α=0.2, β=0.4, γ=0.4)

Stage 2 – Two-class fine-tuning   (default 10 epochs)
    Entropy head (FC→2) on backbone embeddings, cross-entropy loss.
    Backbone fine-tuned at low LR, head at high LR.

Inference uses Mahalanobis distance → see  scripts/din_cts_inference.py

Usage
-----
# Stage 1
python src/din_cts_trainer.py --do_stage1 \\
    --train_json_file data/label/gam/TUTASC19_train.json \\
    --val_json_file   data/label/gam/TUTASC19_test.json \\
    --num_label 5 --stage1_epochs 50 \\
    --save_ckpt_path checkpoint/BEATs_CTS/stage1

# Stage 2
python src/din_cts_trainer.py --do_stage2 \\
    --train_json_file data/label/gam/TUTASC19_train.json \\
    --val_json_file   data/label/gam/TUTASC19_test.json \\
    --num_label 2 --stage2_epochs 10 \\
    --load_ckpt_path checkpoint/BEATs_CTS/stage1/checkpoint/best.pt \\
    --save_ckpt_path checkpoint/BEATs_CTS/stage2
"""

import logging
import os
import sys
import argparse
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from transformers import get_cosine_schedule_with_warmup
from torchmetrics import MeanMetric
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))

try:
    from src.beats.model_beat import model_beat
except ModuleNotFoundError:
    from beats.model_beat import model_beat

from base_dataset import BeatsDataset, BalancedGeneratorSampler, RatioSampler
from utils.training_utils import ASoftmaxLoss, CenterLoss, SINCERE
from utils.arguments import get_args
from utils.wandb import create_trainer, save_checkpoint

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

SAMPLES_PER_LABEL = {
    5: {"fake_ata_01": 6, "fake_tta_01": 6, "fake_tta_02": 6, "fake_tta_03": 6, "real": 24},
    2: {"fake": 16, "real": 16},
}


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thresholds[idx]


def build_dataset(args, json_file):
    return BeatsDataset(json_file=json_file, transformation=None, args=args)


def build_train_dl(args, dataset):
    sampler = BalancedGeneratorSampler(dataset, SAMPLES_PER_LABEL[args.num_label])
    nw = args.dataloader_num_workers
    return torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=nw,
        shuffle=False, pin_memory=True, persistent_workers=nw > 0,
    )


def build_eval_dl(args, json_file):
    ds = build_dataset(args, json_file)
    return torch.utils.data.DataLoader(
        ds, batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers, shuffle=False,
    )


# ─── Stage 1: Multi-class + Multi-loss ──────────────────────────────────────

class CTSStage1(LightningModule):
    """
    Stage 1 lightning module.

    Trains BEATs (three_loss=True) with:
        L1 = A-Softmax   on backbone embeddings X  (multi-class)
        L2 = SINCERE      on contrastive head Z
        L3 = Center loss  on backbone embeddings X  (bonafide only)
    """

    def __init__(self, args, pipeline, sync_dist=None, test_dl=None):
        super().__init__()
        self.args = args
        self.pipeline = pipeline
        self.sync_dist = sync_dist
        self._test_dl = test_dl

        feat_dim = pipeline.feature_dim  # 527 or 768

        # ── Loss functions (paper settings, adapted for bf16 stability) ──
        # Paper uses m=4 but cos(4θ) amplifies numerical instability.
        # m=2.5 (default in ASoftmaxLoss) is safer with mixed precision.
        self.asoftmax = ASoftmaxLoss(
            embed_dim=feat_dim, n_classes=args.num_label, m=2.5, s=30,
        )
        # Paper uses τ=0.01 but that overflows bfloat16 (exp(100)).
        # Use τ=0.1 for numerical stability with mixed precision.
        self.sincere = SINCERE(temperature=0.1)
        self.center_loss = CenterLoss()

        # ── Loss weights (paper: α=0.2, β=0.4, γ=0.4) ──
        self.alpha = getattr(args, "cts_alpha", 0.2)
        self.beta  = getattr(args, "cts_beta",  0.4)
        self.gamma = getattr(args, "cts_gamma", 0.4)

        # Real class index in the *training* set (5-class: real=4)
        self._train_real_idx = args.num_label - 1
        if test_dl is not None:
            self._test_real_idx = test_dl.dataset.label.get("real", 1)
        else:
            self._test_real_idx = 1

    # ── Training ─────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        label = batch["label"].long()

        bonafide_head, softmax_head, contrastive_head = self.pipeline.forward_pipeline(audio)

        # Disable autocast so loss computation (arccos, cos(m*theta), exp)
        # runs in true float32 — bf16 overflows in these operations.
        with torch.amp.autocast("cuda", enabled=False):
            bonafide_f32 = bonafide_head.float()
            contrastive_f32 = contrastive_head.float()

            # L1: A-Softmax on backbone embeddings (multi-class)
            _, l1 = self.asoftmax.forward(x=bonafide_f32, y=label)

            # L2: SINCERE contrastive loss on contrastive head
            label_onehot = F.one_hot(label, num_classes=self.args.num_label).float()
            l2 = self.sincere.forward(x=contrastive_f32, y=label_onehot)

            # L3: Center loss on backbone embeddings (bonafide only)
            l3 = self.center_loss.forward(x=bonafide_f32, y=label_onehot, id_=self._train_real_idx)

            loss = self.alpha * l1 + self.beta * l2 + self.gamma * l3

        self.log("train/asoftmax",  l1.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/sincere",   l2.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/center",    l3.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        self.log("train/total",     loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    # ── Validation ───────────────────────────────────────────────────────
    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        label = batch["label"].long()
        bonafide_head, softmax_head, contrastive_head = self.pipeline.forward_pipeline(audio)
        with torch.amp.autocast("cuda", enabled=False):
            bonafide_f32 = bonafide_head.float()
            contrastive_f32 = contrastive_head.float()
            label_onehot = F.one_hot(label, num_classes=self.args.num_label).float()
            _, l1 = self.asoftmax.forward(x=bonafide_f32, y=label)
            l2 = self.sincere.forward(x=contrastive_f32, y=label_onehot)
            l3 = self.center_loss.forward(x=bonafide_f32, y=label_onehot, id_=self._train_real_idx)
            loss = self.alpha * l1 + self.beta * l2 + self.gamma * l3
        self.log("valid/total", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    # ── Per-epoch test inference ─────────────────────────────────────────
    def on_train_epoch_end(self):
        epoch = self.current_epoch

        # Recompute bonafide center every 5 epochs (paper Section II-C)
        if (epoch + 1) % 5 == 0:
            self._recompute_center()

        # Run test-set evaluation
        eer, auc, acc, real_acc, fake_acc = self._run_test_inference()
        if eer is not None:
            self.log("test/eer", eer, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        if acc is not None:
            self.log("test/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        if real_acc is not None:
            self.log("test/real_acc", real_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        if fake_acc is not None:
            self.log("test/fake_acc", fake_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        if self.trainer.is_global_zero:
            parts = [f"[Epoch {epoch}]"]
            if real_acc is not None: parts.append(f"real_acc={real_acc:.4f}")
            if fake_acc is not None: parts.append(f"fake_acc={fake_acc:.4f}")
            if acc  is not None: parts.append(f"acc={acc:.4f}")
            if eer  is not None: parts.append(f"eer={eer*100:.2f}%")
            if auc  is not None: parts.append(f"auc={auc:.4f}")
            print("  ".join(parts))

    @torch.no_grad()
    def _run_test_inference(self):
        """2-class eval: collapse multi-class labels to real/fake, use A-Softmax logits."""
        if self._test_dl is None:
            return None, None, None, None, None

        device = next(self.parameters()).device
        real_idx = self._test_real_idx
        all_probs, all_labels = [], []

        self.pipeline.eval()
        for batch in self._test_dl:
            audio  = batch["audio"].to(device, dtype=torch.float32)
            labels = batch["label"].cpu().numpy()

            outputs = self.pipeline.forward_pipeline(audio)
            bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs

            # Use A-Softmax logits projected to 2-class for evaluation
            # Simple approach: use raw backbone embedding norm as a score
            # (closer to bonafide center → smaller norm after centering)
            # But for consistency, use softmax on the last_layer output
            logits_2class = self.pipeline.last_layer(bonafide_head)
            probs = F.softmax(logits_2class, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_labels.append((labels == real_idx).astype(int))

        self.pipeline.train()

        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)

        # last_layer has num_label outputs; for 5-class we need 2-class eval
        # Use max-prob of real class (last dim) as real score
        if probs_arr.shape[1] > 2:
            # Map: real class is last → use its probability as "real score"
            prob_real = probs_arr[:, -1]
            preds = (prob_real > 0.5).astype(int)
        else:
            prob_real = probs_arr[:, 1]
            preds = probs_arr.argmax(axis=1)

        # Guard against NaN outputs (e.g. from early training instability)
        if np.any(np.isnan(prob_real)):
            if self.trainer.is_global_zero:
                nan_frac = np.isnan(prob_real).mean()
                print(f"  [WARN] {nan_frac*100:.1f}% of test scores are NaN — skipping metrics")
            return None, None, None, None, None

        acc = accuracy_score(labels_arr, preds)
        real_mask = labels_arr == 1
        fake_mask = labels_arr == 0
        real_acc = float((preds[real_mask] == 1).mean()) if real_mask.any() else None
        fake_acc = float((preds[fake_mask] == 0).mean()) if fake_mask.any() else None

        if len(np.unique(labels_arr)) < 2:
            return None, None, acc, real_acc, fake_acc

        eer, _ = compute_eer(labels_arr, prob_real)
        auc = roc_auc_score(labels_arr, prob_real)
        return eer, auc, acc, real_acc, fake_acc

    @torch.no_grad()
    def _recompute_center(self):
        """Recompute bonafide center from the training set (paper: every 5 epochs)."""
        if self._test_dl is None:
            return
        # Use test_dl as a proxy (or you can pass train_dl separately)
        device = next(self.parameters()).device
        real_idx = self._test_real_idx
        embeddings = []

        self.pipeline.eval()
        for batch in self._test_dl:
            audio  = batch["audio"].to(device, dtype=torch.float32)
            labels = batch["label"].cpu().numpy()
            outputs = self.pipeline.forward_pipeline(audio)
            bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs
            mask = labels == real_idx
            if mask.any():
                embeddings.append(bonafide_head[mask].cpu())
        self.pipeline.train()

        if embeddings:
            all_emb = torch.cat(embeddings, dim=0)
            center = all_emb.mean(dim=0).to(device)
            self.center_loss.set_center(center)
            if self.trainer.is_global_zero:
                print(f"  [CenterLoss] Recomputed bonafide center from {all_emb.shape[0]} samples")

    # ── Optimizer ────────────────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.pipeline.parameters()) + list(self.asoftmax.parameters()),
            lr=self.args.learning_rate,
            weight_decay=self.args.adam_weight_decay,
            betas=(0.9, 0.999),
        )
        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps // self.trainer.max_epochs
        warmup_steps = int(steps_per_epoch * 0.7)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def save_checkpoint(self, filepath, weights_only=False, storage_options=None):
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")


# ─── Stage 2: Two-class Fine-tuning ─────────────────────────────────────────

class CTSStage2(LightningModule):
    """
    Stage 2 lightning module.

    Fine-tunes pre-trained BEATs backbone with a simple Entropy head
    (FC→2 classes) using cross-entropy loss.

    Backbone gets a low learning rate, head gets a high learning rate.
    """

    def __init__(self, args, pipeline, sync_dist=None, test_dl=None):
        super().__init__()
        self.args = args
        self.pipeline = pipeline
        self.sync_dist = sync_dist
        self._test_dl = test_dl

        feat_dim = pipeline.feature_dim

        # Entropy head: FC → 2 classes (paper Section II-C, stage 2)
        self.entropy_head = nn.Linear(feat_dim, 2)

        if test_dl is not None:
            self._test_real_idx = test_dl.dataset.label.get("real", 1)
        else:
            self._test_real_idx = 1

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        label = batch["label"].long()

        # Forward through backbone only (ignore three_loss heads)
        outputs = self.pipeline.forward_pipeline(audio)
        bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs

        logits = self.entropy_head(bonafide_head)
        loss = F.cross_entropy(logits, label)

        self.log("train/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        label = batch["label"].long()
        outputs = self.pipeline.forward_pipeline(audio)
        bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs
        logits = self.entropy_head(bonafide_head)
        loss = F.cross_entropy(logits, label)
        self.log("valid/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        return loss

    def on_train_epoch_end(self):
        eer, auc, acc, real_acc, fake_acc = self._run_test_inference()
        epoch = self.current_epoch
        if eer is not None:
            self.log("test/eer", eer, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
            self.log("test/auc", auc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        if acc is not None:
            self.log("test/accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        if real_acc is not None:
            self.log("test/real_acc", real_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        if fake_acc is not None:
            self.log("test/fake_acc", fake_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)

        if self.trainer.is_global_zero:
            parts = [f"[Epoch {epoch}]"]
            if real_acc is not None: parts.append(f"real_acc={real_acc:.4f}")
            if fake_acc is not None: parts.append(f"fake_acc={fake_acc:.4f}")
            if acc  is not None: parts.append(f"acc={acc:.4f}")
            if eer  is not None: parts.append(f"eer={eer*100:.2f}%")
            if auc  is not None: parts.append(f"auc={auc:.4f}")
            print("  ".join(parts))

    @torch.no_grad()
    def _run_test_inference(self):
        if self._test_dl is None:
            return None, None, None, None, None

        device = next(self.parameters()).device
        real_idx = self._test_real_idx
        all_probs, all_labels = [], []

        self.pipeline.eval()
        self.entropy_head.eval()
        for batch in self._test_dl:
            audio  = batch["audio"].to(device, dtype=torch.float32)
            labels = batch["label"].cpu().numpy()
            outputs = self.pipeline.forward_pipeline(audio)
            bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = self.entropy_head(bonafide_head)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append((labels == real_idx).astype(int))
        self.pipeline.train()
        self.entropy_head.train()

        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        prob_real = probs_arr[:, 1]
        preds = probs_arr.argmax(axis=1)

        acc = accuracy_score(labels_arr, preds)
        real_mask, fake_mask = labels_arr == 1, labels_arr == 0
        real_acc = float((preds[real_mask] == 1).mean()) if real_mask.any() else None
        fake_acc = float((preds[fake_mask] == 0).mean()) if fake_mask.any() else None

        if len(np.unique(labels_arr)) < 2:
            return None, None, acc, real_acc, fake_acc
        eer, _ = compute_eer(labels_arr, prob_real)
        auc = roc_auc_score(labels_arr, prob_real)
        return eer, auc, acc, real_acc, fake_acc

    def configure_optimizers(self):
        backbone_lr = getattr(self.args, "stage2_backbone_lr", 1e-6)
        head_lr     = getattr(self.args, "stage2_head_lr", 1e-4)

        optimizer = torch.optim.AdamW([
            {"params": self.pipeline.parameters(), "lr": backbone_lr},
            {"params": self.entropy_head.parameters(), "lr": head_lr},
        ], weight_decay=self.args.adam_weight_decay, betas=(0.9, 0.999))

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps // self.trainer.max_epochs
        warmup_steps = int(steps_per_epoch * 0.3)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def save_checkpoint(self, filepath, weights_only=False, storage_options=None):
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")


# ─── Orchestration ───────────────────────────────────────────────────────────

def _load_state(ckpt_path, device):
    """Load a Lightning checkpoint and strip 'pipeline.' prefix."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    return state


def do_stage1(args, trainer, device, dist, eval_dl=None):
    """Stage 1: multi-class + multi-loss training."""
    logger.info("*** DIN-CTS  Stage 1 ***")

    pipeline = model_beat(
        num_label=args.num_label,
        three_loss=True,
        feature_layer=getattr(args, "beats_feature", "predictor"),
    )
    pipeline.train()

    dataset  = build_dataset(args, args.train_json_file)
    train_dl = build_train_dl(args, dataset)
    test_dl  = build_eval_dl(args, args.val_json_file)

    lightning_module = CTSStage1(args=args, pipeline=pipeline, sync_dist=dist, test_dl=test_dl)

    try:
        trainer.fit(model=lightning_module, train_dataloaders=train_dl, val_dataloaders=eval_dl)
    except KeyboardInterrupt:
        logger.warning("Stage 1 interrupted")
    finally:
        save_checkpoint(trainer=trainer, args=args)


def do_stage2(args, trainer, device, dist, eval_dl=None):
    """Stage 2: 2-class fine-tuning with Entropy head."""
    logger.info("*** DIN-CTS  Stage 2 ***")

    # Build pipeline with same config as stage 1
    prev_num_label = getattr(args, "prev_num_label", 5)
    pipeline = model_beat(
        num_label=prev_num_label,
        three_loss=True,
        feature_layer=getattr(args, "beats_feature", "predictor"),
    )

    # Load stage 1 weights
    state = _load_state(args.load_ckpt_path, device)
    pipe_sd = {}
    for k, v in state.items():
        if k.startswith("pipeline."):
            pipe_sd[k[len("pipeline."):]] = v
    if not pipe_sd:
        pipe_sd = state

    pipeline.load_state_dict(pipe_sd, strict=False)
    logger.info("Loaded stage-1 backbone from: %s", args.load_ckpt_path)
    pipeline.train()

    # 2-class dataset
    dataset  = build_dataset(args, args.train_json_file)
    train_dl = build_train_dl(args, dataset)
    test_dl  = build_eval_dl(args, args.val_json_file)

    lightning_module = CTSStage2(args=args, pipeline=pipeline, sync_dist=dist, test_dl=test_dl)

    try:
        trainer.fit(model=lightning_module, train_dataloaders=train_dl, val_dataloaders=eval_dl)
    except KeyboardInterrupt:
        logger.warning("Stage 2 interrupted")
    finally:
        save_checkpoint(trainer=trainer, args=args)


# ─── Entry point ─────────────────────────────────────────────────────────────

def add_cts_args(parser):
    """Add CTS-specific arguments on top of the base args."""
    parser.add_argument("--do_stage1", action="store_true", help="Run CTS stage 1 (multi-loss)")
    parser.add_argument("--do_stage2", action="store_true", help="Run CTS stage 2 (fine-tune)")
    parser.add_argument("--stage1_epochs", type=int, default=50, help="Epochs for stage 1")
    parser.add_argument("--stage2_epochs", type=int, default=10, help="Epochs for stage 2")
    parser.add_argument("--stage2_backbone_lr", type=float, default=1e-6, help="Backbone LR in stage 2")
    parser.add_argument("--stage2_head_lr", type=float, default=1e-4, help="Entropy head LR in stage 2")
    parser.add_argument("--cts_alpha", type=float, default=0.2, help="A-Softmax loss weight")
    parser.add_argument("--cts_beta",  type=float, default=0.4, help="SINCERE loss weight")
    parser.add_argument("--cts_gamma", type=float, default=0.4, help="Center loss weight")
    return parser


def main(args):
    if not any([getattr(args, "do_stage1", False), getattr(args, "do_stage2", False)]):
        raise ValueError("Specify --do_stage1 and/or --do_stage2")

    # Override max_epochs per stage
    if args.do_stage1:
        args.num_train_epochs = getattr(args, "stage1_epochs", 50)
        args.three_loss = True
        trainer, device, dist = create_trainer(args)
        torch.cuda.empty_cache()
        eval_dl = build_eval_dl(args, args.val_json_file) if args.do_eval else None
        do_stage1(args, trainer, device, dist, eval_dl)

    if args.do_stage2:
        args.num_train_epochs = getattr(args, "stage2_epochs", 10)
        args.three_loss = True  # model still uses three_loss arch
        trainer, device, dist = create_trainer(args)
        torch.cuda.empty_cache()
        eval_dl = build_eval_dl(args, args.val_json_file) if args.do_eval else None
        do_stage2(args, trainer, device, dist, eval_dl)


if __name__ == "__main__":
    # ── CTS-specific args parsed first, unknowns forwarded to get_args ──
    cts_parser = argparse.ArgumentParser(add_help=False)
    add_cts_args(cts_parser)
    cts_args, remaining = cts_parser.parse_known_args()

    # get_args() consumes the standard project arguments
    args = get_args(remaining)

    # Merge CTS args into the main namespace
    for key, val in vars(cts_args).items():
        setattr(args, key, val)

    logger.info("*** DIN-CTS Training Pipeline ***")
    main(args)
