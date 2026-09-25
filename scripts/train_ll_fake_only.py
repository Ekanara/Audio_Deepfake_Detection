"""
train_ll_fake_only.py — 2-Stage BEATs Training for LL_fake_only
================================================================

Trains a BEATs audio deepfake detection model using a 2-stage pipeline.
The model learns to separate real/fake audio in a 527-dim embedding space
via ArcFace + CCL + CE losses, then inference uses Log-Likelihood scoring
on the fake-only training distribution (see infer_ll_fake_only.py).

Usage:
------
  # Stage 1: Feature learning (15 epochs)
  python scripts/train_ll_fake_only.py --stage 1

  # Stage 2: Refinement (8 epochs, loads best Stage 1 checkpoint)
  python scripts/train_ll_fake_only.py --stage 2 \
      --load_ckpt_path checkpoint/ll_fake_only_stage1/sample-XX.ckpt

  # Both stages sequentially
  python scripts/train_ll_fake_only.py --stage both

Requirements:
  - BEATs pretrained checkpoint at src/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  - Training JSON: data/label/beats/Event_train_stage1.json
  - Validation JSON: data/label/beats/test_track2.json
  - wandb account (for logging)
"""
import argparse
import glob
import logging
import os
import sys

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from base_dataset import BeatsDataset, BalancedGeneratorSampler
from base_ptln import BasePTLN
from beats.model_beat import model_beat
from utils.training_utils import ArcFaceLoss, CenterContrastiveLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameters (matching the paper's best config)
# ─────────────────────────────────────────────────────────────────────────────
STAGE1_DEFAULTS = dict(
    num_train_epochs=15,
    learning_rate=3e-7,
    ce_w=3.0,
    ce_real_w=2.5,
    ccl_real_w=4.0,
)
STAGE2_DEFAULTS = dict(
    num_train_epochs=8,
    learning_rate=1.5e-7,
    ce_w=3.0,
    ce_real_w=4.0,
    ccl_real_w=4.0,
)

SAMPLES_PER_LABEL = {
    "fake_ata_01": 6,
    "fake_tta_01": 6,
    "fake_tta_02": 6,
    "fake_tta_03": 6,
    "real": 24,
}


def parse_args():
    p = argparse.ArgumentParser(description="Train BEATs for LL_fake_only")
    p.add_argument("--stage", choices=["1", "2", "both"], default="both",
                   help="Which training stage to run")
    p.add_argument("--train_json", default="data/label/beats/Event_train_stage1.json")
    p.add_argument("--val_json", default="data/label/beats/test_track2.json")
    p.add_argument("--load_ckpt_path", default=None,
                   help="Checkpoint to load for stage 2 (auto-detected if --stage both)")
    p.add_argument("--output_dir", default="checkpoint/ll_fake_only",
                   help="Base output directory (stage suffix appended automatically)")
    p.add_argument("--wandb_project", default="LL_fake_only")
    p.add_argument("--wandb_api", default=None, help="wandb API key")

    p.add_argument("--devices", type=int, nargs="+", default=[0])
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 for WSL)")
    p.add_argument("--train_batch_size", type=int, default=None,
                   help="Override per-GPU batch size (default: sum of SAMPLES_PER_LABEL = 48)")
    p.add_argument("--val_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Override stage defaults
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--ce_w", type=float, default=None)
    p.add_argument("--ce_real_w", type=float, default=None)
    p.add_argument("--ccl_real_w", type=float, default=None)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model / Dataset / DataLoader builders
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    """Build BEATs pipeline in 3-loss (5-class) mode with predictor features."""
    pipeline = model_beat(
        num_label=5,
        three_loss=True,
        feature_layer="predictor",  # 527-dim bonafide_head
    )
    return pipeline


def build_dataset(json_file):
    """Build BeatsDataset with 5-class labels, no augmentation."""
    class Args:
        num_label = 5
        three_loss = True
        audio_aug = False
        audio_mixup = False
        audio_aug_prob = 0
        audio_mixup_prob = 0

    return BeatsDataset(json_file=json_file, transformation=None, args=Args())


def build_train_dataloader(dataset, num_workers=0):
    """Build balanced training DataLoader using BalancedGeneratorSampler.

    Each batch contains exactly:
      - 6 samples from each of 4 fake generators = 24 fake
      - 24 real samples
      - Total = 48 per batch
    """
    sampler = BalancedGeneratorSampler(
        dataset=dataset,
        samples_per_label=SAMPLES_PER_LABEL,
        cycle_short_labels=True,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_val_dataloader(json_file, batch_size=64, num_workers=0):
    """Build simple validation DataLoader."""
    dataset = build_dataset(json_file)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_stage(stage_num, args, stage_defaults, load_ckpt=None):
    """Run one training stage."""
    # Merge defaults with CLI overrides
    lr = args.learning_rate or stage_defaults["learning_rate"]
    epochs = args.num_train_epochs or stage_defaults["num_train_epochs"]
    ce_w = args.ce_w if args.ce_w is not None else stage_defaults["ce_w"]
    ce_real_w = args.ce_real_w if args.ce_real_w is not None else stage_defaults["ce_real_w"]
    ccl_real_w = args.ccl_real_w if args.ccl_real_w is not None else stage_defaults["ccl_real_w"]

    output_dir = f"{args.output_dir}_stage{stage_num}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"STAGE {stage_num} TRAINING")
    logger.info(f"  LR          : {lr}")
    logger.info(f"  Epochs      : {epochs}")
    logger.info(f"  CE weight   : {ce_w}")
    logger.info(f"  CE real_w   : {ce_real_w}")
    logger.info(f"  CCL real_w  : {ccl_real_w}")
    logger.info(f"  Output      : {output_dir}")
    if load_ckpt:
        logger.info(f"  Load ckpt   : {load_ckpt}")
    logger.info("=" * 60)

    # Build model
    pipeline = build_model()

    # Load checkpoint for stage 2
    if load_ckpt:
        logger.info(f"Loading checkpoint: {load_ckpt}")
        ckpt = torch.load(load_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        pipeline_state = {
            k.replace("pipeline.", ""): v
            for k, v in state_dict.items()
            if k.startswith("pipeline.")
        }
        pipeline.load_state_dict(pipeline_state)
        logger.info("Checkpoint loaded.")

    # Build data
    train_dataset = build_dataset(args.train_json)
    train_dl = build_train_dataloader(train_dataset, num_workers=args.num_workers)
    val_dl = build_val_dataloader(args.val_json, batch_size=args.val_batch_size,
                                  num_workers=args.num_workers)

    # Build Lightning module
    # BasePTLN expects an args namespace with these fields:
    class TrainArgs:
        pass

    train_args = TrainArgs()
    train_args.learning_rate = lr
    train_args.adam_beta1 = 0.9
    train_args.adam_beta2 = 0.999
    train_args.adam_weight_decay = 0
    train_args.max_grad_norm = 1.0
    train_args.num_label = 5
    train_args.three_loss = True
    train_args.embed_dim = 527
    train_args.ccl_real_w = ccl_real_w
    train_args.ce_w = ce_w
    train_args.ce_real_w = ce_real_w
    train_args.mode = "beats"
    train_args.lr_mult = 10

    pl_module = BasePTLN(
        args=train_args,
        pipeline=pipeline,
        test_dl=val_dl,
    )

    # Trainer
    if args.wandb_api:
        os.system(f"wandb login --relogin {args.wandb_api}")

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"stage{stage_num}_LR{lr}_ce{ce_w}_realW{ce_real_w}",
        log_model=False,
    )

    checkpoint_cb = ModelCheckpoint(
        save_top_k=5,
        monitor="valid/loss",
        mode="min",
        dirpath=output_dir,
        filename="sample-{epoch:02d}",
        save_weights_only=False,
    )

    dist = len(args.devices) > 1
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_cb],
        strategy="ddp_find_unused_parameters_true" if dist else "auto",
        log_every_n_steps=10,
        logger=wandb_logger,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    trainer.fit(pl_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # Return best checkpoint path
    best = checkpoint_cb.best_model_path
    logger.info(f"Stage {stage_num} done. Best checkpoint: {best}")
    return best


def find_best_stage1_ckpt(output_dir):
    """Auto-detect best stage 1 checkpoint."""
    pattern = f"{output_dir}_stage1/sample-*.ckpt"
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(
            f"No stage 1 checkpoints found at {pattern}. "
            f"Run --stage 1 first or specify --load_ckpt_path."
        )
    return ckpts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.stage in ("1", "both"):
        best_s1 = train_stage(1, args, STAGE1_DEFAULTS)

    if args.stage in ("2", "both"):
        if args.stage == "both":
            load_ckpt = best_s1
        elif args.load_ckpt_path:
            load_ckpt = args.load_ckpt_path
        else:
            load_ckpt = find_best_stage1_ckpt(args.output_dir)

        train_stage(2, args, STAGE2_DEFAULTS, load_ckpt=load_ckpt)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
