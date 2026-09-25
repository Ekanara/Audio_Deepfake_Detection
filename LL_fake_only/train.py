"""
train.py — 2-Stage BEATs Training for LL_fake_only
====================================================

Trains a BEATs audio deepfake detection model using a 2-stage pipeline:

  Stage 1 (Feature Learning):  15 epochs, LR=3e-7, CE=3, real_w=2.5
  Stage 2 (Refinement):         8 epochs, LR=1.5e-7, CE=3, real_w=4.0

The model learns discriminative 527-dim embeddings via three losses:
  L = 0.2 * ArcFace + 0.8 * CCL + ce_w * CE

After training, use infer.py to score test samples against the fake
training distribution using Log-Likelihood scoring.

Usage
-----
  # Stage 1 only
  python LL_fake_only/train.py --stage 1

  # Stage 2 (loads best stage 1 checkpoint)
  python LL_fake_only/train.py --stage 2 \
      --load_ckpt_path checkpoint/ll_fake_only_stage1/sample-XX.ckpt

  # Both stages sequentially
  python LL_fake_only/train.py --stage both

  # Custom data paths
  python LL_fake_only/train.py --stage both \
      --train_json LL_fake_only/data/Event_train_stage1.json \
      --val_json LL_fake_only/data/test_track2.json

Prerequisites
-------------
  - BEATs pretrained weights at src/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  - Training JSON with 5-class labels (real + 4 fake generators)
  - wandb account for logging (pass --wandb_api <key>)
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

_HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the bundled src/ (standalone package); fall back to the repo's ../src.
for _cand in (os.path.join(_HERE, "src"), os.path.join(_HERE, "..", "src")):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.dirname(_cand))  # parent → enables `import src.xxx`
        sys.path.insert(0, _cand)                    # src/   → enables `from base_dataset import`
        break

from base_dataset import BeatsDataset, BalancedGeneratorSampler
from base_ptln import BasePTLN
from beats.model_beat import model_beat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Default hyperparameters (matching the paper's best config)
# ─────────────────────────────────────────────────────────────────────────────
STAGE1_DEFAULTS = dict(
    num_train_epochs=15,
    learning_rate=3e-7,      # backbone LR; heads get 10x
    ce_w=3.0,                # cross-entropy loss weight
    ce_real_w=2.5,           # real-class weight in CE
    ccl_real_w=4.0,          # real-class weight in CCL
)
STAGE2_DEFAULTS = dict(
    num_train_epochs=8,
    learning_rate=1.5e-7,
    ce_w=3.0,
    ce_real_w=4.0,           # increased from 2.5 → stronger real separation
    ccl_real_w=4.0,
)

# Balanced batch: 6 per fake generator + 24 real = 48 total
SAMPLES_PER_LABEL = {
    "fake_ata_01": 6,   # ATA-Audioldm1
    "fake_tta_01": 6,   # TTA-Audiogen
    "fake_tta_02": 6,   # TTA-Audioldm1
    "fake_tta_03": 6,   # TTA-Audioldm2
    "real": 24,
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="2-Stage BEATs training for LL_fake_only")

    p.add_argument("--stage", choices=["1", "2", "both"], default="both",
                   help="Training stage: 1, 2, or both (default: both)")
    p.add_argument("--train_json", default=os.path.join(_HERE, "data", "Event_train_stage1.json"),
                   help="Training data JSON (5-class: real + 4 fake generators)")
    p.add_argument("--val_json", default=os.path.join(_HERE, "data", "test_track2.json"),
                   help="Validation data JSON")
    p.add_argument("--load_ckpt_path", default=None,
                   help="Checkpoint for stage 2 (auto-detected when --stage both)")
    p.add_argument("--output_dir", default=os.path.join(_HERE, "checkpoint", "ll_fake_only"),
                   help="Base output directory (_stage1/_stage2 suffix added)")
    p.add_argument("--wandb_project", default="LL_fake_only")
    p.add_argument("--wandb_api", default=None, help="wandb API key")

    # Hardware
    p.add_argument("--devices", type=int, nargs="+", default=[0])
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader num_workers (0 for WSL)")
    p.add_argument("--val_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Override stage defaults (None = use stage default)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--num_train_epochs", type=int, default=None)
    p.add_argument("--ce_w", type=float, default=None)
    p.add_argument("--ce_real_w", type=float, default=None)
    p.add_argument("--ccl_real_w", type=float, default=None)

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    """BEATs backbone in 3-loss mode → 527-dim predictor embeddings."""
    return model_beat(num_label=5, three_loss=True, feature_layer="predictor")


def build_dataset(json_file):
    """BeatsDataset with 5-class labels, no augmentation."""
    class Args:
        num_label = 5
        three_loss = True
        audio_aug = False
        audio_mixup = False
        audio_aug_prob = 0
        audio_mixup_prob = 0
    return BeatsDataset(json_file=json_file, transformation=None, args=Args())


def build_train_dataloader(dataset, num_workers=0):
    """BalancedGeneratorSampler: 6 per fake gen + 24 real = 48/batch."""
    sampler = BalancedGeneratorSampler(
        dataset=dataset,
        samples_per_label=SAMPLES_PER_LABEL,
        cycle_short_labels=True,
    )
    return DataLoader(dataset, batch_sampler=sampler,
                      num_workers=num_workers, pin_memory=True)


def build_val_dataloader(json_file, batch_size=64, num_workers=0):
    dataset = build_dataset(json_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_stage(stage_num, args, stage_defaults, load_ckpt=None):
    """Run one training stage and return the best checkpoint path."""

    # Merge: CLI overrides > stage defaults
    lr         = args.learning_rate    or stage_defaults["learning_rate"]
    epochs     = args.num_train_epochs or stage_defaults["num_train_epochs"]
    ce_w       = args.ce_w       if args.ce_w       is not None else stage_defaults["ce_w"]
    ce_real_w  = args.ce_real_w  if args.ce_real_w  is not None else stage_defaults["ce_real_w"]
    ccl_real_w = args.ccl_real_w if args.ccl_real_w is not None else stage_defaults["ccl_real_w"]

    output_dir = f"{args.output_dir}_stage{stage_num}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"STAGE {stage_num}")
    logger.info(f"  LR={lr}  epochs={epochs}  ce_w={ce_w}  "
                f"ce_real_w={ce_real_w}  ccl_real_w={ccl_real_w}")
    logger.info(f"  output: {output_dir}")
    if load_ckpt:
        logger.info(f"  resume: {load_ckpt}")
    logger.info("=" * 60)

    # Model
    pipeline = build_model()
    if load_ckpt:
        ckpt = torch.load(load_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        pipeline_state = {k.replace("pipeline.", ""): v
                          for k, v in state_dict.items() if k.startswith("pipeline.")}
        pipeline.load_state_dict(pipeline_state)
        logger.info("Checkpoint loaded.")

    # Data
    train_ds = build_dataset(args.train_json)
    train_dl = build_train_dataloader(train_ds, num_workers=args.num_workers)
    val_dl = build_val_dataloader(args.val_json, batch_size=args.val_batch_size,
                                  num_workers=args.num_workers)

    # Lightning module (BasePTLN expects an args namespace)
    class TrainArgs:
        pass
    ta = TrainArgs()
    ta.learning_rate = lr
    ta.adam_beta1 = 0.9
    ta.adam_beta2 = 0.999
    ta.adam_weight_decay = 0
    ta.max_grad_norm = 1.0
    ta.num_label = 5
    ta.three_loss = True
    ta.embed_dim = 527
    ta.ccl_real_w = ccl_real_w
    ta.ce_w = ce_w
    ta.ce_real_w = ce_real_w

    pl_module = BasePTLN(args=ta, pipeline=pipeline, test_dl=val_dl)

    # Trainer
    if args.wandb_api:
        os.system(f"wandb login --relogin {args.wandb_api}")

    wandb_logger = WandbLogger(
        project=args.wandb_project,
        name=f"stage{stage_num}_LR{lr}_ce{ce_w}_realW{ce_real_w}",
        log_model=False,
    )
    # Monitor test/eer logged by BasePTLN.on_train_epoch_end().
    # validation_step for num_label=5 has a known bug (uses undefined attributes)
    # so we cannot rely on valid/loss as the monitor metric.
    checkpoint_cb = ModelCheckpoint(
        save_top_k=5, monitor="test/eer", mode="min",
        dirpath=output_dir, filename="sample-{epoch:02d}",
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

    best = checkpoint_cb.best_model_path
    logger.info(f"Stage {stage_num} complete. Best: {best}")
    return best


def find_best_stage1_ckpt(output_dir):
    pattern = f"{output_dir}_stage1/sample-*.ckpt"
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(
            f"No stage 1 checkpoints found at {pattern}. "
            f"Run --stage 1 first or specify --load_ckpt_path.")
    return ckpts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    best_s1 = None
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
