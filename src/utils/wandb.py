
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import Trainer
from src.utils.arguments import get_args
# from asuka.modeling.mae_module import MAE_Module
from datasets import load_from_disk

import logging
import torch
import os
import os

def create_trainer(args):
    os.system(f"wandb login --relogin {args.wandb_api}")
    wandb_logger = WandbLogger(
        project=args.wandb_run_name,
        log_model=False,
    )
    # create callback function
    model_checkpoint = ModelCheckpoint(
        save_top_k=args.save_top_k,
        monitor="valid/loss",
        mode="min",
        dirpath=args.output_dir,
        filename="sample-{epoch:02d}-{valid/loss:.2f}",
        save_weights_only=False,
    )

    dist = True if len(args.devices) > 1 else False

    # create trainer
    trainer = Trainer(
        max_epochs=args.num_train_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[model_checkpoint],
        strategy="ddp_find_unused_parameters_true" if dist else "auto",
        log_every_n_steps=args.log_steps,
        logger=wandb_logger,
        precision=args.precision,
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )
    device = trainer.global_rank if dist else 0
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

    return trainer, device, dist

def save_checkpoint(trainer, args):
    save_ckpt_path = f'{args.save_ckpt_path}/checkpoint'
    os.makedirs(save_ckpt_path, exist_ok=True)
    saved_ckpt_path = f'{save_ckpt_path}/best.pt'
    trainer.save_checkpoint(saved_ckpt_path)

def load_from_disk_(dataset_name, split = ['train', 'val', 'test']):
    dataset_name = dataset_name.split("/")[-1]
    dataset_path = f"{dataset_name}/{split}"
    dataset = load_from_disk(dataset_path)
    return dataset