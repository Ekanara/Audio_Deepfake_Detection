import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from typing import Any, Optional
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric

class BasePTLN(LightningModule):
    def __init__(self, args, sync_dist=None, pipeline=None):
        super().__init__()
        self.args = args
        self.pipeline=pipeline
        self.sync_dist = sync_dist
        self.mean_valid_loss = MeanMetric()
        for param in self.pipeline.parameters():
            param.requires_grad = True
    
    
    def training_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']
        # print(f"Model device: {next(self.parameters()).device}")
        # print(f"Input device: {batch['image'].device}")
        # print(f"Label device: {batch['label'].device}")
        # print(f"Current device: {self.device}")
        # model_pred, target, timesteps = self.forward(model_input, image, mask, original_size, crop_top_lefts, noise)
        #print(image.shape)
        model_pred = self.pipeline.forward_pipeline(image)
        loss = F.cross_entropy(model_pred, label)
        
        self.log("train/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']
        # print(f"Model device: {next(self.parameters()).device}")
        # print(f"Input device: {batch['image'].device}")
        # print(f"Label device: {batch['label'].device}")
        # print(f"Current device: {self.device}")
        # model_pred, target, timesteps = self.forward(model_input, image, mask, original_size, crop_top_lefts, noise)
        #print(image.shape)
        model_pred = self.pipeline.forward_pipeline(image)
        loss = F.cross_entropy(model_pred, label)
        
        self.mean_valid_loss.update(loss, weight=image.shape[0])
        
        return loss
    
    def on_validation_epoch_end(self):
        
        # TODO: if calculate the metric in validation_step, then log it here
        
        # log metrics
        self.log("valid/loss", self.mean_valid_loss.compute(), prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
        # reset metrics
        self.mean_valid_loss.reset()
        
    def test_step(self, batch, batch_idx):
        condition = batch['model_input']
        gt = batch['gt']
        encoder_hidden_states = batch['encoder_hidden_states']
        
        noise = torch.randn_like(gt)
        if self.args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.args.noise_offset * torch.randn(
                (gt.shape[0], gt.shape[1], 1, 1), device=gt.device
            )
        
        # model_pred, target, timesteps = self.forward(model_input, image, mask, original_size, crop_top_lefts, noise)
        model_pred, timesteps, noise_scheduler = self.pipeline.forward_pipeline(noise=noise, condition=condition, gt=gt, encoder_hidden_states = encoder_hidden_states)
        target = self.register_target(noise, noise_scheduler=noise_scheduler)
        loss = self.compute_loss(model_pred=model_pred, timesteps=timesteps, target=target, noise_scheduler=noise_scheduler)
        
        # Loss
        self.mean_valid_loss.update(loss, weight=gt.shape[0])
        
        # TODO: calculate Metric
        
    def on_test_epoch_end(self):
        
        # TODO: if calculate the metric in test_step, then log it here
        
        # log metrics
        self.log("test/loss", self.mean_valid_loss.compute(), prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
        # reset metrics
        self.mean_valid_loss.reset()
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.pipeline.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.adam_weight_decay,
            betas=(0.9, 0.999),  # standard for AdamW
        )
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.03 * total_steps),
            num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    # TODO: check save
    def save_checkpoint(self, filepath, weights_only: bool = False, storage_options:Optional[Any]=None) -> None:
        checkpoint = self._checkpoint_connector.dump_checkpoint(weights_only)
        self.strategy.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
        self.strategy.barrier("Trainer.save_checkpoint")
