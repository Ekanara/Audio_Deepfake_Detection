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
from src.base_ptln import BasePTLN

class ENVITPTLN(BasePTLN):
    def __init__(self, args, sync_dist=None, pipeline=None):
        super().__init__(args=args, sync_dist=sync_dist, pipeline=pipeline)
    
    
    def training_step(self, batch, batch_idx):
        audio = batch['audio']
        label = batch['label']
        model_pred = self.pipeline.forward(audio)
        loss = F.cross_entropy(model_pred, label)
        self.log("train/loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.sync_dist)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        audio = batch['audio']
        label = batch['label']
        model_pred = self.pipeline.forward(audio)
        loss = F.cross_entropy(model_pred, label)
        
        self.mean_valid_loss.update(loss, weight=audio.shape[0])
        
        return loss
    
    def on_validation_epoch_end(self):
        
        # TODO: if calculate the metric in validation_step, then log it here
        
        # log metrics
        self.log("valid/loss", self.mean_valid_loss.compute(), prog_bar=True, sync_dist=self.sync_dist, logger=True)
        
        # reset metrics
        self.mean_valid_loss.reset()

