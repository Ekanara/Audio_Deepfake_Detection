import logging

import torch
from torchvision.models import efficientnet_b1, inception_v3, resnet50, densenet161

from src.base_pipeline import BasePipeline
from src.beats.model_beat import model_beat

logger = logging.getLogger(__name__)


_BACKBONES = {
    'efficientnet': lambda: efficientnet_b1(pretrained=False),
    'resnet':       lambda: resnet50(weights=None),
    'densenet':     lambda: densenet161(weights=None),
    'inception':    lambda: inception_v3(weights=None, aux_logits=False),
}


def build_pipeline(mode, num_classes=2, in_features=3, device='cuda',
                   truncate_layers=0, three_loss=False, stage1=False,
                   beats_feature='predictor'):
    """Build a pipeline (untrained) for the given mode.

    Args:
        mode: 'beats' | 'efficientnet' | 'resnet' | 'densenet' | 'inception'
        num_classes: number of output classes
        in_features: input feature channels (used for image-backbone pipelines)
        device: target device
        truncate_layers: drop the last N BEATs encoder layers (beats only)
        three_loss: enable three-loss mode for BEATs (returns embedding tuple)
        stage1: stage1 flag passed to BasePipeline
        beats_feature: 'predictor' (527-dim) or 'encoder' (768-dim) — beats only
    """
    if mode == 'beats':
        pipeline = model_beat(
            num_label=num_classes,
            three_loss=three_loss,
            feature_layer=beats_feature,
        )
        if truncate_layers > 0:
            pipeline.BEATs.encoder.layers = pipeline.BEATs.encoder.layers[:-truncate_layers]
            logger.info(f"Truncated last {truncate_layers} BEATs encoder layers")
        return pipeline

    if mode not in _BACKBONES:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            f"Must be one of {sorted(list(_BACKBONES) + ['beats'])}"
        )

    backbone = _BACKBONES[mode]()
    pipeline = BasePipeline(
        model=backbone,
        args=None,
        device=device,
        in_features=in_features,
        num_label=num_classes,
        mode=mode,
        stage1=stage1,
    ).to(device)
    return pipeline


def load_pipeline(checkpoint_path, mode, num_classes=2, in_features=3,
                  device='cuda', truncate_layers=0, three_loss=False,
                  stage1=False, beats_feature='predictor'):
    """Build a pipeline and load weights from a Lightning or direct checkpoint.

    Lightning checkpoints expose weights under the 'pipeline.' prefix; direct
    checkpoints contain the raw state_dict.
    """
    logger.info(f"Loading pipeline from {checkpoint_path} (mode={mode})")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    pipeline = build_pipeline(
        mode=mode, num_classes=num_classes, in_features=in_features,
        device=device, truncate_layers=truncate_layers,
        three_loss=three_loss, stage1=stage1,
        beats_feature=beats_feature,
    )

    if 'state_dict' in checkpoint:
        pipeline_state = {
            k[len('pipeline.'):]: v
            for k, v in checkpoint['state_dict'].items()
            if k.startswith('pipeline.')
        }
        pipeline.load_state_dict(pipeline_state)
        logger.info("Pipeline loaded from Lightning checkpoint")
    else:
        pipeline.load_state_dict(checkpoint)
        logger.info("Pipeline loaded from direct checkpoint")

    pipeline.eval()
    return pipeline
