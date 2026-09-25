import logging

import torch
import torch.nn as nn
from torchvision.models import densenet161, efficientnet_b1, inception_v3, resnet50, vgg16, mobilenet_v2, convnext_tiny
import timm

from base_dataset import (
    BaselineDataset,
    BeatsDataset,
    BalancedGeneratorSampler,
    RatioSampler,
    mixup_collate_fn,
)
from base_pipeline import BasePipeline
from base_ptln import BasePTLN
try:
    from src.beats.model_beat import model_beat
except ModuleNotFoundError:
    from beats.model_beat import model_beat
from utils.arguments import get_args
from utils.wandb import create_trainer, save_checkpoint

logger = logging.getLogger(__name__)


SAMPLES_PER_LABEL = {
    5: {"fake_ata_01": 6, "fake_tta_01": 6, "fake_tta_02": 6, "fake_tta_03": 6, "real": 24},
    2: {"fake": 16, "real": 16},
}


def build_model(mode: str) -> nn.Module:
    builders = {
        "efficientnet": lambda: efficientnet_b1(weights=None),
        "resnet": lambda: resnet50(weights=None),
        "inception": lambda: inception_v3(weights=None, aux_logits=False),
        "densenet": lambda: densenet161(weights=None),
        "vgg": lambda: vgg16(weights=None),
        "mobilenet": lambda: mobilenet_v2(weights=None),
        "convnext": lambda: convnext_tiny(weights=None),
        "xception": lambda: timm.create_model("xception", pretrained=False),
        "nasnet": lambda: timm.create_model("nasnetalarge", pretrained=False),
    }
    if mode not in builders:
        raise ValueError(f"Unsupported vision mode: {mode!r}")
    return builders[mode]()


def build_pipeline(args, num_label: int, three_loss: bool) -> nn.Module:
    if args.mode == "beats":
        return model_beat(
            num_label=num_label,
            three_loss=three_loss,
            feature_layer=getattr(args, "beats_feature", "predictor"),
        )

    model = build_model(args.mode)
    model.train()
    return BasePipeline(
        args=args,
        model=model,
        device="cuda",  # moved to device by Lightning
        in_features=3,
        num_label=num_label,
        mode=args.mode,
    )


def replace_classifier_head(pipeline: nn.Module, mode: str, new_num_classes: int, device):
    replacements = {
        "efficientnet": lambda p: setattr(
            p.model.classifier,
            "1",
            nn.Linear(p.model.classifier[1].in_features, new_num_classes).to(device),
        ),
        "resnet": lambda p: setattr(
            p.model,
            "fc",
            nn.Linear(p.model.fc.in_features, new_num_classes).to(device),
        ),
        "inception": lambda p: setattr(
            p.model,
            "fc",
            nn.Linear(p.model.fc.in_features, new_num_classes).to(device),
        ),
        "densenet": lambda p: setattr(
            p.model,
            "classifier",
            nn.Linear(p.model.classifier.in_features, new_num_classes).to(device),
        ),
        "vgg": lambda p: setattr(
            p.model.classifier,
            "6",
            nn.Linear(p.model.classifier[6].in_features, new_num_classes).to(device),
        ),
        "mobilenet": lambda p: setattr(
            p.model.classifier,
            "1",
            nn.Linear(p.model.classifier[1].in_features, new_num_classes).to(device),
        ),
        "convnext": lambda p: setattr(
            p.model.classifier,
            "2",
            nn.Linear(p.model.classifier[2].in_features, new_num_classes).to(device),
        ),
        "xception": lambda p: setattr(
            p.model,
            "fc",
            nn.Linear(p.model.fc.in_features, new_num_classes).to(device),
        ),
        "nasnet": lambda p: setattr(
            p.model,
            "last_linear",
            nn.Linear(p.model.last_linear.in_features, new_num_classes).to(device),
        ),
        "beats": lambda p: setattr(
            p,
            "last_layer",
            nn.Linear(527, new_num_classes).to(device),
        ),
    }
    if mode not in replacements:
        raise ValueError(f"Unsupported mode for head replacement: {mode!r}")
    replacements[mode](pipeline)


def build_dataset(args, json_file: str):
    dataset_cls = BeatsDataset if args.mode == "beats" else BaselineDataset
    return dataset_cls(json_file=json_file, transformation=None, args=args)


def build_train_dataloader(args, dataset):
    if args.mixup:
        logger.info("Applying MixUp")
        sampler = RatioSampler(dataset, batch_size=args.train_batch_size)
        collate_fn = mixup_collate_fn
    else:
        sampler = BalancedGeneratorSampler(dataset, SAMPLES_PER_LABEL[args.num_label])
        collate_fn = None

    num_workers = args.dataloader_num_workers
    return torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_eval_dataloader(args, json_file: str):
    dataset = build_dataset(args, json_file)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.val_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
    )


def run_trainer(trainer, lightning_module, train_dl, eval_dl, args):
    try:
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dl,
            val_dataloaders=eval_dl,
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        save_checkpoint(trainer=trainer, args=args)


def do_train(args, trainer, device, dist, eval_dl=None):
    logger.info("*** Stage 1 training ***")
    dataset = build_dataset(args, args.train_json_file)
    train_dl = build_train_dataloader(args, dataset)

    pipeline = build_pipeline(args, num_label=args.num_label, three_loss=args.three_loss)
    pipeline.train()

    test_dl = build_eval_dataloader(args, args.val_json_file)
    logger.info("Per-epoch test dataloader built from: %s", args.val_json_file)

    lightning_module = BasePTLN(args=args, pipeline=pipeline, sync_dist=dist, test_dl=test_dl)
    run_trainer(trainer, lightning_module, train_dl, eval_dl, args)


def do_train_attention(args, trainer, device, dist, eval_dl=None):
    logger.info("*** Attention head fine-tuning ***")

    pipeline = model_beat(
        num_label=args.num_label,
        three_loss=args.three_loss,
        feature_layer=getattr(args, "beats_feature", "predictor"),
    )
    checkpoint = torch.load(args.load_ckpt_path, map_location=device)
    checkpoint_state = checkpoint.get("state_dict", checkpoint)

    old_state_dict = {
        key[9:]: value
        for key, value in checkpoint_state.items()
        if key.startswith("pipeline.")
    }
    if not old_state_dict:
        old_state_dict = checkpoint_state

    new_state_dict = pipeline.state_dict()
    compatible_state = {
        key: value
        for key, value in old_state_dict.items()
        if key in new_state_dict and value.shape == new_state_dict[key].shape
    }

    skipped = sorted(set(old_state_dict) - set(compatible_state))
    logger.info("Loaded : %d keys", len(compatible_state))
    logger.info("Skipped: %d keys (shape mismatch or new layers)", len(skipped))
    if skipped:
        logger.info("Skipped keys: %s", skipped)

    pipeline.load_state_dict(compatible_state, strict=False)
    logger.info("Checkpoint loaded - new layers remain randomly initialized")

    for param in pipeline.parameters():
        param.requires_grad = False

    trainable_roots = {"temporal_attention", "last_layer", "softmax_head", "contrastive_head"}
    for name, param in pipeline.named_parameters():
        if name.split(".")[0] in trainable_roots:
            param.requires_grad = True

    frozen = [name for name, param in pipeline.named_parameters() if not param.requires_grad]
    trainable = [name for name, param in pipeline.named_parameters() if param.requires_grad]
    logger.info("Frozen   : %d params", len(frozen))
    logger.info("Trainable: %d params", len(trainable))

    pipeline.train()

    dataset = build_dataset(args, args.train_json_file)
    train_dl = build_train_dataloader(args, dataset)
    lightning_module = BasePTLN(args=args, pipeline=pipeline, sync_dist=dist)
    run_trainer(trainer, lightning_module, train_dl, eval_dl, args)


def do_train_stage2(args, trainer, device, dist, eval_dl=None):
    logger.info("*** Stage 2 fine-tuning ***")

    pipeline = build_pipeline(args, num_label=args.prev_num_label, three_loss=args.three_loss)
    pipeline.train()
    lightning_module = BasePTLN(args=args, pipeline=pipeline, sync_dist=dist)

    checkpoint = torch.load(args.load_ckpt_path, map_location=device)
    checkpoint_state = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key: value
        for key, value in checkpoint_state.items()
        if "asoftmax_loss.weight" not in key and "asoftmax_loss.bias" not in key
    }
    lightning_module.load_state_dict(state_dict, strict=False)
    logger.info("Loaded stage-1 weights")

    replace_classifier_head(lightning_module.pipeline, args.mode, args.num_label, device)

    dataset = build_dataset(args, args.train_json_file)
    train_dl = build_train_dataloader(args, dataset)

    if args.do_eval and eval_dl:
        logger.info("Validation dataloader length: %d", len(eval_dl))

    stage2_trainer, _, _ = create_trainer(args)
    run_trainer(stage2_trainer, lightning_module, train_dl, eval_dl, args)


def main(args):
    if not any([args.do_train, args.do_train_attention, args.do_train_stage2, args.do_eval]):
        raise ValueError(
            "At least one of `do_train`, `do_train_attention`, `do_train_stage2`, or `do_eval` must be True."
        )

    trainer, device, dist = create_trainer(args)
    torch.cuda.empty_cache()

    eval_dl = build_eval_dataloader(args, args.val_json_file) if args.do_eval else None

    if args.do_train:
        do_train(args, trainer, device, dist, eval_dl)

    if args.do_train_attention:
        do_train_attention(args, trainer, device, dist, eval_dl)

    if args.do_train_stage2:
        do_train_stage2(args, trainer, device, dist, eval_dl)


if __name__ == "__main__":
    opt = get_args()
    logger.info("*** Training mode ***")
    main(opt)
