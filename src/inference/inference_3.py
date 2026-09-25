import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.base_dataset import BaselineDataset
from src.inference._lib.loader import load_pipeline
from src.utils.arguments import get_args

logger = logging.getLogger(__name__)


class SimpleEvalDataset(torch.utils.data.Dataset):
    """Loads pre-computed feature .npy files from a flat directory."""
    def __init__(self, audio_dir):
        self.audio_dir = audio_dir
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.npy')])
        logger.info(f"Found {len(self.audio_files)} .npy files in {audio_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        filename = self.audio_files[idx]
        filepath = os.path.join(self.audio_dir, filename)
        audio = np.load(filepath)
        audio = torch.from_numpy(audio).float()
        # Ensure shape is (C, H, W); add channel dim if needed.
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        return {'audio': audio, 'path': filepath}


def load_ensemble_pipelines(ensemble_configs, device='cuda', num_classes=2):
    """Load every pipeline listed in `ensemble_configs`."""
    pipelines = []
    for i, config in enumerate(ensemble_configs):
        logger.info(f"Loading model {i+1}/{len(ensemble_configs)}: "
                     f"{config['name']} ({config['mode']})")
        pipelines.append(load_pipeline(
            checkpoint_path=config['checkpoint'],
            mode=config['mode'],
            num_classes=num_classes,
            device=device,
            in_features=2,
            stage1=False,
        ))
    logger.info(f"Successfully loaded {len(pipelines)} models for ensemble")
    return pipelines


def predicted_batch_multi_preprocessing(pipelines, dataloaders, model_names,
                                         device='cuda', ensemble_method='average',
                                         modes=None):
    """Run each pipeline against its own preprocessing and combine outputs.

    Inception V3 inputs are upsampled 3x to match the architecture's expected size.
    `ensemble_method` is 'average' (mean of probabilities) or 'voting' (majority).
    """
    if modes is None:
        modes = ['resnet'] * len(pipelines)
        logger.warning("No modes provided, defaulting to 'resnet' for all models")

    all_probabilities = []
    all_predictions = []
    all_confidences = []
    all_paths = []

    dataloader_iters = [iter(dl) for dl in dataloaders]
    num_batches = len(dataloaders[0])

    logger.info(f"Processing {num_batches} batches with {len(pipelines)} models")
    logger.info(f"Modes: {modes}")

    with torch.no_grad():
        for batch_idx in range(num_batches):
            batches = []
            for dl_iter in dataloader_iters:
                try:
                    batches.append(next(dl_iter))
                except StopIteration:
                    logger.error(f"Dataloader ran out of data at batch {batch_idx}")
                    raise

            # Reference paths from the first dataloader (assumes aligned ordering).
            first = batches[0]
            paths = first['path'] if isinstance(first, dict) else (first[-1] if len(first) > 1 else first[0])

            batch_probs = []
            for pipeline, batch, mode in zip(pipelines, batches, modes):
                inputs = batch['audio'] if isinstance(batch, dict) else (
                    batch[0] if len(batch) > 1 else batch
                )
                inputs = inputs.squeeze(1).to(device, dtype=torch.float32)

                if mode == 'inception':
                    inputs = F.interpolate(inputs, scale_factor=3, mode='bilinear', align_corners=False)

                logits = pipeline.forward_pipeline(inputs)
                batch_probs.append(F.softmax(logits, dim=1))

            stacked_probs = torch.stack(batch_probs, dim=0)

            if ensemble_method == 'average':
                ensemble_prob = torch.mean(stacked_probs, dim=0)
            elif ensemble_method == 'voting':
                individual_preds = torch.argmax(stacked_probs, dim=2)
                ensemble_pred = torch.mode(individual_preds, dim=0)[0]
                ensemble_prob = F.one_hot(ensemble_pred, num_classes=stacked_probs.shape[2]).float()
            else:
                raise ValueError(f"Unknown ensemble method: {ensemble_method}")

            predicted_class = torch.argmax(ensemble_prob, dim=1)
            confidence = torch.max(ensemble_prob, dim=1)[0]

            all_predictions.extend(predicted_class.cpu().numpy())
            all_probabilities.extend(ensemble_prob.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_paths.extend(paths)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{num_batches} batches")

    return {
        'paths': np.array(all_paths),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'confidences': np.array(all_confidences),
    }


def create_eval_csv(results, output_path="eval_scores.csv"):
    """Write three CSVs: log-ratio scores, P(fake), P(real)."""
    paths = results['paths']
    probabilities = results['probabilities']

    filenames = [os.path.basename(path) for path in paths]
    prob_fake = probabilities[:, 0]
    prob_real = probabilities[:, 1]

    eps = 1e-10
    scores = np.log10(np.clip(prob_real, eps, 1.0) / np.clip(prob_fake, eps, 1.0))

    df_scores = pd.DataFrame({'file_name': filenames, 'score': scores})
    df_scores.to_csv(output_path, index=False)

    df_fake = pd.DataFrame({'file_name': filenames, 'prob_fake': prob_fake})
    fake_path = output_path.replace('.csv', '_prob_fake.csv')
    df_fake.to_csv(fake_path, index=False)

    df_real = pd.DataFrame({'file_name': filenames, 'prob_real': prob_real})
    real_path = output_path.replace('.csv', '_prob_real.csv')
    df_real.to_csv(real_path, index=False)

    logger.info("=" * 60)
    logger.info("Evaluation CSVs saved:")
    logger.info(f"  1. Main scores:        {output_path}")
    logger.info(f"  2. Fake probabilities: {fake_path}")
    logger.info(f"  3. Real probabilities: {real_path}")
    logger.info(f"Total files: {len(df_scores)}")
    logger.info(f"Score (log10 P(real)/P(fake)): mean={scores.mean():.4f}, "
                 f"median={np.median(scores):.4f}, std={scores.std():.4f}")
    logger.info(f"P(fake): mean={prob_fake.mean():.4f}, median={np.median(prob_fake):.4f}")
    logger.info(f"P(real): mean={prob_real.mean():.4f}, median={np.median(prob_real):.4f}")
    logger.info("=" * 60)

    return df_scores, df_fake, df_real


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = 'cuda'
    args = get_args()

    # ── Configuration ─────────────────────────────────────────────────────────
    USE_SIMPLE_LOADER = True       # True → load .npy files directly from a dir
    AUDIO_DIR = 'data/test/11'     # Used only when USE_SIMPLE_LOADER is True

    ensemble_configs = [
        {'name': 'gam_inception',    'mode': 'inception',    'checkpoint': 'checkpoint/Inceptionv3_Gam_stage4_5_epoch_new/checkpoint/best.pt',     'json_file': 'data/audio_labels_gam_eval.json'},
        {'name': 'gam_resnet',       'mode': 'resnet',       'checkpoint': 'checkpoint/Resnet50_Gam_stage4_5_epoch_new/checkpoint/best.pt',         'json_file': 'data/audio_labels_gam_eval.json'},
        {'name': 'gam_efficientnet', 'mode': 'efficientnet', 'checkpoint': 'checkpoint/Efficientnetb1_Gam_stage4_5epoch_new/checkpoint/best.pt',    'json_file': 'data/audio_labels_gam_eval.json'},
        {'name': 'mel_inception',    'mode': 'inception',    'checkpoint': 'checkpoint/Inceptionv3_Mel_stage4_5_epoch_new/checkpoint/best.pt',     'json_file': 'data/audio_labels_gam_eval.json'},
        {'name': 'mel_resnet',       'mode': 'resnet',       'checkpoint': 'checkpoint/Resnet50_Mel_stage4_5_epoch_new/checkpoint/best.pt',         'json_file': 'data/audio_labels_gam_eval.json'},
        {'name': 'mel_efficientnet', 'mode': 'efficientnet', 'checkpoint': 'checkpoint/Efficientnetb1_Mel_stage4_5epoch_new/checkpoint/best.pt',    'json_file': 'data/audio_labels_gam_eval.json'},
    ]

    logger.info("=" * 60)
    logger.info("EVALUATION MODE - ENSEMBLE INFERENCE")
    logger.info("=" * 60)

    logger.info(f"\nLoading {len(ensemble_configs)} models:")
    for config in ensemble_configs:
        logger.info(f"  - {config['name']} ({config['mode']}): {config['checkpoint']}")

    pipelines = load_ensemble_pipelines(ensemble_configs, device=device)
    model_names = [c['name'] for c in ensemble_configs]
    modes = [c['mode'] for c in ensemble_configs]

    # ── Datasets ──────────────────────────────────────────────────────────────
    datasets = []
    dataloaders = []

    logger.info("\nCreating datasets and dataloaders...")
    if USE_SIMPLE_LOADER:
        logger.info(f"Using simple directory loader from: {AUDIO_DIR}")
        for i, model_name in enumerate(model_names):
            logger.info(f"Creating dataset {i+1}/{len(model_names)}: {model_name}")
            dataset = SimpleEvalDataset(audio_dir=AUDIO_DIR)
            dataloader = DataLoader(dataset, batch_size=200, num_workers=16, shuffle=False)
            datasets.append(dataset)
            dataloaders.append(dataloader)
            logger.info(f"  Dataset size: {len(dataset)} samples")
    else:
        json_files = [c['json_file'] for c in ensemble_configs]
        for i, (json_file, model_name) in enumerate(zip(json_files, model_names)):
            logger.info(f"Loading dataset {i+1}/{len(json_files)}: {model_name}")
            dataset = BaselineDataset(json_file=json_file, transformation=None, args=args)
            dataloader = DataLoader(dataset, batch_size=200, num_workers=16, shuffle=False)
            datasets.append(dataset)
            dataloaders.append(dataloader)
            logger.info(f"  Dataset size: {len(dataset)} samples")

    dataset_sizes = [len(ds) for ds in datasets]
    if len(set(dataset_sizes)) != 1:
        logger.error(f"Dataset size mismatch! Sizes: {dataset_sizes}")
        raise ValueError("All datasets must have the same number of samples")
    logger.info(f"\nAll datasets verified: {dataset_sizes[0]} samples each")

    logger.info("\nRunning ensemble inference...")
    results = predicted_batch_multi_preprocessing(
        pipelines=pipelines,
        dataloaders=dataloaders,
        model_names=model_names,
        device=device,
        ensemble_method='average',
        modes=modes,
    )

    logger.info(f"\nInference complete! Processed {len(results['predictions'])} samples")

    output_csv = "eval_scores.csv"
    df_scores, _, _ = create_eval_csv(results, output_path=output_csv)

    logger.info("First 10 rows of the evaluation file:")
    print(df_scores.head(10).to_string(index=False))

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output: {output_csv}")
    logger.info(f"Format: file_name, score   (score = log10(P(real)/P(fake)))")
    logger.info("  - Positive scores → predicted REAL")
    logger.info("  - Negative scores → predicted FAKE")
    logger.info("=" * 60)
