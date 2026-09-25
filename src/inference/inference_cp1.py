import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.base_dataset import BaselineDataset, BeatsDataset
from src.inference._lib.generator_score import (
    assign_generator_index,
    compute_per_generator_scores,
    save_generator_scores,
)
from src.inference._lib.loader import load_pipeline
from src.inference._lib.metrics import (
    apply_threshold,
    compute_classification_metrics,
    compute_eer,
    create_metrics_summary,
    log_metrics_summary,
)
from src.inference._lib.plots import (
    plot_confusion_and_multi_roc,
    plot_detailed_metrics,
    plot_eer_curve,
)
from src.utils.arguments import get_args

logger = logging.getLogger(__name__)


def predicted_batch(pipeline, dataloader, device='cuda', real_class_idx=0, threshold=0.5):
    """Run multi-class inference and collapse to binary (real / fake).

    Stores both multi-class and binary probabilities in the result dict so
    downstream code can sweep thresholds via apply_threshold without re-running.
    """
    all_probs_multi = []
    all_preds_multi = []
    all_probs_binary = []
    all_labels = []
    all_paths = []

    pipeline = pipeline.to(device)
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['audio'].to(device, dtype=torch.float32)
            labels = batch['label']
            paths = batch['path']

            logits = pipeline.forward_pipeline(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]

            probs_multi = F.softmax(logits, dim=1)
            preds_multi = torch.argmax(probs_multi, dim=1)

            prob_real = probs_multi[:, real_class_idx].cpu().numpy()
            prob_fake = 1.0 - prob_real
            probs_binary = np.stack([prob_fake, prob_real], axis=1)

            all_preds_multi.extend(preds_multi.cpu().numpy())
            all_probs_multi.extend(probs_multi.cpu().numpy())
            all_probs_binary.extend(probs_binary)
            all_paths.extend(paths)

            if labels is not None:
                all_labels.extend(labels.cpu().numpy())

    results = {
        'paths': np.array(all_paths),
        'predictions_multiclass': np.array(all_preds_multi),
        'probabilities': np.array(all_probs_binary),
        'probabilities_multiclass': np.array(all_probs_multi),
    }
    if all_labels:
        results['labels'] = np.array(all_labels)

    apply_threshold(results, threshold=threshold)
    return results


def evaluate_model(results, class_names=None):
    """Compute metrics with EER and emit detailed plots."""
    if 'labels' not in results:
        logger.warning("No ground truth labels available for evaluation")
        return None

    metrics = compute_classification_metrics(
        labels=results['labels'],
        predictions=results['predictions'],
        probabilities=results['probabilities'],
        confidences=results['confidences'],
        class_names=class_names,
        include_eer=True,
    )
    log_metrics_summary(metrics)

    plot_eer_curve(
        labels=results['labels'],
        prob_real=results['probabilities'][:, 1],
        eer=metrics['eer'],
        eer_threshold=metrics['eer_threshold'],
        save_path="eer_curve.png",
    )
    plot_confusion_and_multi_roc(
        labels=results['labels'],
        predictions=results['predictions'],
        probabilities=results['probabilities'],
        class_names=metrics['class_names'],
        auc_score=metrics.get('auc_macro') or 0,
        eer=metrics.get('eer'),
        save_path='evaluation_metrics.png',
    )
    return metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = 'cuda'
    args = get_args()
    class_names = ['fake', 'real']
    real_class_idx = 4  # MODIFY THIS based on your model's class ordering

    # ── Threshold configuration ───────────────────────────────────────────────
    # --threshold accepts a float in [0,1] OR the literal 'eer'.
    raw_threshold_arg = getattr(args, 'threshold', '0.5')
    use_eer_threshold = str(raw_threshold_arg).strip().lower() == 'eer'
    initial_threshold = 0.5

    mode = args.mode
    json_file = args.val_json_file
    if mode == 'beats':
        baseline_dataset = BeatsDataset(json_file=json_file, transformation=None, args=args)
    else:
        baseline_dataset = BaselineDataset(json_file=json_file, transformation=None, args=args)

    pipeline = load_pipeline(
        checkpoint_path=args.load_ckpt_path, mode=mode,
        num_classes=5, truncate_layers=2,
        beats_feature=getattr(args, 'beats_feature', 'predictor'),
    )

    dataloader = torch.utils.data.DataLoader(
        baseline_dataset, batch_size=100, num_workers=16, shuffle=False,
    )

    results = predicted_batch(
        pipeline, dataloader, device=device,
        real_class_idx=real_class_idx, threshold=initial_threshold,
    )

    if use_eer_threshold and 'labels' in results:
        eer_val, eer_thr, _, _ = compute_eer(results['labels'], results['probabilities'][:, 1])
        logger.info(f"EER threshold mode: using threshold = {eer_thr:.4f} (EER = {eer_val*100:.2f}%)")
        print(f"[threshold=eer]  EER = {eer_val*100:.2f}%  →  setting threshold to {eer_thr:.4f}")
        apply_threshold(results, threshold=eer_thr)

    active_threshold = results.get('threshold', initial_threshold)
    print(f"Active decision threshold: {active_threshold:.4f}")

    if 'labels' in results:
        metrics = evaluate_model(results, class_names)

        print("=" * 50)
        print("KEY METRICS SUMMARY:")
        print(f"Threshold:     {active_threshold:.4f}")
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"EER:           {metrics['eer'] * 100:.2f}%")
        print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
        print(f"Macro F1:      {metrics['f1_macro']:.4f}")
        print(f"Weighted F1:   {metrics['f1_weighted']:.4f}")
        if metrics['auc_macro'] is not None:
            print(f"Macro AUC-ROC: {metrics['auc_macro']:.4f}")
        print("=" * 50)

        plot_detailed_metrics(metrics, "my_model_metrics.png")
        create_metrics_summary(metrics, "my_model_summary.txt")

        labels = results['labels']
        confidences = results['confidences']
        predictions = results['predictions']
        paths = results['paths']

        real_conf = confidences[labels == 1]
        fake_conf = confidences[labels == 0]

        print("=" * 50)
        print("Confidence Analysis:")
        print(f"Avg confidence (Real): {real_conf.mean():.4f}")
        print(f"Avg confidence (Fake): {fake_conf.mean():.4f}")
        print(f"Min/Max (Real): {real_conf.min():.4f} / {real_conf.max():.4f}")
        print(f"Min/Max (Fake): {fake_conf.min():.4f} / {fake_conf.max():.4f}")
        print("=" * 50)

        df = pd.DataFrame({
            "path": paths,
            "label": ["real" if l == 1 else "fake" for l in labels],
            "prediction": ["real" if p == 1 else "fake" for p in predictions],
            "prediction_multiclass": results['predictions_multiclass'],
            "confidence": confidences,
            "prob_fake": results['probabilities'][:, 0],
            "prob_real": results['probabilities'][:, 1],
            "threshold": active_threshold,
        })
        df.to_csv("per_audio_confidences.csv", index=False)
        print("Saved detailed per-audio confidence scores to per_audio_confidences.csv")

        df, generator_names, gen_to_idx = assign_generator_index(df, label_col="label")
        df, _, _ = compute_per_generator_scores(df, gen_to_idx, generator_names)
        save_generator_scores(df, csv_path="generator_score.csv", npy_path="generator_score.npy")
        print("Saved generator_score.csv and generator_score.npy")
    else:
        print("No ground truth labels available for evaluation")
