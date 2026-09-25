import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve

from src.base_dataset import BaselineDataset, BeatsDataset
from src.inference._lib.loader import load_pipeline
from src.inference._lib.metrics import (
    compute_eer_from_score_arrays,
    compute_classification_metrics,
    create_metrics_summary,
    log_metrics_summary,
)
from src.inference._lib.plots import (
    plot_confusion_and_multi_roc,
    plot_detailed_metrics,
    plot_log_curve,
)
from src.utils.arguments import get_args

logger = logging.getLogger(__name__)


def predict_single(pipeline, input_tensors, device):
    with torch.no_grad():
        if len(input_tensors.shape) == 3:
            input_tensors = input_tensors.unsqueeze(0)
        input_tensors = input_tensors.to(device)
        logits = pipeline.forward_pipeline(input_tensors)
        probability = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probability, dim=1)
        confidence = torch.max(probability, dim=1)[0]

    return {
        "logits": logits,
        "probability": probability,
        "predicted_class": predicted_class,
        "confidence": confidence,
    }


def predicted_batch(pipeline, dataloader, device='cuda', prob_threshold=0.7):
    """Run inference, applying a custom probability threshold on P(fake).

    A sample is predicted real (1) iff P(fake) <= prob_threshold.
    If labels are present, also computes log-likelihood-ratio scores
    log10(P(real)/P(fake)) and EER on those scores.
    """
    all_probabilities = []
    all_predictions = []
    all_confidences = []
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
            probability = F.softmax(logits, dim=1)

            predicted_class = (probability[:, 0] <= prob_threshold).long()
            confidence = torch.max(probability, dim=1)[0]

            all_predictions.extend(predicted_class.cpu().numpy())
            all_probabilities.extend(probability.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_paths.extend(paths)

            if labels is not None:
                all_labels.extend(labels.cpu().numpy())

    results = {
        'paths': np.array(all_paths),
        'predictions': np.array(all_predictions),
        'probabilities': np.array(all_probabilities),
        'confidences': np.array(all_confidences),
        'prob_threshold': prob_threshold,
    }

    if all_labels:
        results['labels'] = np.array(all_labels)
        _augment_with_llr_scores(results, prob_threshold)

    return results


def _augment_with_llr_scores(results, prob_threshold):
    """Add log10(P(real)/P(fake)) scores + EER to results dict."""
    eps = 1e-10
    eval_pred = np.clip(results['probabilities'], eps, 1.0)
    llr_scores = np.log10(eval_pred[:, 1] / eval_pred[:, 0])
    llr_threshold = np.log10((1 - prob_threshold) / prob_threshold)

    fake_mask = results['labels'] == 0
    real_mask = results['labels'] == 1
    fake_arr = llr_scores[fake_mask]
    real_arr = llr_scores[real_mask]

    eer, fpr_eer, fnr_eer, _, org_thresh = compute_eer_from_score_arrays(fake_arr, real_arr)

    results['eer'] = eer
    results['eer_threshold'] = org_thresh
    results['fpr_eer'] = fpr_eer
    results['fnr_eer'] = fnr_eer
    results['llr_scores'] = llr_scores
    results['fake_scores'] = fake_arr
    results['real_scores'] = real_arr
    results['llr_threshold'] = llr_threshold

    logger.info(f"Using probability threshold: {prob_threshold:.2f}")
    logger.info(f"Corresponding LLR threshold: {llr_threshold:.4f}")
    logger.info(f"EER: {eer:.4f}, threshold: {org_thresh:.4f}")
    logger.info(f"LLR range: min={llr_scores.min():.4f}, max={llr_scores.max():.4f}")
    logger.info(f"Fake scores: mean={fake_arr.mean():.4f}, std={fake_arr.std():.4f}")
    logger.info(f"Real scores: mean={real_arr.mean():.4f}, std={real_arr.std():.4f}")


def compute_mahalanobis_scores(X_real, X_fake, eps=1e-6):
    """Mahalanobis distances of real/fake points relative to the bonafide cluster."""
    mu = np.mean(X_real, axis=0)
    cov = np.cov(X_real, rowvar=False)
    cov += eps * np.eye(cov.shape[0])

    c, lower = cho_factor(cov)

    def mahalanobis(x):
        diff = x - mu
        return np.sqrt(diff @ cho_solve((c, lower), diff))

    real_scores = np.array([mahalanobis(x) for x in X_real])
    fake_scores = np.array([mahalanobis(x) for x in X_fake])
    return fake_scores, real_scores


def evaluate_model(results, class_names=None):
    """Compute metrics + plots for binary deepfake-detection results."""
    if 'labels' not in results:
        logger.warning("No ground truth labels available for evaluation")
        return None

    metrics = compute_classification_metrics(
        labels=results['labels'],
        predictions=results['predictions'],
        probabilities=results['probabilities'],
        confidences=results['confidences'],
        class_names=class_names,
        include_eer=False,
    )
    # The inference.py flow uses LLR-based EER (already in results)
    metrics['eer'] = results.get('eer')
    metrics['eer_threshold'] = results.get('eer_threshold')

    log_metrics_summary(metrics)

    plot_confusion_and_multi_roc(
        labels=results['labels'],
        predictions=results['predictions'],
        probabilities=results['probabilities'],
        class_names=metrics['class_names'],
        auc_score=metrics.get('auc_macro') or 0,
        eer=metrics.get('eer'),
        save_path='evaluation_metrics.png',
        binary_roc_scores=results.get('llr_scores'),
    )

    if metrics.get('eer') is not None and 'fake_scores' in results:
        plot_log_curve(
            results['fake_scores'], results['real_scores'],
            metrics['eer'], metrics['eer_threshold'], "log_score.png",
        )

    return metrics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    device = 'cuda'
    args = get_args()
    class_names = ['fake', 'real']
    mode = args.mode

    if mode == 'beats':
        baseline_dataset = BeatsDataset(json_file=args.val_json_file, transformation=None, args=args)
    else:
        baseline_dataset = BaselineDataset(json_file=args.val_json_file, transformation=None, args=args)

    pipeline = load_pipeline(
        checkpoint_path=args.load_ckpt_path, mode=mode, num_classes=2,
        beats_feature=getattr(args, 'beats_feature', 'predictor'),
    )
    dataloader = torch.utils.data.DataLoader(
        baseline_dataset, batch_size=100, num_workers=16, shuffle=False,
    )

    results = predicted_batch(pipeline, dataloader, device=device, prob_threshold=0.7)

    if 'labels' in results:
        metrics = evaluate_model(results, class_names)

        print("=" * 50)
        print("KEY METRICS SUMMARY:")
        print(f"Probability Threshold: 0.728")
        print(f"LLR Threshold: {results.get('llr_threshold', float('nan')):.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        if metrics['auc_macro'] is not None:
            print(f"Macro AUC-ROC: {metrics['auc_macro']:.4f}")
        if metrics.get('eer') is not None:
            print(f"Equal Error Rate (EER): {metrics['eer']:.4f}")
            print(f"EER Threshold: {metrics['eer_threshold']:.4f}")
        print("=" * 50)

        plot_detailed_metrics(metrics, "my_model_metrics.png", show_eer_panel=True)
        create_metrics_summary(metrics, "my_model_summary.txt")

        if metrics.get('eer') is not None and 'fake_scores' in results:
            plot_log_curve(
                results['fake_scores'], results['real_scores'],
                metrics['eer'], metrics['eer_threshold'], "log_score.png",
                custom_threshold=None,
            )

    if 'labels' in results and 'paths' in results:
        labels = results['labels']
        fake_probs = results['probabilities'][:, 0]
        llr_scores = results['llr_scores']

        save_array = np.array(list(zip(labels, fake_probs, llr_scores)), dtype=object)
        np.save("path_fake_label_llr.npy", save_array, allow_pickle=True)
        logger.info("Saved labels, fake probabilities, and LLR scores to path_fake_label_llr.npy")
        data = np.load("path_fake_label_llr.npy", allow_pickle=True)
        print(data[:5])
