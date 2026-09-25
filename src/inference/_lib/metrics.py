import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_eer(labels, scores, pos_label=1):
    """Equal Error Rate where FAR (FPR) ≈ FRR (FNR).

    Args:
        labels: binary ground-truth (1 = target/real, 0 = non-target/fake)
        scores: continuous score; higher = more likely target
        pos_label: positive class value
    Returns:
        eer, threshold, fpr, fnr (the latter two for plotting)
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=pos_label)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    threshold = thresholds[eer_idx]
    return eer, threshold, fpr, fnr


def compute_eer_from_score_arrays(fake_scores, real_scores):
    """EER from separated score arrays (used by LLR-style scoring).

    Returns: (eer, fpr_at_eer, fnr_at_eer, thresholds, threshold_at_eer)
    """
    scores = np.concatenate([fake_scores, real_scores])
    labels = np.concatenate([np.zeros(len(fake_scores)), np.ones(len(real_scores))])
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return eer, fpr[eer_idx], fnr[eer_idx], thresholds, thresholds[eer_idx]


def apply_threshold(results, threshold=0.5):
    """Re-derive predictions / confidences from stored probabilities.

    Sample → real (1) iff P(real) >= threshold, else fake (0).
    Mutates `results` in place and returns it.
    """
    prob_real = results['probabilities'][:, 1]
    prob_fake = results['probabilities'][:, 0]
    predictions = (prob_real >= threshold).astype(int)
    confidence = np.where(predictions == 1, prob_real, prob_fake)

    results['predictions'] = predictions
    results['confidences'] = confidence
    results['threshold'] = threshold
    logger.info(f"Applied decision threshold = {threshold:.4f}")
    return results


def compute_classification_metrics(labels, predictions, probabilities,
                                    confidences, class_names=None,
                                    include_eer=False, eer_score_key=None,
                                    eer_scores=None):
    """Compute scalar/per-class metrics + optional EER.

    Args:
        labels, predictions, probabilities, confidences: arrays from inference
        class_names: list of class names; defaults to Class_0, Class_1, ...
        include_eer: if True, also compute EER
        eer_scores: array of scores for EER (overrides probabilities[:, 1])
    Returns:
        metrics dict suitable for plot_detailed_metrics / create_metrics_summary
    """
    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    num_classes = len(np.unique(labels))
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    if num_classes == 2:
        auc_score = roc_auc_score(labels, probabilities[:, 1])
        auc_macro = auc_weighted = auc_score
    else:
        try:
            auc_macro = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted')
        except ValueError as e:
            logger.warning(f"Could not calculate AUC: {e}")
            auc_macro = np.nan
            auc_weighted = np.nan

    eer = eer_threshold = None
    if include_eer:
        scores_for_eer = eer_scores if eer_scores is not None else probabilities[:, 1]
        eer, eer_threshold, _, _ = compute_eer(labels, scores_for_eer, pos_label=1)

    cm = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions, target_names=class_names)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'auc_macro': auc_macro if not np.isnan(auc_macro) else None,
        'auc_weighted': auc_weighted if not np.isnan(auc_weighted) else None,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'confusion_matrix': cm,
        'classification_report': report,
        'avg_confidence': float(np.mean(confidences)),
        'class_names': class_names,
    }


def log_metrics_summary(metrics):
    """Log overall + per-class metrics at INFO level."""
    logger.info("=== Overall Performance Metrics ===")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Average Confidence: {metrics['avg_confidence']:.4f}")
    if metrics.get('eer') is not None:
        logger.info(f"EER: {metrics['eer'] * 100:.2f}%  (threshold={metrics['eer_threshold']:.4f})")
    logger.info(f"Macro Precision: {metrics['precision_macro']:.4f}")
    logger.info(f"Macro Recall: {metrics['recall_macro']:.4f}")
    logger.info(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    logger.info(f"Weighted Precision: {metrics['precision_weighted']:.4f}")
    logger.info(f"Weighted Recall: {metrics['recall_weighted']:.4f}")
    logger.info(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    if metrics['auc_macro'] is not None:
        logger.info(f"Macro AUC-ROC: {metrics['auc_macro']:.4f}")
        logger.info(f"Weighted AUC-ROC: {metrics['auc_weighted']:.4f}")

    logger.info("\n=== Per-Class Metrics ===")
    logger.info(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    logger.info("-" * 60)
    for i, cn in enumerate(metrics['class_names']):
        logger.info(
            f"{cn:<15} {metrics['per_class_precision'][i]:<10.4f} "
            f"{metrics['per_class_recall'][i]:<10.4f} "
            f"{metrics['per_class_f1'][i]:<10.4f} "
            f"{metrics['per_class_support'][i]:<10}"
        )

    logger.info("\nDetailed Classification Report:")
    logger.info(f"\n{metrics['classification_report']}")


def create_metrics_summary(metrics, save_path="metrics_summary.txt", title="MODEL EVALUATION SUMMARY"):
    """Write a plain-text summary of metrics to disk."""
    if metrics is None:
        return None

    lines = ["=" * 60, title, "=" * 60, ""]
    lines.append("OVERALL PERFORMANCE:")
    lines.append(f"  Accuracy:           {metrics['accuracy']:.4f}")
    lines.append(f"  Average Confidence: {metrics['avg_confidence']:.4f}")
    if metrics.get('eer') is not None:
        lines.append(f"  Equal Error Rate:   {metrics['eer']:.4f}")
        lines.append(f"  EER Threshold:      {metrics['eer_threshold']:.4f}")
    lines.append("")

    lines.append("MACRO AVERAGES:")
    lines.append(f"  Precision: {metrics['precision_macro']:.4f}")
    lines.append(f"  Recall:    {metrics['recall_macro']:.4f}")
    lines.append(f"  F1-Score:  {metrics['f1_macro']:.4f}")
    if metrics['auc_macro'] is not None:
        lines.append(f"  AUC-ROC:   {metrics['auc_macro']:.4f}")
    lines.append("")

    lines.append("WEIGHTED AVERAGES:")
    lines.append(f"  Precision: {metrics['precision_weighted']:.4f}")
    lines.append(f"  Recall:    {metrics['recall_weighted']:.4f}")
    lines.append(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
    if metrics['auc_weighted'] is not None:
        lines.append(f"  AUC-ROC:   {metrics['auc_weighted']:.4f}")
    lines.append("")

    lines.append("PER-CLASS BREAKDOWN:")
    lines.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    lines.append("-" * 60)
    for i, cn in enumerate(metrics['class_names']):
        lines.append(
            f"{cn:<15} {metrics['per_class_precision'][i]:<10.4f} "
            f"{metrics['per_class_recall'][i]:<10.4f} "
            f"{metrics['per_class_f1'][i]:<10.4f} "
            f"{metrics['per_class_support'][i]:<10}"
        )
    lines.append("")

    text = "\n".join(lines)
    with open(save_path, 'w') as f:
        f.write(text)
    logger.info(f"Metrics summary saved to: {save_path}")
    return text
