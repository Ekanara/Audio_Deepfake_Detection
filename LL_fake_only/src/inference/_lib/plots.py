import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_curve, roc_auc_score

logger = logging.getLogger(__name__)


def plot_detailed_metrics(metrics, save_path="detailed_metrics.png", show_eer_panel=False):
    """6-panel summary: per-class P/R/F1, support, confusion matrix, key metrics.

    If `show_eer_panel` is True, the bottom-right panel shows
    Accuracy / F1 / AUC / EER instead of Macro-vs-Weighted bars.
    """
    if metrics is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    class_names = metrics['class_names']

    axes[0, 0].bar(class_names, metrics['per_class_precision'])
    axes[0, 0].set_title('Precision per Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].bar(class_names, metrics['per_class_recall'])
    axes[0, 1].set_title('Recall per Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(0, 1)

    axes[0, 2].bar(class_names, metrics['per_class_f1'])
    axes[0, 2].set_title('F1-Score per Class')
    axes[0, 2].set_ylabel('F1-Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].set_ylim(0, 1)

    axes[1, 0].bar(class_names, metrics['per_class_support'])
    axes[1, 0].set_title('Support (Sample Count)')
    axes[1, 0].set_ylabel('Number of Samples')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    if show_eer_panel:
        names = ['Accuracy', 'F1-Score', 'AUC-ROC', 'EER']
        values = [
            metrics['accuracy'],
            metrics['f1_macro'],
            metrics.get('auc_macro') or 0,
            metrics.get('eer') or 0,
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        bars = axes[1, 2].bar(names, values, color=colors)
        axes[1, 2].set_title('Key Performance Metrics')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_ylim(0, 1)
        for bar in bars:
            h = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width() / 2., h,
                            f'{h:.4f}', ha='center', va='bottom')
    else:
        names = ['Precision', 'Recall', 'F1-Score']
        macro = [metrics['precision_macro'], metrics['recall_macro'], metrics['f1_macro']]
        weighted = [metrics['precision_weighted'], metrics['recall_weighted'], metrics['f1_weighted']]
        x = np.arange(len(names))
        w = 0.35
        axes[1, 2].bar(x - w / 2, macro, w, label='Macro Average')
        axes[1, 2].bar(x + w / 2, weighted, w, label='Weighted Average')
        axes[1, 2].set_title('Macro vs Weighted Averages')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(names)
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Detailed metrics plot saved to: {save_path}")


def plot_confusion_and_roc(labels, predictions, prob_real, auc_score, eer, eer_threshold,
                           class_names, save_path="evaluation_metrics.png"):
    """Side-by-side confusion matrix + ROC curve with EER point."""
    from sklearn.metrics import confusion_matrix as _cm
    cm = _cm(labels, predictions, labels=[0, 1])
    has_both = len(np.unique(labels)) >= 2

    if has_both:
        fpr, tpr, _ = roc_curve(labels, prob_real)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14}, ax=axes[0])
    axes[0].set_title('Confusion Matrix', fontsize=14)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)

    axes[1].plot(fpr, tpr, color='steelblue',
                 label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    if has_both and not np.isnan(eer):
        eer_idx = int(np.argmin(np.abs(fpr - (1 - tpr))))
        axes[1].scatter([fpr[eer_idx]], [tpr[eer_idx]], color='red', zorder=5, s=80,
                        label=f'EER = {eer:.3f}')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate', fontsize=12)
    axes[1].set_ylabel('True Positive Rate', fontsize=12)
    axes[1].set_title('ROC Curves', fontsize=14)
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Evaluation metrics plot saved → {save_path}")


def plot_confusion_and_multi_roc(labels, predictions, probabilities, class_names,
                                  auc_score=None, eer=None, save_path="evaluation_metrics.png",
                                  binary_roc_scores=None):
    """Confusion matrix + ROC curve(s) for binary or multi-class.

    For binary tasks, `binary_roc_scores` (e.g. log-likelihood ratios) overrides
    the default `probabilities[:, 1]` when computing the ROC curve.
    """
    from sklearn.metrics import confusion_matrix as _cm
    cm = _cm(labels, predictions)
    num_classes = len(np.unique(labels))

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.subplot(1, 2, 2)
    if num_classes == 2:
        scores = binary_roc_scores if binary_roc_scores is not None else probabilities[:, 1]
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        if eer is not None:
            fnr = 1 - tpr
            eer_idx = int(np.argmin(np.abs(fpr - fnr)))
            plt.plot(fpr[eer_idx], tpr[eer_idx], 'ro', markersize=10,
                     label=f'EER = {eer:.3f}')
    else:
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        labels_bin = label_binarize(labels, classes=range(num_classes))
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
        for i, color in zip(range(num_classes), colors):
            if np.sum(labels_bin[:, i]) > 0:
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probabilities[:, i])
                roc_auc = roc_auc_score(labels_bin[:, i], probabilities[:, i])
                plt.plot(fpr, tpr, color=color,
                         label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Evaluation metrics plot saved to: {save_path}")


def plot_eer_curve(labels, prob_real, eer, eer_threshold, save_path="eer_curve.png"):
    """FAR / FRR curves with EER point marked."""
    fpr, tpr, thresholds = roc_curve(labels, prob_real, pos_label=1)
    fnr = 1.0 - tpr

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, fpr, label='FAR (False Acceptance Rate)', color='blue')
    plt.plot(thresholds, fnr, label='FRR (False Rejection Rate)',  color='red')
    plt.axvline(x=eer_threshold, color='green', linestyle='--',
                label=f'EER threshold = {eer_threshold:.4f}')
    plt.scatter([eer_threshold], [eer], color='green', zorder=5,
                label=f'EER = {eer * 100:.2f}%')
    plt.xlabel('Threshold  P(real)')
    plt.ylabel('Error Rate')
    plt.title('FAR / FRR Curve — Equal Error Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"EER curve saved → {save_path}")


def plot_score_distribution(labels, prob_real, save_path="score_distribution.png"):
    """Histogram of P(real) split by true label."""
    plt.figure(figsize=(8, 5))
    plt.hist(prob_real[labels == 0], bins=50, alpha=0.6, color='red',   label='Fake')
    plt.hist(prob_real[labels == 1], bins=50, alpha=0.6, color='green', label='Real')
    plt.xlabel('P(real) — score')
    plt.ylabel('Count')
    plt.title('Score Distribution: Real vs Fake')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Score distribution saved → {save_path}")


def plot_log_curve(fake_scores, real_scores, eer, threshold,
                   save_path="mahalanobis_scores.png", custom_threshold=None):
    """Mahalanobis distance histogram fit on bonafide, with EER marker."""
    fake_scores = np.asarray(fake_scores)
    real_scores = np.asarray(real_scores)

    if fake_scores.ndim == 1:
        fake_scores = fake_scores[:, None]
    if real_scores.ndim == 1:
        real_scores = real_scores[:, None]

    mu = np.mean(real_scores, axis=0)
    cov = np.atleast_2d(np.cov(real_scores, rowvar=False))
    cov += 1e-6 * np.eye(cov.shape[0])

    c, lower = cho_factor(cov)

    def mahalanobis(X):
        diff = X - mu
        return np.sqrt(np.einsum("ij,ij->i", diff, cho_solve((c, lower), diff.T).T))

    real_md = mahalanobis(real_scores)
    fake_md = mahalanobis(fake_scores)
    threshold_md = mahalanobis(np.atleast_2d(threshold))[0]

    if custom_threshold is not None:
        custom_threshold = mahalanobis(np.atleast_2d(custom_threshold))[0]

    plt.figure(figsize=(8, 6))
    plt.hist(fake_md, bins=60, alpha=0.7, label='spoof', edgecolor='black')
    plt.hist(real_md, bins=60, alpha=0.7, label='bonafide', edgecolor='black')
    plt.axvline(x=threshold_md, linestyle='--', linewidth=2, color='red',
                label=f'EER = {eer:.3%}')
    if custom_threshold is not None:
        plt.axvline(x=custom_threshold, linestyle=':', linewidth=2, color='blue',
                    label=f'Custom Threshold = {custom_threshold:.4f}')
    plt.xlabel('Mahalanobis distance')
    plt.ylabel('Number of audio samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_probability_distributions(results, save_path="probability_distributions.png"):
    """Histogram of P(Fake)|TrueFake and P(Real)|TrueReal."""
    if 'labels' not in results or 'probabilities' not in results:
        logger.warning("No labels or probabilities available for plotting")
        return

    probabilities = results['probabilities']
    labels = results['labels']
    true_fake_pfake = probabilities[labels == 0, 0]
    true_real_preal = probabilities[labels == 1, 1]

    def safe_bins(arr, n_bins=50):
        if len(arr) == 0:
            return 1
        if arr.max() - arr.min() == 0:
            logger.warning(f"Array has zero range (all values = {arr[0]:.4f}), using 1 bin.")
            return 1
        return n_bins

    plt.figure(figsize=(8, 6))
    plt.hist(true_fake_pfake, bins=safe_bins(true_fake_pfake), alpha=0.7,
             label='Fake', edgecolor='black')
    plt.hist(true_real_preal, bins=safe_bins(true_real_preal), alpha=0.7,
             label='Real', edgecolor='black')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Number of audio samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Probability distribution plot saved to: {save_path}")
    logger.info("=== Probability Distribution Statistics ===")
    if len(true_fake_pfake) > 0:
        logger.info(f"True Fake → P(Fake): mean={true_fake_pfake.mean():.4f}, std={true_fake_pfake.std():.4f}")
    if len(true_real_preal) > 0:
        logger.info(f"True Real → P(Real): mean={true_real_preal.mean():.4f}, std={true_real_preal.std():.4f}")
