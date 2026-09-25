"""
infer_ll_fake_only.py — Log-Likelihood_fake_only Inference
==========================================================

Scores test audio samples by measuring their distance from the fake training
distribution in the model's 527-dim bonafide_head embedding space.

Pipeline:
  1. Load trained BEATs checkpoint
  2. Extract bonafide_head embeddings from all fake training samples
  3. Fit a multivariate Gaussian N(μ_fake, Σ_fake) on these embeddings
  4. For each test sample, compute score = -log p(x | N(μ_fake, Σ_fake))
     - High score → far from fake → classified as REAL
     - Low score  → close to fake → classified as FAKE
  5. Use EER threshold from ROC curve for binary classification

Usage:
------
  # Basic: score test_track2 against fake training distribution
  python scripts/infer_ll_fake_only.py

  # Custom checkpoint and test set
  python scripts/infer_ll_fake_only.py \
      --checkpoint checkpoint/my_model/sample-03.ckpt \
      --fake_ref data/label/beats/Event_train_stage1_fakeonly.json \
      --test_json data/label/beats/Event_test_5class.json \
      --output_dir inference_outputs/my_run

  # With plots (confusion matrix + ROC + score distributions)
  python scripts/infer_ll_fake_only.py --plot
"""
import argparse
import logging
import os
import sys

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from argparse import Namespace
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian fitting and scoring
# ─────────────────────────────────────────────────────────────────────────────
def fit_gaussian(X, eps=1e-6):
    """Fit a multivariate Gaussian to X.

    Returns (mu, cholesky_factor, lower_flag, log_determinant).
    Uses Cholesky decomposition for efficient scoring.
    """
    X = X.astype(np.float64)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    cf, lower = cho_factor(cov)
    _, logdet = np.linalg.slogdet(cov)
    return mu, cf, lower, logdet


def log_likelihood(X, gaussian):
    """Compute log p(x | N(mu, Sigma)) for each sample in X.

    log p(x) = -0.5 * [(x-mu)^T Sigma^{-1} (x-mu) + log|Sigma| + d*log(2pi)]
    The constant d*log(2pi) is omitted (doesn't affect ranking).
    """
    mu, cf, lower, logdet = gaussian
    diff = X.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    mahal_sq = np.einsum("ij,ij->i", diff, solved)
    return -0.5 * (mahal_sq + logdet)


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(pipeline, json_file, device, batch_size=512, num_workers=0):
    """Extract bonafide_head (527-dim) embeddings from all samples in a JSON dataset.

    Returns:
        embeddings: np.ndarray of shape (N, 527)
        labels: np.ndarray of shape (N,) — integer label indices
        label_map: dict mapping label_name -> label_index
    """
    ds_args = Namespace(
        num_label=5, three_loss=True,
        audio_aug=False, audio_mixup=False,
        audio_aug_prob=0, audio_mixup_prob=0,
    )
    dataset = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    label_map = dataset.label
    inv_map = {v: k for k, v in label_map.items()}

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        num_workers=num_workers, shuffle=False, pin_memory=True,
    )

    all_embeddings, all_labels, all_paths = [], [], []
    pipeline.eval()

    for batch in dataloader:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs
        all_embeddings.append(bonafide_head.float().cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())
        all_paths.extend(batch["path"])

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    logger.info(f"Extracted {len(embeddings)} embeddings from {json_file}")
    return embeddings, labels, label_map, all_paths


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_ll_fake_only(scores, binary_labels):
    """Compute metrics for -LL(fake) scoring.

    Args:
        scores: -log_likelihood values (higher = more likely real)
        binary_labels: 1 = real, 0 = fake
    """
    # ROC and EER
    fpr, tpr, thresholds = roc_curve(binary_labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]
    auc = roc_auc_score(binary_labels, scores)

    # Predictions at EER threshold
    predictions = (scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(binary_labels, predictions)
    cm = confusion_matrix(binary_labels, predictions, labels=[0, 1])

    prec, rec, f1, sup = precision_recall_fscore_support(
        binary_labels, predictions, labels=[0, 1], average=None, zero_division=0,
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        binary_labels, predictions, labels=[0, 1], average="macro", zero_division=0,
    )

    results = {
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "per_class_f1": f1,
        "per_class_precision": prec,
        "per_class_recall": rec,
        "per_class_support": sup,
        "fpr": fpr,
        "tpr": tpr,
        "predictions": predictions,
    }
    return results


def print_results(results, dataset_name=""):
    """Print evaluation results."""
    print(f"\n{'=' * 55}")
    print(f"LL_fake_only Results{f' — {dataset_name}' if dataset_name else ''}")
    print(f"{'=' * 55}")
    print(f"  AUC              : {results['auc']:.4f}")
    print(f"  Accuracy         : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  F1 macro         : {results['f1_macro']:.4f}")
    print(f"  EER              : {results['eer']*100:.2f}%")
    print(f"  EER threshold    : {results['eer_threshold']:.4f}")
    print()

    cm = results["confusion_matrix"]
    class_names = ["fake", "real"]
    print("  Confusion Matrix:")
    print(f"  {'':>15} {'Pred Fake':>10} {'Pred Real':>10}")
    for i, name in enumerate(class_names):
        print(f"  {'Actual '+name:>15} {cm[i,0]:>10d} {cm[i,1]:>10d}")
    print()

    for i, name in enumerate(class_names):
        print(f"  {name:>8}  P={results['per_class_precision'][i]:.4f}  "
              f"R={results['per_class_recall'][i]:.4f}  "
              f"F1={results['per_class_f1'][i]:.4f}  "
              f"n={results['per_class_support'][i]}")
    print(f"{'=' * 55}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results, scores, binary_labels, output_dir, dataset_name=""):
    """Generate confusion matrix + ROC curve + score distribution plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    class_names = ["fake", "real"]

    # --- Confusion matrix + ROC (side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm = results["confusion_matrix"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot_labels = np.array([[f"{v:.2f}" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot_labels, fmt="s", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 11}, ax=axes[0])
    axes[0].set_title("Confusion Matrix", fontsize=14)
    axes[0].set_ylabel("True Label", fontsize=12)
    axes[0].set_xlabel("Predicted Label", fontsize=12)

    fpr, tpr = results["fpr"], results["tpr"]
    eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
    axes[1].plot(fpr, tpr, color="steelblue",
                 label=f"ROC (AUC = {results['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].scatter([fpr[eer_idx]], [tpr[eer_idx]], color="red", zorder=5, s=80,
                    label=f"EER = {results['eer']:.3f}")
    axes[1].set_xlabel("False Positive Rate", fontsize=12)
    axes[1].set_ylabel("True Positive Rate", fontsize=12)
    axes[1].set_title("ROC Curve", fontsize=14)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_roc_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    # --- Score distribution ---
    fig, ax = plt.subplots(figsize=(10, 6))
    real_scores = scores[binary_labels == 1]
    fake_scores = scores[binary_labels == 0]

    lo = min(np.percentile(real_scores, 1), np.percentile(fake_scores, 1))
    hi = max(np.percentile(real_scores, 99), np.percentile(fake_scores, 99))
    bins = np.linspace(lo, hi, 80)

    ax.hist(fake_scores, bins=bins, alpha=0.5, color="#E24B4A",
            label=f"Fake (n={len(fake_scores)})", density=True)
    ax.hist(real_scores, bins=bins, alpha=0.5, color="#1D9E75",
            label=f"Real (n={len(real_scores)})", density=True)
    ax.axvline(x=results["eer_threshold"], color="black", linestyle="--",
               label=f"EER threshold = {results['eer_threshold']:.1f}")
    ax.set_xlabel("-Log-Likelihood(fake)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution — {dataset_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    path = os.path.join(output_dir, f"score_dist_{dataset_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────
def save_per_audio_csv(scores, binary_labels, paths, results, output_dir, dataset_name=""):
    """Save per-sample scores and predictions to CSV."""
    import pandas as pd

    df = pd.DataFrame({
        "path": paths,
        "label": ["real" if l == 1 else "fake" for l in binary_labels],
        "neg_ll_fake": scores,
        "prediction": ["real" if p == 1 else "fake" for p in results["predictions"]],
        "correct": (results["predictions"] == binary_labels).astype(int),
        "eer_threshold": results["eer_threshold"],
    })
    path = os.path.join(output_dir, f"per_audio_{dataset_name}.csv")
    df.to_csv(path, index=False)
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_infer_args():
    p = argparse.ArgumentParser(description="LL_fake_only inference")
    p.add_argument(
        "--checkpoint",
        default="checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt",
        help="Path to trained model checkpoint",
    )
    p.add_argument(
        "--fake_ref",
        default="data/label/beats/Event_train_stage1_fakeonly.json",
        help="JSON file with fake-only training samples (for building Gaussian reference)",
    )
    p.add_argument(
        "--test_json", nargs="+",
        default=["data/label/beats/test_track2.json"],
        help="One or more test set JSON files to evaluate",
    )
    p.add_argument("--output_dir", default="inference_outputs/ll_fake_only")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--plot", action="store_true", help="Generate plots")
    p.add_argument("--save_csv", action="store_true", help="Save per-audio CSV")
    return p.parse_args()


def main():
    args = parse_infer_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load model ──────────────────────────────────────────────────
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    pipeline, _, _ = load_pipeline_and_ccl(
        checkpoint_path=args.checkpoint,
        device=device,
        mode="beats",
        num_classes=5,
        embed_dim=527,
        truncate_layers=0,
        beats_feature="predictor",
    )

    # ── Step 2: Build fake reference Gaussian ───────────────────────────────
    logger.info(f"Building fake reference from: {args.fake_ref}")
    fake_embeddings, _, _, _ = extract_embeddings(
        pipeline, args.fake_ref, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    gaussian_fake = fit_gaussian(fake_embeddings)
    logger.info(f"Gaussian fitted on {len(fake_embeddings)} fake training samples "
                f"(embed_dim={fake_embeddings.shape[1]})")

    # ── Step 3: Score each test set ─────────────────────────────────────────
    for test_json in args.test_json:
        dataset_name = os.path.splitext(os.path.basename(test_json))[0]
        logger.info(f"\nScoring: {test_json}")

        test_embeddings, test_labels, label_map, test_paths = extract_embeddings(
            pipeline, test_json, device,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )

        # Determine which label index is "real"
        real_idx = label_map.get("real", None)
        if real_idx is None:
            logger.error(f"No 'real' label found in {test_json}. Labels: {label_map}")
            continue

        binary_labels = (test_labels == real_idx).astype(int)
        n_real = binary_labels.sum()
        n_fake = len(binary_labels) - n_real
        logger.info(f"  {len(test_labels)} samples: {n_real} real, {n_fake} fake")

        # Compute -LL(fake) scores: high = far from fake = real
        ll_fake = log_likelihood(test_embeddings, gaussian_fake)
        scores = -ll_fake

        # Evaluate
        results = evaluate_ll_fake_only(scores, binary_labels)
        print_results(results, dataset_name)

        # Optional outputs
        if args.plot:
            plot_results(results, scores, binary_labels, args.output_dir, dataset_name)

        if args.save_csv:
            save_per_audio_csv(scores, binary_labels, test_paths, results,
                               args.output_dir, dataset_name)

    logger.info("\nInference complete!")


if __name__ == "__main__":
    main()
