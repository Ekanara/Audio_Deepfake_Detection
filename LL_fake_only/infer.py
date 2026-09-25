"""
infer.py — Log-Likelihood_fake_only Inference
===============================================

Scores test samples by measuring how far their embeddings are from the
fake training distribution:

  1. Load trained BEATs checkpoint
  2. Extract 527-dim bonafide_head embeddings from fake training samples
  3. Fit multivariate Gaussian  N(μ_fake, Σ_fake)
  4. Score each test sample:  score(x) = -log p(x | N(μ_fake, Σ_fake))
       High score → far from fake → REAL
       Low score  → close to fake → FAKE
  5. Classify at EER threshold from ROC curve

Usage
-----
  # Default: test_track2 against fake training reference
  python LL_fake_only/infer.py

  # Custom checkpoint + multiple test sets + plots
  python LL_fake_only/infer.py \
      --checkpoint checkpoint/my_model/sample-03.ckpt \
      --test_json LL_fake_only/data/test_track2.json \
                  LL_fake_only/data/Event_test_5class.json \
      --plot --save_csv

  # Custom fake reference
  python LL_fake_only/infer.py \
      --fake_ref LL_fake_only/data/Event_train_stage1_fakeonly.json
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

_HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the bundled src/ (standalone package); fall back to the repo's ../src.
for _cand in (os.path.join(_HERE, "src"), os.path.join(_HERE, "..", "src")):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.dirname(_cand))  # parent → enables `import src.xxx`
        sys.path.insert(0, _cand)                    # src/   → enables `from base_dataset import`
        break

from argparse import Namespace
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian fitting and Log-Likelihood scoring
# ─────────────────────────────────────────────────────────────────────────────
def fit_gaussian(X, eps=1e-6):
    """Fit multivariate Gaussian to embeddings X.

    Returns (mean, cholesky_factor, lower_flag, log_determinant) for
    efficient log-likelihood computation via Cholesky decomposition.
    """
    X = X.astype(np.float64)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    cf, lower = cho_factor(cov)
    _, logdet = np.linalg.slogdet(cov)
    return mu, cf, lower, logdet


def log_likelihood(X, gaussian):
    """Compute log p(x | N(mu, Sigma)) for each row in X.

    log p(x) = -0.5 * [(x-mu)^T Sigma^{-1} (x-mu) + log|Sigma|]
    (constant d*log(2pi) omitted — does not affect ranking)
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
    """Extract 527-dim bonafide_head embeddings from all samples.

    Returns (embeddings, labels, label_map, paths).
    """
    ds_args = Namespace(
        num_label=5, three_loss=True,
        audio_aug=False, audio_mixup=False,
        audio_aug_prob=0, audio_mixup_prob=0,
    )
    dataset = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    label_map = dataset.label

    dl = DataLoader(dataset, batch_size=batch_size,
                    num_workers=num_workers, shuffle=False, pin_memory=True)

    all_emb, all_lbl, all_paths = [], [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        bonafide = outputs[0] if isinstance(outputs, tuple) else outputs
        all_emb.append(bonafide.float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        all_paths.extend(batch["path"])

    embeddings = np.concatenate(all_emb, axis=0)
    labels = np.array(all_lbl)
    logger.info(f"Extracted {len(embeddings)} embeddings (dim={embeddings.shape[1]}) "
                f"from {json_file}")
    return embeddings, labels, label_map, all_paths


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(scores, binary_labels):
    """Compute AUC, EER, accuracy, confusion matrix, per-class metrics.

    Args:
        scores: -log_likelihood(fake) values — higher = more likely real
        binary_labels: 1 = real, 0 = fake
    """
    fpr, tpr, thresholds = roc_curve(binary_labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx]
    eer_thr = thresholds[eer_idx]
    auc = roc_auc_score(binary_labels, scores)

    preds = (scores >= eer_thr).astype(int)
    acc = accuracy_score(binary_labels, preds)
    cm = confusion_matrix(binary_labels, preds, labels=[0, 1])
    prec, rec, f1, sup = precision_recall_fscore_support(
        binary_labels, preds, labels=[0, 1], average=None, zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(
        binary_labels, preds, labels=[0, 1], average="macro", zero_division=0)

    return dict(
        auc=auc, eer=eer, eer_threshold=eer_thr,
        accuracy=acc, f1_macro=f1_macro,
        confusion_matrix=cm, predictions=preds,
        per_class_f1=f1, per_class_precision=prec,
        per_class_recall=rec, per_class_support=sup,
        fpr=fpr, tpr=tpr,
    )


def print_results(results, name=""):
    header = f"LL_fake_only — {name}" if name else "LL_fake_only"
    cm = results["confusion_matrix"]
    print(f"\n{'=' * 55}")
    print(f"  {header}")
    print(f"{'=' * 55}")
    print(f"  AUC       : {results['auc']:.4f}")
    print(f"  Accuracy  : {results['accuracy']*100:.2f}%")
    print(f"  F1 macro  : {results['f1_macro']:.4f}")
    print(f"  EER       : {results['eer']*100:.2f}%  (thr={results['eer_threshold']:.2f})")
    print()
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"  {'':>15} {'Pred Fake':>10} {'Pred Real':>10}")
    for i, name_ in enumerate(["fake", "real"]):
        print(f"  {'True '+name_:>15} {cm[i,0]:>10d} {cm[i,1]:>10d}")
    print()
    for i, name_ in enumerate(["fake", "real"]):
        print(f"  {name_:>6}  P={results['per_class_precision'][i]:.4f}  "
              f"R={results['per_class_recall'][i]:.4f}  "
              f"F1={results['per_class_f1'][i]:.4f}  "
              f"n={results['per_class_support'][i]}")
    print(f"{'=' * 55}")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_results(results, scores, binary_labels, output_dir, name=""):
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)
    class_names = ["fake", "real"]

    # Confusion matrix + ROC side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm = results["confusion_matrix"]
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{v:.2f}" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot, fmt="s", cmap="Blues",
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
    p = os.path.join(output_dir, f"confusion_roc_{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {p}")

    # Score distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    real_s = scores[binary_labels == 1]
    fake_s = scores[binary_labels == 0]
    lo = min(np.percentile(real_s, 1), np.percentile(fake_s, 1))
    hi = max(np.percentile(real_s, 99), np.percentile(fake_s, 99))
    bins = np.linspace(lo, hi, 80)
    ax.hist(fake_s, bins=bins, alpha=0.5, color="#E24B4A",
            label=f"Fake (n={len(fake_s)})", density=True)
    ax.hist(real_s, bins=bins, alpha=0.5, color="#1D9E75",
            label=f"Real (n={len(real_s)})", density=True)
    ax.axvline(x=results["eer_threshold"], color="black", linestyle="--",
               label=f"EER thr = {results['eer_threshold']:.1f}")
    ax.set_xlabel("-Log-Likelihood(fake)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Score Distribution — {name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)
    p = os.path.join(output_dir, f"score_dist_{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {p}")


def save_csv(scores, binary_labels, paths, results, output_dir, name=""):
    import pandas as pd
    df = pd.DataFrame({
        "path": paths,
        "label": ["real" if l == 1 else "fake" for l in binary_labels],
        "neg_ll_fake": scores,
        "prediction": ["real" if p == 1 else "fake" for p in results["predictions"]],
        "correct": (results["predictions"] == binary_labels).astype(int),
        "eer_threshold": results["eer_threshold"],
    })
    p = os.path.join(output_dir, f"per_audio_{name}.csv")
    df.to_csv(p, index=False)
    logger.info(f"Saved: {p}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LL_fake_only inference")
    p.add_argument("--checkpoint",
        default=os.path.join(_HERE, "checkpoint", "ll_fake_only_stage2.ckpt"))
    p.add_argument("--fake_ref",
        default=os.path.join(_HERE, "data", "Event_train_stage1_fakeonly.json"),
        help="Fake-only training samples for Gaussian fitting")
    p.add_argument("--test_json", nargs="+",
        default=[os.path.join(_HERE, "data", "test_track2.json")],
        help="Test set JSON file(s) to evaluate")
    p.add_argument("--output_dir", default="inference_outputs/ll_fake_only")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--plot", action="store_true", help="Generate plots")
    p.add_argument("--save_csv", action="store_true", help="Save per-audio CSV")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    pipeline, _, _ = load_pipeline_and_ccl(
        checkpoint_path=args.checkpoint, device=device, mode="beats",
        num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

    # 2. Build fake reference Gaussian
    logger.info(f"Building fake reference from: {args.fake_ref}")
    fake_emb, _, _, _ = extract_embeddings(
        pipeline, args.fake_ref, device,
        batch_size=args.batch_size, num_workers=args.num_workers)
    g_fake = fit_gaussian(fake_emb)
    logger.info(f"Gaussian fitted: {len(fake_emb)} fake samples, dim={fake_emb.shape[1]}")

    # 3. Score each test set
    for test_json in args.test_json:
        ds_name = os.path.splitext(os.path.basename(test_json))[0]
        logger.info(f"\nScoring: {test_json}")

        test_emb, test_lbl, label_map, test_paths = extract_embeddings(
            pipeline, test_json, device,
            batch_size=args.batch_size, num_workers=args.num_workers)

        real_idx = label_map.get("real")
        if real_idx is None:
            logger.error(f"No 'real' label in {test_json}. Labels: {label_map}")
            continue

        binary = (test_lbl == real_idx).astype(int)
        logger.info(f"  {len(test_lbl)} samples: {binary.sum()} real, "
                     f"{len(binary) - binary.sum()} fake")

        # -LL(fake): high = far from fake = real
        scores = -log_likelihood(test_emb, g_fake)

        results = evaluate(scores, binary)
        print_results(results, ds_name)

        if args.plot:
            plot_results(results, scores, binary, args.output_dir, ds_name)
        if args.save_csv:
            save_csv(scores, binary, test_paths, results, args.output_dir, ds_name)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
