"""Distribution-based deepfake scoring.

Workflow:
    1. Run the model on a real-only reference set (e.g. Event_real.json) and
       collect the logits for every sample.
    2. Fit a Gaussian (μ, Σ) over those reference logits.
    3. For every test sample, compute the Mahalanobis distance from μ.
    4. Predict real iff the distance is <= threshold; else fake.
"""

import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve
from torch.utils.data import DataLoader

from src.base_dataset import BaselineDataset, BeatsDataset
from src.inference._lib.loader import load_pipeline
from src.inference._lib.metrics import (
    compute_classification_metrics,
    create_metrics_summary,
    log_metrics_summary,
)
from src.inference._lib.plots import plot_detailed_metrics
from src.inference.inference_ccl import load_pipeline_and_ccl
from src.utils.arguments import get_args

logger = logging.getLogger(__name__)


def _make_dataset(json_file, mode, args):
    if mode == 'beats':
        return BeatsDataset(json_file=json_file, transformation=None, args=args)
    return BaselineDataset(json_file=json_file, transformation=None, args=args)


@torch.no_grad()
def _collect_plain(pipeline, dataloader, device):
    """Forward pass through a vanilla pipeline. Returns logits + softmax probs."""
    pipeline = pipeline.to(device).eval()

    all_logits, all_probs, all_paths, all_labels = [], [], [], []
    for batch in dataloader:
        inputs = batch['audio'].to(device, dtype=torch.float32)
        paths = batch['path']
        labels = batch.get('label')

        logits = pipeline.forward_pipeline(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = F.softmax(logits, dim=1)

        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_paths.extend(paths)
        if labels is not None:
            all_labels.extend(labels.cpu().numpy())

    return {
        'logits': np.concatenate(all_logits, axis=0),
        'probabilities': np.concatenate(all_probs, axis=0),
        'paths': np.array(all_paths),
        'labels': np.array(all_labels) if all_labels else None,
    }


@torch.no_grad()
def _collect_ccl(pipeline, ccl_head, dataloader, device):
    """Forward through pipeline → CCL head; returns 2D cosine logits + softmax probs.

    The CCL head outputs cos_sim ∈ [-1, 1] over the 2 centers (col 0 = fake, col 1 = real).
    """
    pipeline = pipeline.to(device).eval()
    ccl_head = ccl_head.to(device).eval()

    all_logits, all_probs, all_paths, all_labels = [], [], [], []
    for batch in dataloader:
        inputs = batch['audio'].to(device, dtype=torch.float32)
        paths = batch['path']
        labels = batch.get('label')
        B = inputs.size(0)

        outputs = pipeline.forward_pipeline(inputs)
        bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs

        dummy = torch.zeros(B, dtype=torch.long, device=device)
        _, ccl_logits = ccl_head.forward(x=bonafide_head, labels=dummy)
        probs = F.softmax(ccl_logits, dim=1)

        all_logits.append(ccl_logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_paths.extend(paths)
        if labels is not None:
            all_labels.extend(labels.cpu().numpy())

    return {
        'logits': np.concatenate(all_logits, axis=0),
        'probabilities': np.concatenate(all_probs, axis=0),
        'paths': np.array(all_paths),
        'labels': np.array(all_labels) if all_labels else None,
    }


def collect_logits(scorer, dataloader, device='cuda'):
    """Run the configured scorer over a dataloader.

    `scorer` is the dict returned by `build_scorer` — either a plain pipeline
    or a (pipeline, ccl_head) pair, depending on `--use_ccl`.
    """
    if scorer['kind'] == 'ccl':
        return _collect_ccl(scorer['pipeline'], scorer['ccl_head'], dataloader, device)
    return _collect_plain(scorer['pipeline'], dataloader, device)


def build_scorer(ckpt_path, mode, device, use_ccl=False, num_classes=2,
                  embed_dim=527, truncate_layers=0, beats_feature='predictor'):
    """Construct the model(s) needed to produce 2D logits per sample."""
    if use_ccl:
        pipeline, ccl_head, _arc_head = load_pipeline_and_ccl(
            checkpoint_path=ckpt_path,
            device=device,
            mode=mode,
            num_classes=5,        # CCL-trained models are 5-class internally
            embed_dim=embed_dim,
            truncate_layers=truncate_layers,
            beats_feature=beats_feature,
        )
        return {'kind': 'ccl', 'pipeline': pipeline, 'ccl_head': ccl_head}

    pipeline = load_pipeline(
        checkpoint_path=ckpt_path, mode=mode,
        num_classes=num_classes, device=device,
        beats_feature=beats_feature,
    )
    return {'kind': 'plain', 'pipeline': pipeline}


def fit_reference_distribution(reference_logits, eps=1e-6):
    """Fit a regularized Gaussian (μ, Σ) over reference logits."""
    X = np.asarray(reference_logits, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected reference_logits with shape [N, D], got {X.shape}")

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += eps * np.eye(cov.shape[0])  # numerical stability for Cholesky

    c, lower = cho_factor(cov)
    logger.info(f"Reference distribution fit: μ={mu.tolist()}, "
                 f"Σ shape={cov.shape}, N={len(X)}")
    return {'mu': mu, 'cov': cov, 'cho_c': c, 'cho_lower': lower}


def mahalanobis_batch(X, ref):
    """Mahalanobis distance from each row of X to the reference Gaussian."""
    diff = np.asarray(X, dtype=np.float64) - ref['mu']
    solved = cho_solve((ref['cho_c'], ref['cho_lower']), diff.T).T
    sq = np.einsum('ij,ij->i', diff, solved)
    return np.sqrt(np.maximum(sq, 0.0))


def classify_by_distance(distances, threshold):
    """Return 1 (real) where distance <= threshold, else 0 (fake)."""
    return (np.asarray(distances) <= threshold).astype(int)


def compute_auto_threshold(method, reference_distances, test_distances=None,
                            test_labels=None, percentile=95.0, dim=2):
    """Derive a distance threshold from data.

    Methods:
        'percentile' — Nth percentile of reference distances. Robust, no labels needed.
        'chi2'       — sqrt(chi2.ppf(N/100, df=dim)). Theoretical Gaussian containment.
        'eer'        — sweep for FAR == FRR on the test set. Requires test labels.

    Returns: (threshold, info_dict)
    """
    method = (method or 'none').lower()

    if method == 'percentile':
        thr = float(np.percentile(reference_distances, percentile))
        return thr, {'method': 'percentile', 'percentile': percentile,
                      'n_reference': int(len(reference_distances))}

    if method == 'chi2':
        from scipy.stats import chi2
        thr = float(np.sqrt(chi2.ppf(percentile / 100.0, df=dim)))
        return thr, {'method': 'chi2', 'percentile': percentile, 'dim': int(dim)}

    if method == 'eer':
        if test_labels is None or test_distances is None:
            raise ValueError("auto_threshold='eer' requires test labels and test distances")
        from sklearn.metrics import roc_curve
        # Lower distance ⇒ more real; use -distance as the "real-likeness" score.
        scores = -np.asarray(test_distances)
        fpr, tpr, thresholds = roc_curve(test_labels, scores, pos_label=1)
        fnr = 1.0 - tpr
        idx = int(np.argmin(np.abs(fpr - fnr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        thr = float(-thresholds[idx])  # flip sign back to distance space
        return thr, {'method': 'eer', 'eer': eer, 'idx': idx}

    raise ValueError(f"Unknown auto_threshold method: {method}")


def plot_distance_distribution(distances, labels, threshold, save_path="distance_distribution.png"):
    """Histogram of Mahalanobis distances split by true label, with threshold line."""
    plt.figure(figsize=(9, 6))

    if labels is not None:
        fake_d = distances[labels == 0]
        real_d = distances[labels == 1]
        if len(fake_d) > 0:
            plt.hist(fake_d, bins=60, alpha=0.6, color='salmon', edgecolor='black', label='Fake')
        if len(real_d) > 0:
            plt.hist(real_d, bins=60, alpha=0.6, color='steelblue', edgecolor='black', label='Real')
    else:
        plt.hist(distances, bins=60, alpha=0.7, edgecolor='black', label='All samples')

    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.4f}')
    plt.xlabel('Mahalanobis distance from Event_real reference')
    plt.ylabel('Number of samples')
    plt.title('Test-set distance distribution vs. Event_real reference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Distance distribution plot saved → {save_path}")


def run(ref_json_file, val_json_file, ckpt_path, mode, threshold, args, device='cuda',
        num_classes=2, output_dir='.', batch_size=64, num_workers=8,
        use_ccl=False, embed_dim=527, truncate_layers=0,
        auto_threshold='none', auto_threshold_pct=95.0,
        beats_feature='predictor'):
    """End-to-end scoring with a real-only reference distribution."""
    os.makedirs(output_dir, exist_ok=True)

    scorer = build_scorer(
        ckpt_path=ckpt_path, mode=mode, device=device,
        use_ccl=use_ccl, num_classes=num_classes,
        embed_dim=embed_dim, truncate_layers=truncate_layers,
        beats_feature=beats_feature,
    )
    logger.info(f"Scorer kind: {scorer['kind']}")

    # ── Reference set: Event_real ─────────────────────────────────────────────
    logger.info(f"Building reference distribution from: {ref_json_file}")
    ref_dataset = _make_dataset(ref_json_file, mode, args)
    ref_loader = DataLoader(ref_dataset, batch_size=batch_size,
                             num_workers=num_workers, shuffle=False)
    ref_outputs = collect_logits(scorer, ref_loader, device=device)
    ref_dist = fit_reference_distribution(ref_outputs['logits'])
    ref_distances = mahalanobis_batch(ref_outputs['logits'], ref_dist)
    logger.info(
        f"Reference distance summary: n={len(ref_distances)}, "
        f"mean={ref_distances.mean():.4f}, std={ref_distances.std():.4f}, "
        f"p50={float(np.percentile(ref_distances, 50)):.4f}, "
        f"p95={float(np.percentile(ref_distances, 95)):.4f}, "
        f"p99={float(np.percentile(ref_distances, 99)):.4f}"
    )

    # ── Test set ──────────────────────────────────────────────────────────────
    logger.info(f"Scoring test set: {val_json_file}")
    test_dataset = _make_dataset(val_json_file, mode, args)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              num_workers=num_workers, shuffle=False)
    test_outputs = collect_logits(scorer, test_loader, device=device)

    distances = mahalanobis_batch(test_outputs['logits'], ref_dist)

    # ── Threshold ─────────────────────────────────────────────────────────────
    auto_method = (auto_threshold or 'none').lower()
    auto_info = None
    if auto_method != 'none':
        if auto_method == 'eer' and test_outputs['labels'] is None:
            logger.warning("auto_threshold='eer' but test set has no labels; "
                            "falling back to 'percentile'.")
            auto_method = 'percentile'
        threshold, auto_info = compute_auto_threshold(
            method=auto_method,
            reference_distances=ref_distances,
            test_distances=distances,
            test_labels=test_outputs['labels'],
            percentile=auto_threshold_pct,
            dim=ref_outputs['logits'].shape[1],
        )
        logger.info(f"Auto-threshold ({auto_method}) → {threshold:.4f}  ({auto_info})")
    else:
        logger.info(f"Using literal --dist_threshold = {threshold:.4f}")

    predictions = classify_by_distance(distances, threshold)

    # Confidence proxy = exp(-distance), bounded in (0, 1].
    confidences = np.exp(-distances)

    results = {
        'paths':         test_outputs['paths'],
        'logits':        test_outputs['logits'],
        'probabilities': test_outputs['probabilities'],
        'distances':     distances,
        'predictions':   predictions,
        'confidences':   confidences,
        'threshold':     threshold,
    }
    if test_outputs['labels'] is not None:
        results['labels'] = test_outputs['labels']

    logger.info(
        f"Distance summary: min={distances.min():.4f}, "
        f"max={distances.max():.4f}, mean={distances.mean():.4f}, "
        f"median={float(np.median(distances)):.4f}"
    )
    if 'labels' in results:
        for lbl, name in [(1, 'real'), (0, 'fake')]:
            mask = results['labels'] == lbl
            if mask.any():
                d = distances[mask]
                logger.info(
                    f"  {name:>4}: n={mask.sum()}, mean={d.mean():.4f}, "
                    f"std={d.std():.4f}, min={d.min():.4f}, max={d.max():.4f}"
                )

    # ── Outputs ───────────────────────────────────────────────────────────────
    df = pd.DataFrame({
        'path':       results['paths'],
        'distance':   distances,
        'confidence': confidences,
        'prob_fake':  results['probabilities'][:, 0],
        'prob_real':  results['probabilities'][:, 1],
        'prediction': ['real' if p == 1 else 'fake' for p in predictions],
        'threshold':  threshold,
    })
    if 'labels' in results:
        df['label']   = ['real' if l == 1 else 'fake' for l in results['labels']]
        df['correct'] = (results['predictions'] == results['labels']).astype(int)
    csv_path = os.path.join(output_dir, "distribution_per_audio.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Per-audio CSV saved → {csv_path}")

    plot_distance_distribution(
        distances=distances,
        labels=results.get('labels'),
        threshold=threshold,
        save_path=os.path.join(output_dir, "distance_distribution.png"),
    )

    # ── Metrics (only if labels present) ──────────────────────────────────────
    metrics = None
    if 'labels' in results:
        # Use 1 - distance/distance.max() as a "real-likeness" score for AUC.
        d_max = max(distances.max(), 1e-9)
        prob_real_proxy = 1.0 - np.clip(distances / d_max, 0, 1)
        prob_fake_proxy = 1.0 - prob_real_proxy
        proxy_probs = np.column_stack([prob_fake_proxy, prob_real_proxy])

        metrics = compute_classification_metrics(
            labels=results['labels'],
            predictions=results['predictions'],
            probabilities=proxy_probs,
            confidences=confidences,
            class_names=['fake', 'real'],
            include_eer=True,
            eer_scores=prob_real_proxy,
        )
        log_metrics_summary(metrics)

        plot_detailed_metrics(
            metrics,
            save_path=os.path.join(output_dir, "distribution_metrics.png"),
            show_eer_panel=True,
        )
        create_metrics_summary(
            metrics,
            save_path=os.path.join(output_dir, "distribution_summary.txt"),
            title="DISTRIBUTION-BASED EVALUATION SUMMARY",
        )

    return results, metrics, ref_dist


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ref_json = args.ref_json_file
    if ref_json is None:
        raise ValueError(
            "Pass --ref_json_file pointing to a real-only label JSON "
            "(e.g. data/label/beats/old/Event_real.json)."
        )

    threshold = float(args.dist_threshold)
    output_dir = getattr(args, 'output_dir', '.')
    use_ccl = bool(getattr(args, 'use_ccl', False))
    auto_threshold = getattr(args, 'auto_threshold', 'none')
    auto_threshold_pct = float(getattr(args, 'auto_threshold_pct', 95.0))

    results, metrics, _ref_dist = run(
        ref_json_file=ref_json,
        val_json_file=args.val_json_file,
        ckpt_path=args.load_ckpt_path,
        mode=args.mode,
        threshold=threshold,
        args=args,
        device=device,
        num_classes=2,
        output_dir=output_dir,
        use_ccl=use_ccl,
        embed_dim=getattr(args, 'embed_dim', 527),
        truncate_layers=getattr(args, 'truncate_layers', 0),
        auto_threshold=auto_threshold,
        auto_threshold_pct=auto_threshold_pct,
        beats_feature=getattr(args, 'beats_feature', 'predictor'),
    )

    print("=" * 60)
    print("DISTRIBUTION-BASED INFERENCE COMPLETE")
    print("=" * 60)
    print(f"Reference: {ref_json}")
    print(f"Test set:  {args.val_json_file}")
    print(f"Scorer:    {'CCL' if use_ccl else 'plain'}")
    print(f"Threshold: {results['threshold']:.4f}  (Mahalanobis; auto={auto_threshold})")
    if metrics is not None:
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Macro F1:  {metrics['f1_macro']:.4f}")
        if metrics.get('eer') is not None:
            print(f"EER:       {metrics['eer'] * 100:.2f}%")
    print(f"Outputs in: {output_dir}/")
    print("=" * 60)
