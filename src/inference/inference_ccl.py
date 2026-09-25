import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.base_dataset import BaselineDataset, BeatsDataset
from src.inference._lib.generator_score import save_generator_breakdown
from src.inference._lib.loader import build_pipeline
from src.inference._lib.metrics import apply_threshold, compute_eer
from src.inference._lib.plots import (
    plot_confusion_and_roc,
    plot_eer_curve,
    plot_score_distribution,
)
from src.utils.arguments import get_args
from src.utils.training_utils import ArcFaceLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CCL head (must match training definition exactly)
# ─────────────────────────────────────────────────────────────────────────────
class CenterContrastiveLoss(nn.Module):
    def __init__(self, embed_dim, n_classes, s=64, m=0.2, lambda_c=1.0):
        super().__init__()
        self.s = s
        self.m = m
        self.lambda_c = lambda_c
        self.n_classes = n_classes
        self.centers = nn.Parameter(torch.randn(n_classes, embed_dim))
        nn.init.xavier_normal_(self.centers)

    def forward(self, x, labels):
        if labels.dim() > 1:
            labels = labels.argmax(dim=1).long()
        else:
            labels = labels.long()

        B = x.size(0)
        x_norm = F.normalize(x, p=2, dim=1)
        c_norm = F.normalize(self.centers, p=2, dim=1)

        cos_sim = x_norm @ c_norm.T
        cos_pos = cos_sim[torch.arange(B), labels]

        num_exp = self.s * (cos_pos - self.m) + 2 * self.lambda_c * cos_pos

        denom_logits = self.s * cos_sim.clone()
        denom_logits[torch.arange(B), labels] = self.s * (cos_pos - self.m)

        log_denom = torch.logsumexp(denom_logits, dim=1)
        loss = -torch.mean(num_exp - log_denom)

        return loss, cos_sim


def _load_head_state(state_dict, prefixes):
    """Strip the first matching prefix and return (substate, prefix) or ({}, None)."""
    for prefix in prefixes:
        substate = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if substate:
            return substate, prefix
    return {}, None


def load_pipeline_and_ccl(checkpoint_path, device, mode, num_classes=5,
                           embed_dim=128, truncate_layers=0,
                           ccl_s=64, ccl_m=0.2, ccl_lambda=1.0,
                           beats_feature='predictor'):
    """Load backbone pipeline + CCL head + ArcFace head from a Lightning ckpt."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    # ── Infer num_classes from checkpoint to avoid size mismatch ─────────────
    for key in state_dict:
        if 'last_layer.weight' in key or ('softmax_head' in key and 'weight' in key):
            inferred = state_dict[key].shape[0]
            if inferred != num_classes:
                logger.warning(
                    f"num_classes mismatch: got {num_classes}, "
                    f"checkpoint has {inferred} — using {inferred}"
                )
                num_classes = inferred
            break

    # ── Build pipeline (BEATs in three-loss mode for raw embeddings) ─────────
    pipeline = build_pipeline(
        mode=mode,
        num_classes=num_classes,
        in_features=3,
        device=device,
        truncate_layers=truncate_layers,
        three_loss=(mode == 'beats'),
        beats_feature=beats_feature,
    )

    pipeline_state = {
        k.replace('pipeline.', ''): v
        for k, v in state_dict.items() if k.startswith('pipeline.')
    }
    pipeline.load_state_dict(pipeline_state)
    pipeline.eval().to(device)
    logger.info("Pipeline loaded.")

    # ── CCL head ──────────────────────────────────────────────────────────────
    ccl_state, matched_prefix = _load_head_state(state_dict, [
        'asoftmax_loss.',
        'center_constrastive_loss.',
        'center_contrastive_loss.',
        'ccl.',
    ])
    if 'centers' in ccl_state:
        inferred_embed_dim = int(ccl_state['centers'].shape[1])
        if inferred_embed_dim != embed_dim:
            logger.warning(
                f"embed_dim={embed_dim} but checkpoint CCL centers have dim={inferred_embed_dim}; "
                f"using {inferred_embed_dim}"
            )
            embed_dim = inferred_embed_dim

    ccl_head = CenterContrastiveLoss(
        embed_dim=embed_dim, n_classes=2,
        s=ccl_s, m=ccl_m, lambda_c=ccl_lambda,
    ).to(device)

    if ccl_state:
        ccl_head.load_state_dict(ccl_state)
        logger.info(f"CCL head loaded (prefix='{matched_prefix}', embed_dim={embed_dim}).")
    else:
        top_keys = sorted(set(k.split('.')[0] for k in state_dict.keys()))
        logger.warning(
            f"CCL head weights NOT found — using random init.\n"
            f"  Top-level checkpoint keys: {top_keys}"
        )
    ccl_head.eval()

    # ── ArcFace head ──────────────────────────────────────────────────────────
    arc_state, arc_prefix = _load_head_state(state_dict, [
        'arcface_loss.', 'asoftmax_loss.', 'arc_loss.',
    ])
    arc_embed_dim = embed_dim
    for weight_key in ('weight', 'W', 'kernel'):
        if weight_key in arc_state and arc_state[weight_key].ndim == 2:
            inferred = int(arc_state[weight_key].shape[1])
            if inferred != arc_embed_dim:
                logger.warning(
                    f"ArcFace embed_dim={arc_embed_dim} but checkpoint head expects {inferred}; "
                    f"using {inferred}"
                )
                arc_embed_dim = inferred
            break

    arc_head = ArcFaceLoss(embed_dim=arc_embed_dim, n_classes=num_classes, m=3, s=30).to(device)
    if arc_state:
        arc_head.load_state_dict(arc_state)
        logger.info(f"ArcFace head loaded (prefix='{arc_prefix}').")
    else:
        top_keys = sorted(set(k.split('.')[0] for k in state_dict.keys()))
        logger.warning(f"ArcFace head NOT found — random init. Keys: {top_keys}")
    arc_head.eval()

    return pipeline, ccl_head, arc_head


# ─────────────────────────────────────────────────────────────────────────────
# Inference: CCL-based real/fake scores
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(pipeline, ccl_head, arc_head, dataloader, device, real_class_idx=4):
    """Score samples with both CCL and ArcFace heads, collapsed to binary real/fake.

    For each sample we compute:
      • CCL  : 2-class softmax over CCL center cosines (col 0 = fake, col 1 = real)
      • Arc  : 5-class softmax over ArcFace cosines, then collapse to binary by
               using prob[:, real_class_idx] as P(real). Predictions follow the
               same threshold-style argmax as CCL after stacking [1-P_real, P_real].

    Returns a dict with `ccl_*` and `arc_*` keyed entries plus shared paths/labels.
    """
    pipeline.to(device).eval()
    arc_head.to(device).eval()
    ccl_head.to(device).eval()

    ccl_probs_all, ccl_preds_all, ccl_confs_all = [], [], []
    arc_probs_all, arc_preds_all, arc_confs_all = [], [], []
    arc_probs_5class_all = []
    all_labels, all_labels_5, all_paths = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            audio  = batch['audio'].to(device, dtype=torch.float32)
            labels = batch['label']
            paths  = batch['path']
            B      = audio.size(0)

            outputs       = pipeline.forward_pipeline(audio)
            bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs

            # ── CCL: 2-class binary scoring ──────────────────────────────────
            dummy = torch.zeros(B, dtype=torch.long, device=device)
            _, ccl_logits = ccl_head.forward(x=bonafide_head, labels=dummy)
            ccl_probs = F.softmax(ccl_logits, dim=1).cpu().numpy()        # [B, 2]
            ccl_preds = ccl_probs.argmax(axis=1)
            ccl_confs = ccl_probs[np.arange(B), ccl_preds]

            # ── ArcFace: 5-class cosines → binary by real_class_idx ──────────
            x_norm = F.normalize(bonafide_head, p=2, dim=1)
            w_norm = F.normalize(arc_head.weight, p=2, dim=1)
            arc_cos = x_norm @ w_norm.T                                   # [B, 5]
            arc_5 = F.softmax(arc_head.s * arc_cos, dim=1).cpu().numpy()  # [B, 5]
            arc_p_real = arc_5[:, real_class_idx]
            arc_p_fake = 1.0 - arc_p_real
            arc_probs = np.stack([arc_p_fake, arc_p_real], axis=1)        # [B, 2]
            arc_preds = arc_probs.argmax(axis=1)
            arc_confs = arc_probs[np.arange(B), arc_preds]

            ccl_probs_all.append(ccl_probs)
            ccl_preds_all.append(ccl_preds)
            ccl_confs_all.append(ccl_confs)
            arc_probs_all.append(arc_probs)
            arc_preds_all.append(arc_preds)
            arc_confs_all.append(arc_confs)
            arc_probs_5class_all.append(arc_5)
            all_paths.extend(paths)

            if labels is not None:
                labels_np = labels.cpu().numpy()
                all_labels_5.append(labels_np)
                all_labels.append((labels_np == real_class_idx).astype(int))

    results = {
        'paths':                    np.array(all_paths),
        # CCL outputs
        'ccl_probabilities':        np.concatenate(ccl_probs_all, axis=0),
        'ccl_predictions':          np.concatenate(ccl_preds_all, axis=0),
        'ccl_confidences':          np.concatenate(ccl_confs_all, axis=0),
        # ArcFace outputs
        'arc_probabilities':        np.concatenate(arc_probs_all, axis=0),
        'arc_predictions':          np.concatenate(arc_preds_all, axis=0),
        'arc_confidences':          np.concatenate(arc_confs_all, axis=0),
        'arc_probabilities_5class': np.concatenate(arc_probs_5class_all, axis=0),
        # Default head used by downstream evaluate(): CCL (kept for back-compat)
        'probabilities':            np.concatenate(ccl_probs_all, axis=0),
        'predictions':              np.concatenate(ccl_preds_all, axis=0),
        'confidences':              np.concatenate(ccl_confs_all, axis=0),
    }
    if all_labels:
        results['labels']        = np.concatenate(all_labels, axis=0)
        results['labels_5class'] = np.concatenate(all_labels_5, axis=0)

    logger.info(f"Inference complete — {len(results['paths'])} samples (both CCL + ArcFace heads).")
    return results


def per_class_summary(results, head_prefix):
    """Print real/fake recall + EER/AUC for one head ('ccl' or 'arc')."""
    if 'labels' not in results:
        return None
    labels = results['labels']
    preds  = results[f'{head_prefix}_predictions']
    probs  = results[f'{head_prefix}_probabilities']
    prob_real = probs[:, 1]

    real_mask = labels == 1
    fake_mask = labels == 0
    real_acc = float((preds[real_mask] == 1).mean()) if real_mask.any() else float('nan')
    fake_acc = float((preds[fake_mask] == 0).mean()) if fake_mask.any() else float('nan')
    overall  = float((preds == labels).mean())

    auc = eer = float('nan')
    if len(np.unique(labels)) >= 2:
        auc = roc_auc_score(labels, prob_real)
        eer, _, _, _ = compute_eer(labels, prob_real)

    return {
        'head':     head_prefix,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'accuracy': overall,
        'auc':      auc,
        'eer':      eer,
        'n_real':   int(real_mask.sum()),
        'n_fake':   int(fake_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(results, class_names=('fake', 'real'), output_dir='.'):
    if 'labels' not in results:
        logger.warning("No ground-truth labels — skipping evaluation.")
        return None

    labels = results['labels']
    predictions = results['predictions']
    prob_real = results['probabilities'][:, 1]
    confidences = results['confidences']
    threshold = results.get('threshold', 0.5)

    label_ints = [0, 1]
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, sup = precision_recall_fscore_support(
        labels, predictions, labels=label_ints, average=None, zero_division=0,
    )
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        labels, predictions, labels=label_ints, average='macro', zero_division=0,
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        labels, predictions, labels=label_ints, average='weighted', zero_division=0,
    )
    cm = confusion_matrix(labels, predictions, labels=label_ints)

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # Test set with only one ground-truth class — AUC/EER undefined.
        logger.warning(
            f"Only one class present in labels: {unique_labels}. "
            f"AUC and EER are undefined."
        )
        auc = float('nan')
        eer = float('nan')
        eer_thr = float('nan')
    else:
        auc = roc_auc_score(labels, prob_real)
        eer, eer_thr, _, _ = compute_eer(labels, prob_real)

    logger.info("=" * 55)
    logger.info("EVALUATION RESULTS (CCL binary)")
    logger.info("=" * 55)
    logger.info(f"Threshold        : {threshold:.4f}")
    logger.info(f"Accuracy         : {acc:.4f}")
    logger.info(f"EER              : {eer * 100:.2f}%  (@ threshold {eer_thr:.4f})")
    logger.info(f"AUC-ROC          : {auc:.4f}")
    logger.info(f"Macro   F1       : {f1_m:.4f}")
    logger.info(f"Weighted F1      : {f1_w:.4f}")
    logger.info(f"Avg Confidence   : {confidences.mean():.4f}")
    logger.info("-" * 55)
    for i, cn in enumerate(class_names):
        logger.info(f"  {cn:<8}  P={prec[i]:.4f}  R={rec[i]:.4f}  F1={f1[i]:.4f}  n={sup[i]}")
    logger.info("=" * 55)

    plot_eer_curve(labels, prob_real, eer, eer_thr,
                   save_path=os.path.join(output_dir, "eer_curve.png"))
    plot_confusion_and_roc(labels, predictions, prob_real, auc, eer, eer_thr,
                           class_names=list(class_names),
                           save_path=os.path.join(output_dir, "evaluation_metrics.png"))
    plot_score_distribution(labels, prob_real,
                            save_path=os.path.join(output_dir, "score_distribution.png"))

    return {
        'accuracy': acc, 'eer': eer, 'eer_threshold': eer_thr,
        'auc': auc, 'f1_macro': f1_m, 'f1_weighted': f1_w,
        'precision_macro': prec_m, 'recall_macro': rec_m,
        'per_class_precision': prec, 'per_class_recall': rec,
        'per_class_f1': f1, 'per_class_support': sup,
        'confusion_matrix': cm, 'avg_confidence': confidences.mean(),
        'class_names': list(class_names),
    }


def save_per_audio_csv(results, save_path="per_audio_scores.csv"):
    df = pd.DataFrame({
        'path':       results['paths'],
        'prob_fake':  results['probabilities'][:, 0],
        'prob_real':  results['probabilities'][:, 1],
        'prediction': ['real' if p == 1 else 'fake' for p in results['predictions']],
        'confidence': results['confidences'],
        'threshold':  results.get('threshold', 0.5),
    })
    if 'labels' in results:
        df['label']        = ['real' if l == 1 else 'fake' for l in results['labels']]
        df['label_5class'] = results['labels_5class']
        df['correct']      = (results['predictions'] == results['labels']).astype(int)

    df.to_csv(save_path, index=False)
    logger.info(f"Per-audio CSV saved → {save_path}")
    return df


def save_summary(metrics, threshold, save_path="summary.txt"):
    if metrics is None:
        return
    lines = [
        "=" * 55,
        "CCL INFERENCE SUMMARY",
        "=" * 55,
        f"Threshold        : {threshold:.4f}",
        f"Accuracy         : {metrics['accuracy']:.4f}",
        f"EER              : {metrics['eer'] * 100:.2f}%",
        f"EER Threshold    : {metrics['eer_threshold']:.4f}",
        f"AUC-ROC          : {metrics['auc']:.4f}",
        f"Macro F1         : {metrics['f1_macro']:.4f}",
        f"Weighted F1      : {metrics['f1_weighted']:.4f}",
        f"Avg Confidence   : {metrics['avg_confidence']:.4f}",
        "",
        "Per-class:",
        f"  {'Class':<8}  {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support'}",
        "-" * 55,
    ]
    for i, cn in enumerate(metrics['class_names']):
        lines.append(
            f"  {cn:<8}  {metrics['per_class_precision'][i]:<10.4f} "
            f"{metrics['per_class_recall'][i]:<10.4f} "
            f"{metrics['per_class_f1'][i]:<10.4f} "
            f"{metrics['per_class_support'][i]}"
        )
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Summary saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ckpt_path   = args.load_ckpt_path
    mode        = args.mode
    json_file   = args.val_json_file
    output_dir  = getattr(args, 'output_dir', '.')
    embed_dim   = getattr(args, 'embed_dim', 527)
    num_classes = 5

    # real_class_idx differs per dataset because BaselineDataset sorts label keys:
    #   training set (efs, fr, fs, fake, real) → real=4
    #   test set     (fake, real)              → real=1
    if mode == 'beats':
        _ds_tmp = BeatsDataset(json_file=json_file, transformation=None, args=args)
    else:
        _ds_tmp = BaselineDataset(json_file=json_file, transformation=None, args=args)
    real_class_idx = _ds_tmp.label.get('real', 4)
    logger.info(f"Label mapping: {_ds_tmp.label}")
    logger.info(f"real_class_idx auto-detected: {real_class_idx}")

    # --threshold accepts a float in [0,1] OR the literal 'eer'.
    raw_thr     = str(getattr(args, 'threshold', '0.5')).strip().lower()
    use_eer_thr = raw_thr == 'eer'
    init_thr    = 0.5 if use_eer_thr else float(raw_thr)

    os.makedirs(output_dir, exist_ok=True)

    if mode == 'beats':
        dataset = BeatsDataset(json_file=json_file, transformation=None, args=args)
    else:
        dataset = BaselineDataset(json_file=json_file, transformation=None, args=args)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False)

    pipeline, ccl_head, arc_head = load_pipeline_and_ccl(
        checkpoint_path=ckpt_path,
        device=device,
        mode=mode,
        num_classes=num_classes,
        embed_dim=embed_dim,
        truncate_layers=getattr(args, 'truncate_layers', 0),
        beats_feature=getattr(args, 'beats_feature', 'predictor'),
    )

    results = run_inference(pipeline, ccl_head, arc_head, dataloader,
                             device, real_class_idx=real_class_idx)
    apply_threshold(results, threshold=init_thr)

    if use_eer_thr and 'labels' in results:
        eer_val, eer_thr, _, _ = compute_eer(results['labels'], results['probabilities'][:, 1])
        logger.info(f"EER mode → threshold set to {eer_thr:.4f} (EER={eer_val*100:.2f}%)")
        apply_threshold(results, threshold=eer_thr)

    active_thr = results.get('threshold', init_thr)

    metrics = evaluate(results, class_names=('fake', 'real'), output_dir=output_dir)

    save_per_audio_csv(results, save_path=os.path.join(output_dir, "per_audio_scores.csv"))
    save_generator_breakdown(
        results,
        csv_path=os.path.join(output_dir, "generator_score.csv"),
        npy_path=os.path.join(output_dir, "generator_score.npy"),
    )
    save_summary(metrics, threshold=active_thr,
                 save_path=os.path.join(output_dir, "summary.txt"))

    if metrics:
        print("\n" + "=" * 50)
        print("CCL INFERENCE COMPLETE")
        print("=" * 50)
        print(f"Threshold  : {active_thr:.4f}")
        print(f"Accuracy   : {metrics['accuracy']:.4f}")
        print(f"EER        : {metrics['eer'] * 100:.2f}%")
        print(f"AUC-ROC    : {metrics['auc']:.4f}")
        print(f"Macro F1   : {metrics['f1_macro']:.4f}")
        print(f"Outputs    : {output_dir}/")
        print("=" * 50)

    # ── Side-by-side comparison: CCL vs ArcFace heads ────────────────────────
    if 'labels' in results:
        ccl_summary = per_class_summary(results, 'ccl')
        arc_summary = per_class_summary(results, 'arc')

        print("\n" + "=" * 60)
        print("HEAD COMPARISON — CCL vs ArcFace")
        print("=" * 60)
        print(f"  n_real = {ccl_summary['n_real']}   n_fake = {ccl_summary['n_fake']}")
        print(f"  {'metric':<12} {'CCL':>10} {'ArcFace':>10}    Δ (arc−ccl)")
        print(f"  {'-'*12} {'-'*10} {'-'*10}    {'-'*12}")
        for k, label in [('real_acc', 'real_acc'),
                         ('fake_acc', 'fake_acc'),
                         ('accuracy', 'accuracy'),
                         ('auc',      'auc-roc'),
                         ('eer',      'eer')]:
            c, a = ccl_summary[k], arc_summary[k]
            delta = a - c
            fmt = (lambda v: f"{v*100:>9.2f}%") if k == 'eer' else (lambda v: f"{v:>10.4f}")
            print(f"  {label:<12} {fmt(c)} {fmt(a)}    {delta:+.4f}")
        print("=" * 60)

        # CSV with both heads' per-audio predictions for further analysis
        compare_path = os.path.join(output_dir, "head_compare_per_audio.csv")
        compare_df = pd.DataFrame({
            'path':              results['paths'],
            'label':             ['real' if l == 1 else 'fake' for l in results['labels']],
            'label_5class':      results['labels_5class'],
            'ccl_pred':          ['real' if p == 1 else 'fake' for p in results['ccl_predictions']],
            'ccl_prob_real':     results['ccl_probabilities'][:, 1],
            'ccl_correct':       (results['ccl_predictions'] == results['labels']).astype(int),
            'arc_pred':          ['real' if p == 1 else 'fake' for p in results['arc_predictions']],
            'arc_prob_real':     results['arc_probabilities'][:, 1],
            'arc_correct':       (results['arc_predictions'] == results['labels']).astype(int),
            'agree':             (results['ccl_predictions'] == results['arc_predictions']).astype(int),
        })
        compare_df.to_csv(compare_path, index=False)
        print(f"Per-audio head comparison saved → {compare_path}")
