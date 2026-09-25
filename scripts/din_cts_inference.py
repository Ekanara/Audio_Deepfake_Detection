"""
DIN-CTS Inference  (Stage 3 + Evaluation)
==========================================
1. Loads Stage-2 checkpoint (BEATs backbone + entropy head)
2. Extracts backbone embeddings from a bonafide reference set
3. Fits a Gaussian  N(μ, Σ)  over bonafide embeddings
4. Scores test utterances via Mahalanobis distance
5. Reports Acc, F1, AUC, EER

Usage
-----
python scripts/din_cts_inference.py \\
    --ckpt checkpoint/BEATs_CTS/stage2/checkpoint/best.pt \\
    --real_ref_json  data/label/gam/TUTASC19_train.json \\
    --test_json      data/label/gam/TUTASC19_test.json \\
    --prev_num_label 5

Optional: skip Mahalanobis and use the Entropy head directly:
    --mode entropy
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from beats.model_beat import model_beat
from base_dataset import BeatsDataset


# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thr[idx]


def load_pipeline(ckpt_path, prev_num_label, beats_feature, device):
    """Load Stage-2 pipeline (backbone weights) from checkpoint."""
    pipeline = model_beat(
        num_label=prev_num_label,
        three_loss=True,
        feature_layer=beats_feature,
    )

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    pipe_sd = {}
    for k, v in state.items():
        if k.startswith("pipeline."):
            pipe_sd[k[len("pipeline."):]] = v
    if not pipe_sd:
        pipe_sd = state

    pipeline.load_state_dict(pipe_sd, strict=False)
    pipeline.to(device).eval()
    print(f"Loaded pipeline from: {ckpt_path}")

    # Also try to load entropy_head weights
    entropy_head = torch.nn.Linear(pipeline.feature_dim, 2)
    head_sd = {}
    for k, v in state.items():
        if k.startswith("entropy_head."):
            head_sd[k[len("entropy_head."):]] = v
    if head_sd:
        entropy_head.load_state_dict(head_sd)
        print("Loaded entropy_head weights")
    else:
        print("No entropy_head weights found, using random init (use --mode mahalanobis)")
    entropy_head.to(device).eval()

    return pipeline, entropy_head


def make_dataloader(json_file, args, batch_size=64, num_workers=4):
    ds = BeatsDataset(json_file=json_file, transformation=None, args=args)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    real_idx = ds.label.get("real", 1)
    return dl, real_idx


# ─── Embedding extraction ───────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(pipeline, dataloader, device, real_idx=None, only_real=False):
    """
    Extract backbone embeddings.

    Returns:
        embeddings : np.ndarray [N, D]
        labels     : np.ndarray [N]   (1=real, 0=fake)
    """
    all_embs, all_labels = [], []
    for batch in dataloader:
        audio  = batch["audio"].to(device, dtype=torch.float32)
        labels = batch["label"].cpu().numpy()

        outputs = pipeline.forward_pipeline(audio)
        bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs
        embs = bonafide_head.cpu().numpy()

        binary_labels = (labels == real_idx).astype(int) if real_idx is not None else labels

        if only_real:
            mask = binary_labels == 1
            if mask.any():
                all_embs.append(embs[mask])
                all_labels.append(binary_labels[mask])
        else:
            all_embs.append(embs)
            all_labels.append(binary_labels)

    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)


# ─── Gaussian distribution ──────────────────────────────────────────────────

def fit_gaussian(embeddings):
    """
    Fit N(μ, Σ) to embeddings and return (μ, Σ⁻¹).

    Uses regularized pseudo-inverse for numerical stability.
    """
    mu = np.mean(embeddings, axis=0)
    centered = embeddings - mu
    sigma = np.cov(centered, rowvar=False)

    # Regularise: add small diagonal to avoid singular Σ
    sigma += np.eye(sigma.shape[0]) * 1e-6
    sigma_inv = np.linalg.inv(sigma)

    print(f"  Gaussian fitted on {embeddings.shape[0]} bonafide samples, dim={embeddings.shape[1]}")
    return mu, sigma_inv


def mahalanobis_distance(x, mu, sigma_inv):
    """
    Compute Mahalanobis distance for each row of x.

    d_i = sqrt( (x_i - μ)^T  Σ⁻¹  (x_i - μ) )

    Returns: np.ndarray [N]
    """
    diff = x - mu  # [N, D]
    left = diff @ sigma_inv  # [N, D]
    dist = np.sqrt(np.sum(left * diff, axis=1))  # [N]
    return dist


# ─── Entropy head inference ─────────────────────────────────────────────────

@torch.no_grad()
def run_entropy_inference(pipeline, entropy_head, dataloader, device, real_idx):
    """Inference using the Stage-2 Entropy head (FC→softmax→2-class)."""
    all_probs, all_labels = [], []
    for batch in dataloader:
        audio  = batch["audio"].to(device, dtype=torch.float32)
        labels = batch["label"].cpu().numpy()
        outputs = pipeline.forward_pipeline(audio)
        bonafide_head = outputs[0] if isinstance(outputs, tuple) else outputs
        logits = entropy_head(bonafide_head)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append((labels == real_idx).astype(int))

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs[:, 1], labels  # prob_real, binary_labels


# ─── Metrics ─────────────────────────────────────────────────────────────────

def print_metrics(name, scores, labels, higher_is_real=True):
    """
    Print evaluation metrics.

    Args:
        scores:  per-sample score (higher = more real if higher_is_real)
        labels:  binary (1=real, 0=fake)
    """
    if not higher_is_real:
        # For Mahalanobis: lower distance = more real → negate
        scores_for_roc = -scores
    else:
        scores_for_roc = scores

    auc = roc_auc_score(labels, scores_for_roc)
    eer, eer_thr = compute_eer(labels, scores_for_roc)

    # Use EER threshold for binary predictions
    preds = (scores_for_roc >= eer_thr).astype(int)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")

    # Per-class metrics
    real_mask = labels == 1
    fake_mask = labels == 0
    real_acc = float((preds[real_mask] == 1).mean()) if real_mask.any() else 0.0
    fake_acc = float((preds[fake_mask] == 0).mean()) if fake_mask.any() else 0.0
    real_f1 = f1_score(labels, preds, average="binary", pos_label=1)
    fake_f1 = f1_score(1 - labels, 1 - preds, average="binary", pos_label=1)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.2f}")
    print(f"  F1:        {f1:.2f}")
    print(f"  AUC:       {auc:.2f}")
    print(f"  EER:       {eer:.2f}")
    print(f"  Real Acc:  {real_acc:.2f}   Fake Acc:  {fake_acc:.2f}")
    print(f"  Real F1:   {real_f1:.2f}   Fake F1:   {fake_f1:.2f}")

    return {"name": name, "acc": acc, "f1": f1, "auc": auc, "eer": eer,
            "real_acc": real_acc, "fake_acc": fake_acc,
            "real_f1": real_f1, "fake_f1": fake_f1}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DIN-CTS Inference (Mahalanobis / Entropy)")
    parser.add_argument("--ckpt", required=True, help="Path to Stage-2 checkpoint (.pt)")
    parser.add_argument("--real_ref_json", default=None,
                        help="JSON with bonafide reference samples (for Mahalanobis). "
                             "If None, uses training split from --test_json dataset.")
    parser.add_argument("--test_json", required=True, help="Test set JSON")
    parser.add_argument("--prev_num_label", type=int, default=5,
                        help="num_label used in Stage-1 (to reconstruct model arch)")
    parser.add_argument("--beats_feature", default="predictor", choices=["predictor", "encoder"])
    parser.add_argument("--mode", default="both", choices=["mahalanobis", "entropy", "both"],
                        help="Inference mode: mahalanobis, entropy, or both")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dist", default=None,
                        help="Save bonafide distribution (mu, sigma_inv) to .pt file")
    parser.add_argument("--load_dist", default=None,
                        help="Load pre-computed bonafide distribution from .pt file")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Minimal args namespace for dataset
    ds_args = argparse.Namespace(
        mode="beats", num_label=2, get_first_dim=False,
    )

    # Load model
    pipeline, entropy_head = load_pipeline(
        args.ckpt, args.prev_num_label, args.beats_feature, device,
    )

    # Test dataloader
    test_dl, test_real_idx = make_dataloader(args.test_json, ds_args, args.batch_size)

    results = []

    # ── Mahalanobis inference ────────────────────────────────────────────
    if args.mode in ("mahalanobis", "both"):
        if args.load_dist:
            dist_data = torch.load(args.load_dist, weights_only=False)
            mu, sigma_inv = dist_data["mu"], dist_data["sigma_inv"]
            print(f"Loaded bonafide distribution from: {args.load_dist}")
        else:
            ref_json = args.real_ref_json or args.test_json
            ref_dl, ref_real_idx = make_dataloader(ref_json, ds_args, args.batch_size)
            print(f"\nExtracting bonafide embeddings from: {ref_json}")
            ref_embs, _ = extract_embeddings(pipeline, ref_dl, device, ref_real_idx, only_real=True)
            mu, sigma_inv = fit_gaussian(ref_embs)

            if args.save_dist:
                torch.save({"mu": mu, "sigma_inv": sigma_inv}, args.save_dist)
                print(f"Saved distribution to: {args.save_dist}")

        # Extract test embeddings
        print(f"\nExtracting test embeddings from: {args.test_json}")
        test_embs, test_labels = extract_embeddings(pipeline, test_dl, device, test_real_idx)

        # Compute Mahalanobis distances
        distances = mahalanobis_distance(test_embs, mu, sigma_inv)
        r = print_metrics("Mahalanobis", distances, test_labels, higher_is_real=False)
        results.append(r)

    # ── Entropy head inference ───────────────────────────────────────────
    if args.mode in ("entropy", "both"):
        prob_real, test_labels = run_entropy_inference(
            pipeline, entropy_head, test_dl, device, test_real_idx,
        )
        r = print_metrics("Entropy Head", prob_real, test_labels, higher_is_real=True)
        results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n{'='*50}")
        print(f"  COMPARISON")
        print(f"{'='*50}")
        print(f"  {'Method':<16} {'Acc':>6} {'F1':>6} {'AUC':>6} {'EER':>6}")
        print(f"  {'-'*42}")
        for r in results:
            print(f"  {r['name']:<16} {r['acc']:.2f}  {r['f1']:.2f}  {r['auc']:.2f}  {r['eer']:.2f}")


if __name__ == "__main__":
    main()
