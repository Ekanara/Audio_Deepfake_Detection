"""
Combined inference: Event_test + test_track2 as one test set.

test_track2 labels are remapped:
  fake → fake_challenge
  real → real_challenge

Runs 5-class softmax inference on both DIN-CTS and LL_fake_only checkpoints
to see which clusters the test_track2 samples fall into.

Also generates t-SNE plots with all ground truth labels.
"""
import os, sys, json, gc
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from argparse import Namespace
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from beats.model_beat import model_beat
from utils.training_utils import CenterContrastiveLoss

SOFTMAX_MAP = {0: "fake_ata_01", 1: "fake_tta_01", 2: "fake_tta_02", 3: "fake_tta_03", 4: "real"}

COLORS = {
    "real":            "#00802b",   # bold dark green
    "fake_ata_01":     "#cc0000",   # bold dark red
    "fake_tta_01":     "#0044cc",   # bold dark blue
    "fake_tta_02":     "#cc8400",   # bold dark yellow/amber
    "fake_tta_03":     "#7b2d8b",   # bold dark purple
    "fake_unknown":    "#4a4a4a",   # bold dark gray
    "real_challenge":  "#0066cc",   # bold blue
    "fake_challenge":  "#ff6600",   # bold orange
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def fit_gaussian(X, eps=1e-6):
    X = X.astype(np.float64)
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    cf, lower = cho_factor(cov)
    _, logdet = np.linalg.slogdet(cov)
    return mu, cf, lower, logdet


def loglik(X, g):
    mu, cf, lower, logdet = g
    diff = X.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    return -0.5 * (np.einsum("ij,ij->i", diff, solved) + logdet)


def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thr[idx]


# ── Build combined JSON ──────────────────────────────────────────────────────
def build_combined_json(event_json, track2_json, out_json):
    """Combine Event_test (5-class) + test_track2, relabeling track2 as *_challenge."""
    with open(event_json) as f:
        event_data = json.load(f)
    with open(track2_json) as f:
        track2_data = json.load(f)

    combined = []
    for entry in event_data:
        combined.append(entry)  # keep 5-class labels (fake_ata_01, fake_tta_01, ..., real)

    for entry in track2_data:
        new_entry = dict(entry)
        if entry["label"] == "real":
            new_entry["label"] = "real_challenge"
        elif entry["label"] == "fake":
            new_entry["label"] = "fake_challenge"
        combined.append(new_entry)

    with open(out_json, "w") as f:
        json.dump(combined, f, indent=2)

    # Count labels
    counts = Counter(e["label"] for e in combined)
    print(f"\n  Combined dataset: {len(combined)} samples")
    for lbl, cnt in sorted(counts.items()):
        print(f"    {lbl}: {cnt}")
    return out_json


# ── Load checkpoint ──────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)
    pipe_state = {}
    for k, v in state_dict.items():
        if k.startswith("pipeline."):
            pipe_state[k[9:]] = v
    cur = pipeline.state_dict()
    compat = {k: v for k, v in pipe_state.items() if k in cur and v.shape == cur[k].shape}
    pipeline.load_state_dict(compat, strict=False)
    print(f"  Pipeline: loaded {len(compat)}/{len(cur)} keys")

    # Also load CCL for distance-based scoring
    ccl_head = CenterContrastiveLoss(embed_dim=527, n_classes=2, m=0.7, s=30, lambda_c=2)
    ccl_state = {}
    for k, v in state_dict.items():
        if k.startswith("center_constrastive_loss."):
            ccl_state[k.replace("center_constrastive_loss.", "")] = v
    if ccl_state:
        ccl_head.load_state_dict(ccl_state, strict=False)
        print(f"  CCL head: loaded {len(ccl_state)} keys")
    ccl_head = ccl_head.to(device)

    return pipeline, ccl_head


@torch.no_grad()
def collect_features(pipeline, json_file, device):
    """Collect bonafide_head, softmax_head, and string labels."""
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv_label = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_bn, all_sm, all_lbl = [], [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        all_sm.append(outputs[1].float().cpu().numpy())
        if "label" in batch:
            all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    bn = np.concatenate(all_bn)
    sm = np.concatenate(all_sm)
    lbl_int = np.array(all_lbl)
    # Convert int labels to string names
    lbl_str = np.array([inv_label[i] for i in lbl_int])
    return bn, sm, lbl_int, lbl_str, inv_label


# ── Analysis functions ───────────────────────────────────────────────────────
def analyze_softmax_distribution(sm_probs, gt_labels, ckpt_name):
    """For each ground truth label, show distribution across 5 softmax classes."""
    unique_gt = sorted(set(gt_labels))
    sm_preds = sm_probs.argmax(axis=1)
    sm_pred_names = np.array([SOFTMAX_MAP[i] for i in sm_preds])

    print(f"\n  === 5-Class Softmax Distribution ({ckpt_name}) ===")
    print(f"  {'GT Label':20s} {'Count':>6s} | {'fake_ata':>9s} {'fake_tta1':>9s} {'fake_tta2':>9s} {'fake_tta3':>9s} {'real':>9s}")
    print("  " + "-" * 85)

    for gt in unique_gt:
        mask = gt_labels == gt
        n = mask.sum()
        preds_subset = sm_preds[mask]
        dist = {}
        for cls_id in range(5):
            pct = (preds_subset == cls_id).sum() / n * 100
            dist[cls_id] = pct
        print(f"  {gt:20s} {n:6d} | {dist[0]:8.1f}% {dist[1]:8.1f}% {dist[2]:8.1f}% {dist[3]:8.1f}% {dist[4]:8.1f}%")

    # Also show mean softmax probability per GT label
    print(f"\n  === Mean Softmax Probabilities ({ckpt_name}) ===")
    print(f"  {'GT Label':20s} {'Count':>6s} | {'p(ata)':>9s} {'p(tta1)':>9s} {'p(tta2)':>9s} {'p(tta3)':>9s} {'p(real)':>9s}")
    print("  " + "-" * 85)
    for gt in unique_gt:
        mask = gt_labels == gt
        n = mask.sum()
        mean_p = sm_probs[mask].mean(axis=0)
        print(f"  {gt:20s} {n:6d} | {mean_p[0]:8.3f}  {mean_p[1]:8.3f}  {mean_p[2]:8.3f}  {mean_p[3]:8.3f}  {mean_p[4]:8.3f} ")


def analyze_ll_scores(bn_features, gt_labels, ref_fake_bn, ref_real_bn, ckpt_name,
                      real_ref_only=False):
    """Show LL scores breakdown per GT label."""
    gR = fit_gaussian(ref_real_bn)
    ll_real = loglik(bn_features, gR)
    unique_gt = sorted(set(gt_labels))

    if real_ref_only:
        # DIN-CTS native: score = LL(test, real_gaussian), higher = more real
        print(f"\n  === LL Score Statistics — real-ref only ({ckpt_name}) ===")
        print(f"  {'GT Label':20s} {'Count':>6s} | {'LL_real mean':>12s}")
        print("  " + "-" * 45)
        for gt in unique_gt:
            mask = gt_labels == gt
            n = mask.sum()
            print(f"  {gt:20s} {n:6d} | {ll_real[mask].mean():12.1f}")

        score = ll_real  # higher = more real
        binary_labels = np.isin(gt_labels, ["real", "real_challenge"]).astype(int)
        if len(np.unique(binary_labels)) >= 2:
            auc = roc_auc_score(binary_labels, score)
            fpr, tpr, thrs = roc_curve(binary_labels, score, pos_label=1)
            fnr = 1.0 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
            thr = thrs[eer_idx]
            preds = (score >= thr).astype(int)

            print(f"\n  LL_real scoring: AUC={auc:.4f}, EER={eer*100:.2f}%")
            print(f"  {'GT Label':20s} {'Pred Real':>10s} {'Pred Fake':>10s} {'Acc':>8s}")
            print("  " + "-" * 55)
            for gt in unique_gt:
                mask = gt_labels == gt
                n = mask.sum()
                pred_real = (preds[mask] == 1).sum()
                pred_fake = (preds[mask] == 0).sum()
                is_real = gt in ("real", "real_challenge")
                acc = pred_real / n if is_real else pred_fake / n
                print(f"  {gt:20s} {pred_real:10d} {pred_fake:10d} {acc*100:7.1f}%")
        return

    gF = fit_gaussian(ref_fake_bn)
    ll_fake = loglik(bn_features, gF)
    score_baseline = -ll_fake        # higher = more real
    score_lr = ll_real - ll_fake     # higher = more real

    print(f"\n  === LL Score Statistics ({ckpt_name}) ===")
    print(f"  {'GT Label':20s} {'Count':>6s} | {'LL_fake mean':>12s} {'LL_real mean':>12s} {'LR mean':>12s} {'baseline mean':>13s}")
    print("  " + "-" * 90)
    for gt in unique_gt:
        mask = gt_labels == gt
        n = mask.sum()
        print(f"  {gt:20s} {n:6d} | {ll_fake[mask].mean():12.1f} {ll_real[mask].mean():12.1f} "
              f"{score_lr[mask].mean():12.1f} {score_baseline[mask].mean():13.1f}")

    # Per-GT accuracy at EER threshold (using all samples)
    binary_labels = np.isin(gt_labels, ["real", "real_challenge"]).astype(int)
    if len(np.unique(binary_labels)) >= 2:
        for method_name, scores in [("baseline (-LL_fake)", score_baseline), ("LR", score_lr)]:
            auc = roc_auc_score(binary_labels, scores)
            fpr, tpr, thrs = roc_curve(binary_labels, scores, pos_label=1)
            fnr = 1.0 - tpr
            eer_idx = np.argmin(np.abs(fpr - fnr))
            eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
            thr = thrs[eer_idx]
            preds = (scores >= thr).astype(int)

            print(f"\n  {method_name}: AUC={auc:.4f}, EER={eer*100:.2f}%")
            print(f"  {'GT Label':20s} {'Pred Real':>10s} {'Pred Fake':>10s} {'Acc':>8s}")
            print("  " + "-" * 55)
            for gt in unique_gt:
                mask = gt_labels == gt
                n = mask.sum()
                pred_real = (preds[mask] == 1).sum()
                pred_fake = (preds[mask] == 0).sum()
                is_real = gt in ("real", "real_challenge")
                acc = pred_real / n if is_real else pred_fake / n
                print(f"  {gt:20s} {pred_real:10d} {pred_fake:10d} {acc*100:7.1f}%")


def plot_tsne_3label(embs, gt_labels, ckpt_name, out_dir):
    """t-SNE with 3 labels: real + real_challenge same green, fake_challenge red."""
    max_per_class = 1500
    np.random.seed(42)

    idx_parts = []
    for gt in sorted(set(gt_labels)):
        gt_idx = np.where(gt_labels == gt)[0]
        if len(gt_idx) > max_per_class:
            gt_idx = np.random.choice(gt_idx, max_per_class, replace=False)
        idx_parts.append(gt_idx)
    idx = np.sort(np.concatenate(idx_parts))

    sub_embs = embs[idx]
    sub_labels = gt_labels[idx]

    print(f"  Running t-SNE (3-label) on {len(idx)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    embs_2d = tsne.fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(12, 9))
    # Plot order: fake first (background), then reals on top
    for gt in ["fake_challenge", "real", "real_challenge"]:
        if gt not in set(sub_labels):
            continue
        m = sub_labels == gt
        if gt == "fake_challenge":
            color, marker, alpha, size = "#cc0000", "^", 0.85, 22  # red triangle
        elif gt == "real":
            color, marker, alpha, size = "#ff8800", "o", 0.7, 14  # orange circle
        else:  # real_challenge
            color, marker, alpha, size = "#00802b", "^", 0.85, 22  # green triangle
        ax.scatter(embs_2d[m, 0], embs_2d[m, 1], c=color, label=gt,
                   alpha=alpha, s=size, marker=marker, edgecolors="none")
    ax.legend(fontsize=11, markerscale=2.5, loc="best")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    title = f"t-SNE real + test_track2 ({ckpt_name})"
    path = os.path.join(out_dir, f"{title}.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE saved: {path}")


def plot_tsne_combined(embs, gt_labels, ckpt_name, out_dir, binary=False):
    """t-SNE with ground truth labels color-coded."""
    if binary:
        # Map to 4 labels: real, fake, real_challenge, fake_challenge
        def map_label(l):
            if l == "real_challenge":
                return "real_challenge"
            elif l == "fake_challenge":
                return "fake_challenge"
            elif "real" in l:
                return "real"
            else:
                return "fake"
        mapped = np.array([map_label(l) for l in gt_labels])
    else:
        mapped = gt_labels

    unique_gt = sorted(set(mapped))
    max_per_class = 1000 if binary else 500
    np.random.seed(42)

    idx_parts = []
    for gt in unique_gt:
        gt_idx = np.where(mapped == gt)[0]
        if len(gt_idx) > max_per_class:
            gt_idx = np.random.choice(gt_idx, max_per_class, replace=False)
        idx_parts.append(gt_idx)
    idx = np.sort(np.concatenate(idx_parts))

    sub_embs = embs[idx]
    sub_labels = mapped[idx]

    print(f"  Running t-SNE ({'2-class' if binary else '8-class'}) on {len(idx)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    embs_2d = tsne.fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(12, 9))

    if binary:
        colors_4c = {
            "real":            "#00802b",   # bold dark green
            "fake":            "#cc0000",   # bold dark red
            "real_challenge":  "#00802b",   # same green as real
            "fake_challenge":  "#ff6600",   # bold orange
        }
        plot_order_4c = ["fake", "real", "fake_challenge", "real_challenge"]
        for gt in plot_order_4c:
            if gt not in set(sub_labels):
                continue
            m = sub_labels == gt
            marker = "^" if "challenge" in gt else "o"
            alpha = 0.85 if "challenge" in gt else 0.7
            size = 25 if "challenge" in gt else 14
            ax.scatter(embs_2d[m, 0], embs_2d[m, 1], c=colors_4c[gt], label=gt,
                       alpha=alpha, s=size, marker=marker, edgecolors="none")
    else:
        plot_order = ["real", "fake_ata_01", "fake_tta_01", "fake_tta_02",
                      "fake_tta_03", "fake_unknown", "real_challenge", "fake_challenge"]
        for gt in plot_order:
            if gt not in set(sub_labels):
                continue
            m = sub_labels == gt
            color = COLORS.get(gt, "#999999")
            marker = "^" if "challenge" in gt else "o"
            alpha = 0.85 if "challenge" in gt else 0.7
            size = 25 if "challenge" in gt else 14
            ax.scatter(embs_2d[m, 0], embs_2d[m, 1], c=color, label=gt,
                       alpha=alpha, s=size, marker=marker, edgecolors="none")

    ax.legend(fontsize=10, markerscale=2.5, loc="best")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    label_tag = "4-label " if binary else ""
    title = f"t-SNE Event_test + test_track2 {label_tag}({ckpt_name})"
    path = os.path.join(out_dir, f"{title.strip()}.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "output/combined_analysis"
    os.makedirs(out_dir, exist_ok=True)

    # Paths — use 5-class Event_test labels
    event_json = "data/label/beats/Event_test_5class.json"
    track2_json = "data/label/beats/test_track2.json"
    combined_json = "data/label/beats/combined_event_track2.json"
    ref_fake_json = "data/label/beats/Event_train_stage1_fakeonly.json"
    ref_real_json = "data/label/beats/Event_train_stage1_realonly.json"

    # Checkpoints: (path, real_ref_only)
    checkpoints = {
        "LL_fake_only": ("checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt", False),
        "retrain_run1_ep06": ("checkpoint/Beats_journal/retrain_cclW8_ceW3_ceRW8_LR5e-08_8ep/epoch-06_bal0.9653.ckpt", False),
    }

    # Build combined JSON
    print("=" * 80)
    print("Building combined test set...")
    build_combined_json(event_json, track2_json, combined_json)

    for ckpt_name, (ckpt_path, real_ref_only) in checkpoints.items():
        print(f"\n{'=' * 80}")
        print(f"  Checkpoint: {ckpt_name}")
        print(f"  Path: {ckpt_path}")
        print(f"  Scoring: {'real-ref only (DIN-CTS native)' if real_ref_only else 'LL_fake + LR'}")
        print(f"{'=' * 80}")

        # Load model
        pipeline, ccl_head = load_checkpoint(ckpt_path, device)

        # Collect features on combined set
        print("\n  Collecting features on combined test set...")
        bn, sm, lbl_int, lbl_str, inv_label = collect_features(pipeline, combined_json, device)
        print(f"  Features: {bn.shape[0]} samples, {bn.shape[1]}-dim bonafide_head")

        # Collect reference features for LL scoring
        print("  Collecting fake reference features...")
        ref_fake_bn, _, _, _, _ = collect_features(pipeline, ref_fake_json, device)
        print("  Collecting real reference features...")
        ref_real_bn, _, _, _, _ = collect_features(pipeline, ref_real_json, device)

        # 1. Softmax distribution analysis
        analyze_softmax_distribution(sm, lbl_str, ckpt_name)

        # 2. LL score analysis
        analyze_ll_scores(bn, lbl_str, ref_fake_bn, ref_real_bn, ckpt_name,
                          real_ref_only=real_ref_only)

        # 2b. test_track2-only scores (fake_challenge + real_challenge)
        track2_mask = np.isin(lbl_str, ["fake_challenge", "real_challenge"])
        if track2_mask.sum() > 0:
            t2_bn = bn[track2_mask]
            t2_labels = lbl_str[track2_mask]
            t2_binary = (t2_labels == "real_challenge").astype(int)

            gF = fit_gaussian(ref_fake_bn)
            gR = fit_gaussian(ref_real_bn)
            t2_ll_fake = loglik(t2_bn, gF)
            t2_ll_real = loglik(t2_bn, gR)
            t2_baseline = -t2_ll_fake
            t2_lr = t2_ll_real - t2_ll_fake

            print(f"\n  === test_track2-ONLY scores ({ckpt_name}) ===")
            print(f"  Samples: {track2_mask.sum()} (fake_challenge={sum(t2_labels=='fake_challenge')}, real_challenge={sum(t2_labels=='real_challenge')})")

            for name, sc in [("baseline (-LL_fake)", t2_baseline), ("LR", t2_lr)]:
                auc = roc_auc_score(t2_binary, sc)
                fpr, tpr, thrs = roc_curve(t2_binary, sc, pos_label=1)
                fnr = 1.0 - tpr
                eer_idx = np.argmin(np.abs(fpr - fnr))
                eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
                thr = thrs[eer_idx]
                preds = (sc >= thr).astype(int)
                fake_acc = (preds[t2_binary == 0] == 0).mean() * 100
                real_acc = (preds[t2_binary == 1] == 1).mean() * 100
                bal_acc = (fake_acc + real_acc) / 2
                print(f"\n  {name}:")
                print(f"    AUC={auc:.4f}, EER={eer*100:.2f}%")
                print(f"    fake_challenge acc: {fake_acc:.1f}%")
                print(f"    real_challenge acc: {real_acc:.1f}%")
                print(f"    balanced acc:       {bal_acc:.1f}%")

                # Score distribution
                print(f"    fake_challenge score: mean={sc[t2_binary==0].mean():.1f}, std={sc[t2_binary==0].std():.1f}")
                print(f"    real_challenge score: mean={sc[t2_binary==1].mean():.1f}, std={sc[t2_binary==1].std():.1f}")

        # 3. t-SNE comparison: raw vs L2-norm vs PCA+L2
        keep_mask = np.isin(lbl_str, ["real", "fake_challenge", "real_challenge"])
        keep_bn = bn[keep_mask]
        keep_lbl = lbl_str[keep_mask]

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import normalize

        # Also score each variant on test_track2
        t2_mask_k = np.isin(keep_lbl, ["fake_challenge", "real_challenge"])
        t2_binary_k = (keep_lbl[t2_mask_k] == "real_challenge").astype(int)

        variants = {}

        # (a) Raw (original)
        variants["raw"] = (keep_bn, ref_fake_bn, ref_real_bn)

        # (b) L2-normalized
        bn_l2 = normalize(keep_bn, norm="l2")
        ref_fake_l2 = normalize(ref_fake_bn, norm="l2")
        ref_real_l2 = normalize(ref_real_bn, norm="l2")
        variants["L2-norm"] = (bn_l2, ref_fake_l2, ref_real_l2)

        # (c) PCA 128 + L2
        pca = PCA(n_components=128, random_state=42)
        pca.fit(np.vstack([ref_fake_bn, ref_real_bn]))  # fit on reference
        bn_pca = normalize(pca.transform(keep_bn), norm="l2")
        ref_fake_pca = normalize(pca.transform(ref_fake_bn), norm="l2")
        ref_real_pca = normalize(pca.transform(ref_real_bn), norm="l2")
        variants["PCA128-L2"] = (bn_pca, ref_fake_pca, ref_real_pca)

        # (d) PCA 64 + L2
        pca64 = PCA(n_components=64, random_state=42)
        pca64.fit(np.vstack([ref_fake_bn, ref_real_bn]))
        bn_pca64 = normalize(pca64.transform(keep_bn), norm="l2")
        ref_fake_pca64 = normalize(pca64.transform(ref_fake_bn), norm="l2")
        ref_real_pca64 = normalize(pca64.transform(ref_real_bn), norm="l2")
        variants["PCA64-L2"] = (bn_pca64, ref_fake_pca64, ref_real_pca64)

        for vname, (v_bn, v_rf, v_rr) in variants.items():
            # Score
            gF = fit_gaussian(v_rf)
            gR = fit_gaussian(v_rr)
            ll_f = loglik(v_bn[t2_mask_k], gF)
            ll_r = loglik(v_bn[t2_mask_k], gR)
            sc_base = -ll_f
            sc_lr = ll_r - ll_f

            print(f"\n  --- {vname} test_track2 scores ---")
            for sname, sc in [("baseline", sc_base), ("LR", sc_lr)]:
                auc = roc_auc_score(t2_binary_k, sc)
                fpr, tpr, thrs = roc_curve(t2_binary_k, sc, pos_label=1)
                fnr = 1.0 - tpr
                eer_idx = np.argmin(np.abs(fpr - fnr))
                eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
                thr = thrs[eer_idx]
                preds = (sc >= thr).astype(int)
                fa = (preds[t2_binary_k == 0] == 0).mean() * 100
                ra = (preds[t2_binary_k == 1] == 1).mean() * 100
                print(f"    {sname}: AUC={auc:.4f} EER={eer*100:.1f}% fake={fa:.1f}% real={ra:.1f}% bal={(fa+ra)/2:.1f}%")

            # t-SNE
            plot_tsne_3label(v_bn, keep_lbl, f"{ckpt_name}_{vname}", out_dir)

        # Cleanup
        del pipeline, ccl_head, bn, sm, ref_fake_bn, ref_real_bn
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'=' * 80}")
    print(f"  Done! Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
