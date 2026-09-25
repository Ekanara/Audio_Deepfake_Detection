"""
Inference: 3 checkpoints × 3 test sets.
  - DIN-CTS: Mahalanobis distance (paper's method)
  - LL_fake_only, paperfaithful: Gaussian LL (baseline + Likelihood Ratio)
  - 5-class detection via softmax head + embedding t-SNE for Event/TUTASC19
  Also generates t-SNE plots per checkpoint/test set.
"""
import os, sys, gc
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

COLORS_2 = {"real": "#2ecc71", "fake": "#e74c3c"}

COLORS_6 = {
    "real": "#2ecc71", "fake_ata_01": "#e74c3c",
    "fake_tta_01": "#3498db", "fake_tta_02": "#f39c12",
    "fake_tta_03": "#9b59b6", "fake_unknown": "#7f8c8d",
}
NAMES_6 = {
    "real": "Real", "fake_ata_01": "Fake ATA",
    "fake_tta_01": "Fake TTA-1", "fake_tta_02": "Fake TTA-2",
    "fake_tta_03": "Fake TTA-3", "fake_unknown": "Unknown Fake",
}
ORDER_6 = ["real", "fake_ata_01", "fake_tta_01", "fake_tta_02", "fake_tta_03", "fake_unknown"]

SOFTMAX_MAP = {0: "fake_ata_01", 1: "fake_tta_01", 2: "fake_tta_02", 3: "fake_tta_03", 4: "real"}


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


def mahalanobis_distance(X, mu, sigma_inv):
    diff = X - mu
    left = diff @ sigma_inv
    return np.sqrt(np.sum(left * diff, axis=1))


def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thr[idx]


def eval_at_eer(score, labels):
    auc_pos = roc_auc_score(labels, score)
    auc_neg = roc_auc_score(labels, -score)
    if auc_neg > auc_pos:
        score = -score
        auc = auc_neg
    else:
        auc = auc_pos
    eer, thr = compute_eer(labels, score)
    preds = (score >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    rm = labels == 1
    fm = labels == 0
    real_acc = float((preds[rm] == 1).mean()) if rm.any() else 0
    fake_acc = float((preds[fm] == 0).mean()) if fm.any() else 0
    f1_r = f1_score(labels, preds, pos_label=1)
    f1_f = f1_score(1 - labels, 1 - preds, pos_label=1)
    return {
        "auc": auc, "eer": eer, "acc": acc,
        "real_acc": real_acc, "fake_acc": fake_acc,
        "f1_r": f1_r, "f1_f": f1_f,
    }


@torch.no_grad()
def collect_features(pipeline, json_file, device):
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
    lbl = np.array(all_lbl) if all_lbl else None
    return bn, sm, lbl, inv_label


# ── 2-class t-SNE ──────────────────────────────────────────────────────────

def plot_tsne(embs, labels_int, ckpt_name, ds_name, out_dir):
    max_per_class = 1000
    np.random.seed(42)
    idx_r = np.where(labels_int == 1)[0]
    idx_f = np.where(labels_int == 0)[0]
    if len(idx_r) > max_per_class:
        idx_r = np.random.choice(idx_r, max_per_class, replace=False)
    if len(idx_f) > max_per_class:
        idx_f = np.random.choice(idx_f, max_per_class, replace=False)
    idx = np.sort(np.concatenate([idx_r, idx_f]))
    sub_embs = embs[idx]
    sub_labels = labels_int[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    embs_2d = tsne.fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    for val, name, color in [(0, "Fake", COLORS_2["fake"]), (1, "Real", COLORS_2["real"])]:
        m = sub_labels == val
        if m.any():
            ax.scatter(embs_2d[m, 0], embs_2d[m, 1], c=color, label=name,
                       alpha=0.6, s=12, edgecolors="none")
    ax.legend(fontsize=11, markerscale=3, loc="best")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, ckpt_name, f"{ds_name}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE saved: {path}")


# ── 6-label t-SNE (5 known + unknown) ─────────────────────────────────────

def plot_tsne_6label(embs, gt_names, ckpt_name, ds_name, out_dir):
    unique = sorted(set(gt_names))
    max_per_class = 400
    np.random.seed(42)
    gt_arr = np.array(gt_names)
    idx_parts = []
    for lbl in unique:
        where = np.where(gt_arr == lbl)[0]
        if len(where) > max_per_class:
            where = np.random.choice(where, max_per_class, replace=False)
        idx_parts.append(where)
    idx = np.sort(np.concatenate(idx_parts))
    sub_embs = embs[idx]
    sub_names = gt_arr[idx]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                max_iter=1000, learning_rate="auto", init="pca")
    embs_2d = tsne.fit_transform(sub_embs)

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in ORDER_6:
        if lbl not in unique:
            continue
        m = sub_names == lbl
        display = NAMES_6.get(lbl, lbl.replace("_", " ").title())
        ax.scatter(embs_2d[m, 0], embs_2d[m, 1],
                   c=COLORS_6.get(lbl, "#888"), label=display,
                   alpha=0.6, s=12, edgecolors="none")
    ax.legend(fontsize=11, markerscale=3, loc="best")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, ckpt_name, f"{ds_name}_5class.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  t-SNE (6-label) saved: {path}")


# ── 5-class softmax head classification ────────────────────────────────────

def classify_softmax_head(sm, gt_names):
    """Use argmax of softmax head [N,5] to predict 5 training classes."""
    pred_idx = np.argmax(sm, axis=1)
    pred_names = np.array([SOFTMAX_MAP[i] for i in pred_idx])
    gt_arr = np.array(gt_names)

    print("    Softmax head classification:")
    print(f"    {'Ground Truth':16s} {'n':>6s}  {'→ Predicted distribution':s}")
    for gt_lbl in ORDER_6:
        m = gt_arr == gt_lbl
        if not m.any():
            continue
        n = int(m.sum())
        preds_for_class = pred_names[m]
        parts = []
        for p_lbl in ORDER_6[:-1]:  # exclude fake_unknown from predictions
            cnt = int((preds_for_class == p_lbl).sum())
            if cnt > 0:
                pct = cnt / n * 100
                parts.append(f"{NAMES_6[p_lbl]}={pct:.1f}%")
        print(f"    {NAMES_6[gt_lbl]:16s} n={n:5d}  {', '.join(parts)}")


def classify_embeddings(bn, gt_names, ref_gaussians):
    """Use embedding log-likelihood under per-class Gaussians to classify."""
    gt_arr = np.array(gt_names)
    class_names = list(ref_gaussians.keys())
    ll_matrix = np.column_stack([loglik(bn, ref_gaussians[c]) for c in class_names])
    pred_idx = np.argmax(ll_matrix, axis=1)
    pred_names = np.array([class_names[i] for i in pred_idx])

    print("    Embedding LL classification:")
    print(f"    {'Ground Truth':16s} {'n':>6s}  {'→ Predicted distribution':s}")
    for gt_lbl in ORDER_6:
        m = gt_arr == gt_lbl
        if not m.any():
            continue
        n = int(m.sum())
        preds_for_class = pred_names[m]
        parts = []
        for p_lbl in class_names:
            cnt = int((preds_for_class == p_lbl).sum())
            if cnt > 0:
                pct = cnt / n * 100
                parts.append(f"{NAMES_6.get(p_lbl, p_lbl)}={pct:.1f}%")
        print(f"    {NAMES_6[gt_lbl]:16s} n={n:5d}  {', '.join(parts)}")


def print_detection_breakdown(ds_name, score, gt_names):
    """Per-class binary detection rate (real vs fake) at EER threshold."""
    gt_arr = np.array(gt_names)
    binary = np.array([1 if n == "real" else 0 for n in gt_arr])
    auc_pos = roc_auc_score(binary, score)
    auc_neg = roc_auc_score(binary, -score)
    if auc_neg > auc_pos:
        score = -score
    eer, thr = compute_eer(binary, score)
    preds = (score >= thr).astype(int)
    print(f"\n  --- {ds_name} (per-class detection @ EER) ---")
    for lbl in ORDER_6:
        m = gt_arr == lbl
        if not m.any():
            continue
        n = int(m.sum())
        if lbl == "real":
            rate = float((preds[m] == 1).mean())
            tag = "recall"
        else:
            rate = float((preds[m] == 0).mean())
            tag = "caught"
        print(f"    {NAMES_6[lbl]:16s}  n={n:5d}  {tag}={rate*100:.2f}%")


def print_metrics(name, r):
    print(f"\n  --- {name} ---")
    print(f"  AUC={r['auc']:.4f}  EER={r['eer']*100:.2f}%  "
          f"Real%={r['real_acc']*100:.2f}%  Fake%={r['fake_acc']*100:.2f}%  "
          f"F1_R={r['f1_r']:.3f}  F1_F={r['f1_f']:.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = {
        "DIN-CTS_mahalanobis": {
            "path": "checkpoint/BEATs_CTS/stage2/checkpoint/best.pt",
            "method": "mahalanobis",
            "folder": "DIN-CTS",
        },
        "DIN-CTS_ll_fake_only": {
            "path": "checkpoint/BEATs_CTS/stage2/checkpoint/best.pt",
            "method": "ll_fake_only",
            "folder": "DIN-CTS_ll_fake_only",
        },
        "LL_fake_only": {
            "path": "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt",
            "method": "ll_fake_only",
            "folder": "LL_fake_only",
        },
        "paperfaithful": {
            "path": "checkpoint/Beats_journal/Beats_Event_2stage_paperfaithful_stage2_ceonly_8epoch/sample-04.ckpt",
            "method": "ll_fake_only",
            "folder": "paperfaithful",
        },
        "LL_realW8_retrained": {
            "path": "checkpoint/Beats_journal/retrain_cclW8_ceW3_ceRW8_LR5e-08_8ep/epoch-07_bal0.9550.ckpt",
            "method": "ll_fake_only",
            "folder": "LL_realW8_retrained",
        },
    }

    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
        "Event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
    }

    test_sets_5class = {
        "TUTASC19_test": "data/label/beats/TUTASC19_test_5class.json",
        "Event_test": "data/label/beats/Event_test_5class.json",
    }

    ref_real_json = "data/label/beats/Event_train_stage1_realonly.json"
    ref_fake_json = "data/label/beats/Event_train_stage1_fakeonly.json"
    ref_train_json = "data/label/beats/Event_train_stage1.json"
    tsne_dir = "inference_outputs/tsne"

    sep = "=" * 80

    for ckpt_name, ckpt_info in checkpoints.items():
        ckpt_path = ckpt_info["path"]
        method = ckpt_info["method"]
        folder = ckpt_info.get("folder", ckpt_name)
        print(f"\n{sep}")
        print(f"  {ckpt_name}  ({ckpt_path})  [{method}]")
        print(sep)

        pipeline, _, _ = load_pipeline_and_ccl(
            checkpoint_path=ckpt_path, device=device, mode="beats",
            num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

        # ── Binary inference (all 3 test sets) ─────────────────────────
        if method == "ll_fake_only":
            print("  Collecting reference features (fake + real)...")
            F_bn, _, _, _ = collect_features(pipeline, ref_fake_json, device)
            R_bn, _, _, _ = collect_features(pipeline, ref_real_json, device)
            gF = fit_gaussian(F_bn)
            gR = fit_gaussian(R_bn)
            print(f"  Fake ref: {F_bn.shape[0]} samples, dim={F_bn.shape[1]}")
            print(f"  Real ref: {R_bn.shape[0]} samples, dim={R_bn.shape[1]}")

            for ds_name, ds_path in test_sets.items():
                T_bn, T_sm, T_lbl, inv_label = collect_features(pipeline, ds_path, device)
                real_idx = {v: k for k, v in inv_label.items()}["real"]
                labels = (T_lbl == real_idx).astype(int)

                # Baseline: -LL(fake)
                score_baseline = -loglik(T_bn, gF)
                r_bl = eval_at_eer(score_baseline, labels)
                print_metrics(f"{ds_name} [baseline: -LL(fake)]", r_bl)

                # Likelihood Ratio: LL(real) - LL(fake)
                score_lr = loglik(T_bn, gR) - loglik(T_bn, gF)
                r_lr = eval_at_eer(score_lr, labels)
                print_metrics(f"{ds_name} [LR: LL(real)-LL(fake)]", r_lr)

                plot_tsne(T_bn, labels, folder, ds_name, tsne_dir)

        elif method == "mahalanobis":
            print("  Collecting real-only reference features...")
            R_bn, _, _, _ = collect_features(pipeline, ref_real_json, device)
            mu = R_bn.mean(axis=0)
            sigma = np.cov(R_bn, rowvar=False) + 1e-6 * np.eye(R_bn.shape[1])
            sigma_inv = np.linalg.inv(sigma)
            print(f"  Bonafide ref: {R_bn.shape[0]} samples, dim={R_bn.shape[1]}")

            for ds_name, ds_path in test_sets.items():
                T_bn, _, T_lbl, inv_label = collect_features(pipeline, ds_path, device)
                real_idx = {v: k for k, v in inv_label.items()}["real"]
                labels = (T_lbl == real_idx).astype(int)
                distances = mahalanobis_distance(T_bn, mu, sigma_inv)
                score = -distances
                r = eval_at_eer(score, labels)
                print_metrics(ds_name, r)
                plot_tsne(T_bn, labels, folder, ds_name, tsne_dir)

        # ── 5-class detection (Event_test, TUTASC19_test) ─────────────
        print(f"\n  === 5-class analysis ===")

        # Fit per-class Gaussians on training embeddings
        print("  Fitting per-class Gaussians on training data...")
        TR_bn, _, TR_lbl, tr_inv = collect_features(pipeline, ref_train_json, device)
        tr_names = np.array([tr_inv[l] for l in TR_lbl])
        ref_gaussians = {}
        for cls in ["real", "fake_ata_01", "fake_tta_01", "fake_tta_02", "fake_tta_03"]:
            m = tr_names == cls
            if m.any():
                ref_gaussians[cls] = fit_gaussian(TR_bn[m])
                print(f"    {cls}: {int(m.sum())} samples")
        del TR_bn, TR_lbl

        for ds_name, ds_path in test_sets_5class.items():
            T_bn, T_sm, T_lbl, inv_label = collect_features(pipeline, ds_path, device)
            gt_names = [inv_label[l] for l in T_lbl]

            # Binary detection breakdown per class
            real_idx = {v: k for k, v in inv_label.items()}["real"]
            binary = (T_lbl == real_idx).astype(int)
            if method == "ll_fake_only":
                score_bl = -loglik(T_bn, gF)
                score_lr = loglik(T_bn, gR) - loglik(T_bn, gF)
                print_detection_breakdown(f"{ds_name} [baseline]", score_bl.copy(), gt_names)
                print_detection_breakdown(f"{ds_name} [LR]", score_lr.copy(), gt_names)
            else:
                score_bl = -mahalanobis_distance(T_bn, mu, sigma_inv)
                print_detection_breakdown(ds_name, score_bl.copy(), gt_names)

            # Softmax head classification
            print(f"\n  --- {ds_name} (5-class classification) ---")
            classify_softmax_head(T_sm, gt_names)

            # Embedding LL classification
            classify_embeddings(T_bn, gt_names, ref_gaussians)

            # 6-label t-SNE
            plot_tsne_6label(T_bn, gt_names, folder, ds_name, tsne_dir)

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{sep}")
    print("Done!")


if __name__ == "__main__":
    main()
