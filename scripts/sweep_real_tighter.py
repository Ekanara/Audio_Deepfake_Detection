"""
Sweep inference methods to improve real recall for LL_fake_only checkpoint.
Tries: likelihood ratio, LedoitWolf, softmax ensemble, rank fusion, etc.
"""
import os, sys, itertools
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from torch.utils.data import DataLoader
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl


def fit_gaussian(X, eps=1e-6):
    X = X.astype(np.float64)
    mu = X.mean(0)
    cov = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    cf, lower = cho_factor(cov)
    _, logdet = np.linalg.slogdet(cov)
    return mu, cf, lower, logdet


def fit_gaussian_lw(X):
    X = X.astype(np.float64)
    mu = X.mean(0)
    lw = LedoitWolf().fit(X)
    cov = lw.covariance_
    cf, lower = cho_factor(cov)
    _, logdet = np.linalg.slogdet(cov)
    return mu, cf, lower, logdet


def loglik(X, g):
    mu, cf, lower, logdet = g
    diff = X.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    return -0.5 * (np.einsum("ij,ij->i", diff, solved) + logdet)


def rank_pct(x):
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def compute_eer(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return (fpr[idx] + fnr[idx]) / 2.0, thr[idx]


def eval_score(score, labels):
    auc_pos = roc_auc_score(labels, score)
    auc_neg = roc_auc_score(labels, -score)
    if auc_neg > auc_pos:
        score = -score
        auc = auc_neg
    else:
        auc = auc_pos
    eer, thr = compute_eer(labels, score)
    preds = (score >= thr).astype(int)
    rm = labels == 1
    fm = labels == 0
    real_acc = float((preds[rm] == 1).mean()) if rm.any() else 0
    fake_acc = float((preds[fm] == 0).mean()) if fm.any() else 0
    f1_r = f1_score(labels, preds, pos_label=1)
    f1_f = f1_score(1 - labels, 1 - preds, pos_label=1)
    return {
        "auc": auc, "eer": eer,
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
    all_bn, all_sm, all_ct, all_lbl = [], [], [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        all_sm.append(outputs[1].float().cpu().numpy())
        if len(outputs) > 2:
            all_ct.append(outputs[2].float().cpu().numpy())
        if "label" in batch:
            all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    bn = np.concatenate(all_bn)
    sm = np.concatenate(all_sm)
    ct = np.concatenate(all_ct) if all_ct else None
    lbl = np.array(all_lbl) if all_lbl else None
    return bn, sm, ct, lbl, inv_label


def print_result(name, r):
    print(f"  {name:45s}  AUC={r['auc']:.4f}  EER={r['eer']*100:5.2f}%  "
          f"Real%={r['real_acc']*100:5.2f}%  Fake%={r['fake_acc']*100:5.2f}%  "
          f"F1_R={r['f1_r']:.3f}  F1_F={r['f1_f']:.3f}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"
    ref_fake_json = "data/label/beats/Event_train_stage1_fakeonly.json"
    ref_real_json = "data/label/beats/Event_train_stage1_realonly.json"
    ref_train_json = "data/label/beats/Event_train_stage1.json"

    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "Event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
    }

    print("Loading model...")
    pipeline, _, _ = load_pipeline_and_ccl(
        checkpoint_path=ckpt_path, device=device, mode="beats",
        num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

    # ── Collect reference features ────────────────────────────────────
    print("Collecting reference features...")
    F_bn, F_sm, F_ct, _, _ = collect_features(pipeline, ref_fake_json, device)
    R_bn, R_sm, R_ct, _, _ = collect_features(pipeline, ref_real_json, device)
    TR_bn, _, _, TR_lbl, tr_inv = collect_features(pipeline, ref_train_json, device)
    tr_names = np.array([tr_inv[l] for l in TR_lbl])

    print(f"  Fake ref: {F_bn.shape[0]} samples")
    print(f"  Real ref: {R_bn.shape[0]} samples")

    # ── Fit reference distributions ───────────────────────────────────
    print("Fitting Gaussians...")
    gF = fit_gaussian(F_bn)
    gR = fit_gaussian(R_bn)
    gF_lw = fit_gaussian_lw(F_bn)
    gR_lw = fit_gaussian_lw(R_bn)

    # Per-class Gaussians
    class_gaussians = {}
    class_gaussians_lw = {}
    for cls in ["real", "fake_ata_01", "fake_tta_01", "fake_tta_02", "fake_tta_03"]:
        m = tr_names == cls
        if m.any():
            class_gaussians[cls] = fit_gaussian(TR_bn[m])
            class_gaussians_lw[cls] = fit_gaussian_lw(TR_bn[m])

    # ── Evaluate each test set ────────────────────────────────────────
    sep = "=" * 120
    for ds_name, ds_path in test_sets.items():
        print(f"\n{sep}")
        print(f"  {ds_name}")
        print(sep)

        T_bn, T_sm, T_ct, T_lbl, inv_label = collect_features(pipeline, ds_path, device)
        real_idx = {v: k for k, v in inv_label.items()}["real"]
        labels = (T_lbl == real_idx).astype(int)
        p_real = T_sm[:, 4]  # softmax head real probability

        results = {}

        # ── Method 1: Baseline (current) ──────────────────────────────
        s = -loglik(T_bn, gF)
        results["1. baseline: -LL(fake)"] = eval_score(s, labels)

        # ── Method 2: LedoitWolf fake ─────────────────────────────────
        s = -loglik(T_bn, gF_lw)
        results["2. -LL(fake) LedoitWolf"] = eval_score(s, labels)

        # ── Method 3: LL(real) only ───────────────────────────────────
        s = loglik(T_bn, gR)
        results["3. LL(real)"] = eval_score(s, labels)

        # ── Method 4: LL(real) LedoitWolf ─────────────────────────────
        s = loglik(T_bn, gR_lw)
        results["4. LL(real) LedoitWolf"] = eval_score(s, labels)

        # ── Method 5: Likelihood ratio ────────────────────────────────
        s = loglik(T_bn, gR) - loglik(T_bn, gF)
        results["5. LR: LL(real)-LL(fake)"] = eval_score(s, labels)

        # ── Method 6: LR LedoitWolf ──────────────────────────────────
        s = loglik(T_bn, gR_lw) - loglik(T_bn, gF_lw)
        results["6. LR LedoitWolf"] = eval_score(s, labels)

        # ── Method 7: Softmax p(real) only ────────────────────────────
        results["7. softmax p(real)"] = eval_score(p_real, labels)

        # ── Method 8: Rank fusion baseline + softmax ──────────────────
        s = rank_pct(-loglik(T_bn, gF)) + rank_pct(p_real)
        results["8. rank: -LL(fake) + p(real)"] = eval_score(s, labels)

        # ── Method 9: Rank fusion LR + softmax ────────────────────────
        lr = loglik(T_bn, gR) - loglik(T_bn, gF)
        s = rank_pct(lr) + rank_pct(p_real)
        results["9. rank: LR + p(real)"] = eval_score(s, labels)

        # ── Method 10: Weighted combos ────────────────────────────────
        ll_fake = -loglik(T_bn, gF)
        for alpha in [0.3, 0.5, 0.7]:
            s = rank_pct(ll_fake) * (1 - alpha) + rank_pct(p_real) * alpha
            results[f"10. weighted rank a={alpha}: -LL(fake)+p(real)"] = eval_score(s, labels)

        # ── Method 11: LR + softmax weighted ──────────────────────────
        lr = loglik(T_bn, gR) - loglik(T_bn, gF)
        for alpha in [0.3, 0.5, 0.7]:
            s = rank_pct(lr) * (1 - alpha) + rank_pct(p_real) * alpha
            results[f"11. weighted rank a={alpha}: LR+p(real)"] = eval_score(s, labels)

        # ── Method 12: Max class LL ───────────────────────────────────
        fake_classes = ["fake_ata_01", "fake_tta_01", "fake_tta_02", "fake_tta_03"]
        ll_per_fake = np.column_stack([loglik(T_bn, class_gaussians[c]) for c in fake_classes])
        max_fake_ll = ll_per_fake.max(axis=1)
        s = -max_fake_ll
        results["12. -max_LL(per_fake_class)"] = eval_score(s, labels)

        # ── Method 13: LR with max fake class ─────────────────────────
        s = loglik(T_bn, gR) - max_fake_ll
        results["13. LL(real) - max_LL(fake_class)"] = eval_score(s, labels)

        # ── Method 14: LR max fake class LedoitWolf ───────────────────
        ll_per_fake_lw = np.column_stack([loglik(T_bn, class_gaussians_lw[c]) for c in fake_classes])
        max_fake_ll_lw = ll_per_fake_lw.max(axis=1)
        s = loglik(T_bn, gR_lw) - max_fake_ll_lw
        results["14. LR max_fake_class LedoitWolf"] = eval_score(s, labels)

        # ── Method 15: Contrastive head if available ──────────────────
        if T_ct is not None:
            gF_ct = fit_gaussian(F_ct)
            gR_ct = fit_gaussian(R_ct)
            s = -loglik(T_ct, gF_ct)
            results["15. contrastive: -LL(fake)"] = eval_score(s, labels)
            s = loglik(T_ct, gR_ct) - loglik(T_ct, gF_ct)
            results["16. contrastive: LR"] = eval_score(s, labels)

            # Rank fusion: bonafide LR + contrastive LR
            lr_bn = loglik(T_bn, gR) - loglik(T_bn, gF)
            lr_ct = loglik(T_ct, gR_ct) - loglik(T_ct, gF_ct)
            s = rank_pct(lr_bn) + rank_pct(lr_ct)
            results["17. rank: bonafide_LR + contrastive_LR"] = eval_score(s, labels)

            s = rank_pct(lr_bn) + rank_pct(lr_ct) + rank_pct(p_real)
            results["18. rank: bn_LR + ct_LR + p(real)"] = eval_score(s, labels)

        # ── Print sorted by real_acc ──────────────────────────────────
        print(f"\n  {'Method':45s}  {'AUC':>6s}  {'EER':>7s}  {'Real%':>7s}  {'Fake%':>7s}  {'F1_R':>5s}  {'F1_F':>5s}")
        print("  " + "-" * 110)
        for name, r in sorted(results.items(), key=lambda x: -x[1]["real_acc"]):
            print_result(name, r)

    print("\nDone!")


if __name__ == "__main__":
    main()
