"""
Sweep V3: Score normalization and calibration methods.
T-norm, Z-norm, AS-norm, Platt scaling, threshold tuning.
Also: adaptive LR that downweights LL(real) when it's uncertain.
"""
import os, sys
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.linear_model import LogisticRegression
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


def eval_at_threshold(score, labels, thr):
    """Evaluate with a fixed threshold instead of EER."""
    preds = (score >= thr).astype(int)
    rm = labels == 1
    fm = labels == 0
    real_acc = float((preds[rm] == 1).mean()) if rm.any() else 0
    fake_acc = float((preds[fm] == 0).mean()) if fm.any() else 0
    f1_r = f1_score(labels, preds, pos_label=1, zero_division=0)
    f1_f = f1_score(1 - labels, 1 - preds, pos_label=1, zero_division=0)
    auc = roc_auc_score(labels, score)
    eer, _ = compute_eer(labels, score)
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


def z_norm(scores, cohort_scores):
    """Z-norm: normalize by cohort statistics."""
    mu = cohort_scores.mean()
    std = cohort_scores.std() + 1e-10
    return (scores - mu) / std


def t_norm(scores, cohort_scores_per_sample):
    """T-norm: per-sample normalization using cohort."""
    mu = cohort_scores_per_sample.mean(axis=1)
    std = cohort_scores_per_sample.std(axis=1) + 1e-10
    return (scores - mu) / std


def as_norm(scores, z_scores, t_scores):
    """Adaptive S-norm: average of Z-norm and T-norm."""
    return 0.5 * (z_scores + t_scores)


def print_result(name, r):
    print(f"  {name:60s}  AUC={r['auc']:.4f}  EER={r['eer']*100:5.2f}%  "
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
    TR_bn, TR_sm, _, TR_lbl, tr_inv = collect_features(pipeline, ref_train_json, device)
    tr_names = np.array([tr_inv[l] for l in TR_lbl])

    print(f"  Fake ref: {F_bn.shape[0]} samples")
    print(f"  Real ref: {R_bn.shape[0]} samples")

    # ── Fit reference distributions ───────────────────────────────────
    print("Fitting Gaussians...")
    gF = fit_gaussian(F_bn)
    gR = fit_gaussian(R_bn)

    # ── Compute training set scores for calibration ───────────────────
    print("Computing training set scores for calibration...")
    tr_ll_fake = -loglik(TR_bn, gF)
    tr_lr = loglik(TR_bn, gR) - loglik(TR_bn, gF)
    tr_p_real = TR_sm[:, 4]
    tr_labels = (tr_names == "real").astype(int)

    # Fit Platt scaling on training set
    print("Fitting Platt calibrators...")
    platt_ll = LogisticRegression(C=1.0).fit(tr_ll_fake.reshape(-1, 1), tr_labels)
    platt_lr = LogisticRegression(C=1.0).fit(tr_lr.reshape(-1, 1), tr_labels)

    # Multi-feature Platt: [ll_fake, p_real]
    tr_feats_2 = np.column_stack([tr_ll_fake, tr_p_real])
    platt_2f = LogisticRegression(C=1.0).fit(tr_feats_2, tr_labels)

    # Multi-feature Platt: [ll_fake, lr, p_real]
    tr_feats_3 = np.column_stack([tr_ll_fake, tr_lr, tr_p_real])
    platt_3f = LogisticRegression(C=1.0).fit(tr_feats_3, tr_labels)

    # Z-norm cohort: compute LL scores on fake reference set
    cohort_ll_fake = -loglik(F_bn, gF)  # fake samples' -LL(fake) scores
    cohort_lr = loglik(F_bn, gR) - loglik(F_bn, gF)
    cohort_ll_fake_real = -loglik(R_bn, gF)  # real samples' -LL(fake) scores
    cohort_lr_real = loglik(R_bn, gR) - loglik(R_bn, gF)

    # Find EER thresholds on training set for calibration
    _, thr_ll_train = compute_eer(tr_labels, tr_ll_fake)
    _, thr_lr_train = compute_eer(tr_labels, tr_lr)

    # ── Adaptive LR weight based on softmax confidence ────────────────
    # Idea: when softmax is confident real, trust LR more; when uncertain, fall back to -LL(fake)
    print("Precomputing adaptive weights...")

    # ── Evaluate each test set ────────────────────────────────────────
    sep = "=" * 140
    for ds_name, ds_path in test_sets.items():
        print(f"\n{sep}")
        print(f"  {ds_name}")
        print(sep)

        T_bn, T_sm, T_ct, T_lbl, inv_label = collect_features(pipeline, ds_path, device)
        real_idx = {v: k for k, v in inv_label.items()}["real"]
        labels = (T_lbl == real_idx).astype(int)
        p_real = T_sm[:, 4]

        results = {}

        # Raw scores
        ll_fake = -loglik(T_bn, gF)
        lr = loglik(T_bn, gR) - loglik(T_bn, gF)

        # ── Baselines ─────────────────────────────────────────────────
        results["A1. baseline: -LL(fake)"] = eval_score(ll_fake, labels)
        results["A2. LR"] = eval_score(lr, labels)

        # ── Platt-calibrated scores ───────────────────────────────────
        platt_s = platt_ll.predict_proba(ll_fake.reshape(-1, 1))[:, 1]
        results["B1. Platt(-LL(fake))"] = eval_score(platt_s, labels)

        platt_s = platt_lr.predict_proba(lr.reshape(-1, 1))[:, 1]
        results["B2. Platt(LR)"] = eval_score(platt_s, labels)

        # Multi-feature Platt
        test_feats_2 = np.column_stack([ll_fake, p_real])
        platt_s = platt_2f.predict_proba(test_feats_2)[:, 1]
        results["B3. Platt(LL+p_real)"] = eval_score(platt_s, labels)

        test_feats_3 = np.column_stack([ll_fake, lr, p_real])
        platt_s = platt_3f.predict_proba(test_feats_3)[:, 1]
        results["B4. Platt(LL+LR+p_real)"] = eval_score(platt_s, labels)

        # ── Z-norm: normalize test scores by fake cohort stats ────────
        s = z_norm(ll_fake, cohort_ll_fake)
        results["C1. Z-norm(fake cohort) -LL(fake)"] = eval_score(s, labels)

        s = z_norm(lr, cohort_lr)
        results["C2. Z-norm(fake cohort) LR"] = eval_score(s, labels)

        # Z-norm by real cohort
        s = z_norm(ll_fake, cohort_ll_fake_real)
        results["C3. Z-norm(real cohort) -LL(fake)"] = eval_score(s, labels)

        s = z_norm(lr, cohort_lr_real)
        results["C4. Z-norm(real cohort) LR"] = eval_score(s, labels)

        # ── Adaptive LR: weight LL(real) by softmax confidence ────────
        # When softmax says "real" with high confidence, trust LL(real) more
        ll_real_raw = loglik(T_bn, gR)
        ll_fake_raw = loglik(T_bn, gF)

        for beta in [0.1, 0.3, 0.5, 0.7]:
            # weight = softmax_confidence ^ beta
            w = np.power(p_real.clip(0.01, 0.99), beta)
            s = w * ll_real_raw - ll_fake_raw
            results[f"D1. adaptive LR: p_real^{beta} * LL(real) - LL(fake)"] = eval_score(s, labels)

        # ── Soft gating: use LL(fake) when softmax is uncertain ───────
        for gate_thr in [0.3, 0.5, 0.7]:
            is_conf_real = (p_real > gate_thr).astype(float)
            # Confident real → use LR, uncertain → use -LL(fake)
            s = is_conf_real * lr + (1 - is_conf_real) * ll_fake
            results[f"E1. gate(p>{gate_thr}): LR else -LL(fake)"] = eval_score(s, labels)

        # ── Smooth gating ─────────────────────────────────────────────
        for temp in [1.0, 3.0, 5.0]:
            gate = 1.0 / (1.0 + np.exp(-temp * (p_real - 0.5)))
            s = gate * rank_pct(lr) + (1 - gate) * rank_pct(ll_fake)
            results[f"E2. smooth gate temp={temp}: LR↔-LL(fake)"] = eval_score(s, labels)

        # ── Training-calibrated thresholds ────────────────────────────
        results["F1. -LL(fake) @ train EER thr"] = eval_at_threshold(ll_fake, labels, thr_ll_train)
        results["F2. LR @ train EER thr"] = eval_at_threshold(lr, labels, thr_lr_train)

        # ── Threshold sweep for real-biased operating points ──────────
        fpr, tpr, thresholds = roc_curve(labels, ll_fake, pos_label=1)
        for target_real in [0.90, 0.92, 0.95]:
            idx = np.argmin(np.abs(tpr - target_real))
            if idx < len(thresholds):
                thr = thresholds[idx]
                r = eval_at_threshold(ll_fake, labels, thr)
                results[f"G1. -LL(fake) @ Real≈{target_real*100:.0f}%"] = r

        fpr, tpr, thresholds = roc_curve(labels, lr, pos_label=1)
        for target_real in [0.90, 0.92, 0.95]:
            idx = np.argmin(np.abs(tpr - target_real))
            if idx < len(thresholds):
                thr = thresholds[idx]
                r = eval_at_threshold(lr, labels, thr)
                results[f"G2. LR @ Real≈{target_real*100:.0f}%"] = r

        # ── Logistic ensemble (train on training set) ─────────────────
        # Use training set scores to learn optimal combination weights
        tr_combo = np.column_stack([
            rank_pct(tr_ll_fake),
            rank_pct(tr_lr),
            rank_pct(tr_p_real),
        ])
        for C in [0.01, 0.1, 1.0, 10.0]:
            lr_model = LogisticRegression(C=C).fit(tr_combo, tr_labels)
            test_combo = np.column_stack([
                rank_pct(ll_fake),
                rank_pct(lr),
                rank_pct(p_real),
            ])
            s = lr_model.predict_proba(test_combo)[:, 1]
            results[f"H1. LogReg(LL+LR+p) C={C}"] = eval_score(s, labels)

        # ── Print sorted by real_acc ──────────────────────────────────
        print(f"\n  {'Method':60s}  {'AUC':>6s}  {'EER':>7s}  {'Real%':>7s}  {'Fake%':>7s}  {'F1_R':>5s}  {'F1_F':>5s}")
        print("  " + "-" * 125)
        for name, r in sorted(results.items(), key=lambda x: -x[1]["real_acc"]):
            print_result(name, r)

    print("\nDone!")


if __name__ == "__main__":
    main()
