"""
Sweep V2: Advanced inference methods to improve real recall for LL_fake_only.
Focuses on: PCA reduction, cosine similarity, k-NN, Mahalanobis, adaptive fusion.
"""
import os, sys
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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


def mahal_dist(X, g):
    mu, cf, lower, _ = g
    diff = X.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    return np.sqrt(np.einsum("ij,ij->i", diff, solved))


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


def cosine_sim(X, center):
    """Cosine similarity between each row of X and center vector."""
    X = X.astype(np.float64)
    center = center.astype(np.float64)
    norms_x = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    norm_c = np.linalg.norm(center) + 1e-10
    return (X @ center) / (norms_x.ravel() * norm_c)


def knn_score(X_test, X_ref, k=5):
    """Average distance to k nearest neighbors in ref set.
    Uses chunked computation to avoid OOM on large sets."""
    from scipy.spatial.distance import cdist
    chunk = 1000
    scores = []
    for i in range(0, len(X_test), chunk):
        D = cdist(X_test[i:i+chunk], X_ref, metric="euclidean")
        topk = np.partition(D, k, axis=1)[:, :k]
        scores.append(topk.mean(axis=1))
    return np.concatenate(scores)


def knn_score_cosine(X_test, X_ref, k=5):
    """Average cosine distance to k nearest neighbors."""
    from scipy.spatial.distance import cdist
    chunk = 1000
    scores = []
    for i in range(0, len(X_test), chunk):
        D = cdist(X_test[i:i+chunk], X_ref, metric="cosine")
        topk = np.partition(D, k, axis=1)[:, :k]
        scores.append(topk.mean(axis=1))
    return np.concatenate(scores)


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
    print(f"  {name:55s}  AUC={r['auc']:.4f}  EER={r['eer']*100:5.2f}%  "
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

    print(f"  Fake ref: {F_bn.shape[0]} samples, dim={F_bn.shape[1]}")
    print(f"  Real ref: {R_bn.shape[0]} samples, dim={R_bn.shape[1]}")

    # ── Precompute reference stats ────────────────────────────────────
    print("Fitting reference distributions...")
    gF = fit_gaussian(F_bn)
    gR = fit_gaussian(R_bn)

    # Centroids
    fake_center = F_bn.mean(axis=0)
    real_center = R_bn.mean(axis=0)

    # PCA at various dimensions
    pca_dims = [32, 64, 128, 256]
    pca_models = {}
    gF_pca = {}
    gR_pca = {}
    for d in pca_dims:
        pca = PCA(n_components=d, random_state=42)
        all_train = np.vstack([F_bn, R_bn])
        pca.fit(all_train)
        pca_models[d] = pca
        gF_pca[d] = fit_gaussian(pca.transform(F_bn))
        gR_pca[d] = fit_gaussian(pca.transform(R_bn))
        print(f"  PCA d={d}: variance explained = {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # StandardScaler (Z-score normalization)
    scaler = StandardScaler()
    all_train = np.vstack([F_bn, R_bn])
    scaler.fit(all_train)
    F_bn_z = scaler.transform(F_bn)
    R_bn_z = scaler.transform(R_bn)
    gF_z = fit_gaussian(F_bn_z)
    gR_z = fit_gaussian(R_bn_z)

    # ── Evaluate each test set ────────────────────────────────────────
    sep = "=" * 130
    for ds_name, ds_path in test_sets.items():
        print(f"\n{sep}")
        print(f"  {ds_name}")
        print(sep)

        T_bn, T_sm, T_ct, T_lbl, inv_label = collect_features(pipeline, ds_path, device)
        real_idx = {v: k for k, v in inv_label.items()}["real"]
        labels = (T_lbl == real_idx).astype(int)
        p_real = T_sm[:, 4]  # softmax real probability

        results = {}

        # ── Baseline ─────────────────────────────────────────────────
        s = -loglik(T_bn, gF)
        results["A1. baseline: -LL(fake)"] = eval_score(s, labels)

        # ── Best from V1: LR ─────────────────────────────────────────
        s = loglik(T_bn, gR) - loglik(T_bn, gF)
        results["A2. LR: LL(real)-LL(fake)"] = eval_score(s, labels)

        # ── Mahalanobis to fake center ────────────────────────────────
        s = mahal_dist(T_bn, gF)
        results["B1. Mahalanobis dist to fake"] = eval_score(s, labels)

        # ── Mahalanobis ratio ─────────────────────────────────────────
        s = mahal_dist(T_bn, gF) - mahal_dist(T_bn, gR)
        results["B2. Mahal(fake) - Mahal(real)"] = eval_score(s, labels)

        # ── Cosine similarity ─────────────────────────────────────────
        cos_r = cosine_sim(T_bn, real_center)
        cos_f = cosine_sim(T_bn, fake_center)
        results["C1. cosine sim to real center"] = eval_score(cos_r, labels)
        results["C2. cosine: sim(real)-sim(fake)"] = eval_score(cos_r - cos_f, labels)

        # ── Cosine + LL fusion ────────────────────────────────────────
        s = rank_pct(-loglik(T_bn, gF)) + rank_pct(cos_r - cos_f)
        results["C3. rank: -LL(fake) + cos_diff"] = eval_score(s, labels)

        s = rank_pct(-loglik(T_bn, gF)) + rank_pct(cos_r)
        results["C4. rank: -LL(fake) + cos(real)"] = eval_score(s, labels)

        # ── PCA methods ──────────────────────────────────────────────
        for d in pca_dims:
            T_pca = pca_models[d].transform(T_bn)

            s = -loglik(T_pca, gF_pca[d])
            results[f"D1. PCA{d}: -LL(fake)"] = eval_score(s, labels)

            s = loglik(T_pca, gR_pca[d]) - loglik(T_pca, gF_pca[d])
            results[f"D2. PCA{d}: LR"] = eval_score(s, labels)

        # ── Z-score normalized ────────────────────────────────────────
        T_bn_z = scaler.transform(T_bn)
        s = -loglik(T_bn_z, gF_z)
        results["E1. Z-norm: -LL(fake)"] = eval_score(s, labels)
        s = loglik(T_bn_z, gR_z) - loglik(T_bn_z, gF_z)
        results["E2. Z-norm: LR"] = eval_score(s, labels)

        # ── k-NN methods ─────────────────────────────────────────────
        for k in [3, 10, 30]:
            knn_f = knn_score(T_bn, F_bn, k=k)
            results[f"F1. kNN(fake) k={k}"] = eval_score(knn_f, labels)

            knn_r = knn_score(T_bn, R_bn, k=k)
            results[f"F2. kNN diff k={k}: kNN(fake)-kNN(real)"] = eval_score(knn_f - knn_r, labels)

        # ── k-NN cosine ──────────────────────────────────────────────
        for k in [3, 10]:
            knn_f_cos = knn_score_cosine(T_bn, F_bn, k=k)
            results[f"G1. kNN_cos(fake) k={k}"] = eval_score(knn_f_cos, labels)
            knn_r_cos = knn_score_cosine(T_bn, R_bn, k=k)
            results[f"G2. kNN_cos diff k={k}"] = eval_score(knn_f_cos - knn_r_cos, labels)

        # ── Softmax entropy ──────────────────────────────────────────
        sm_probs = T_sm / T_sm.sum(axis=1, keepdims=True).clip(1e-10)
        entropy = -np.sum(sm_probs * np.log(sm_probs + 1e-10), axis=1)
        results["H1. softmax entropy (low=confident)"] = eval_score(-entropy, labels)

        # ── Max fake softmax prob ─────────────────────────────────────
        max_fake_prob = T_sm[:, :4].max(axis=1)  # max of 4 fake classes
        results["H2. -max_fake_softmax"] = eval_score(-max_fake_prob, labels)

        # ── p(real) / max(p(fake)) ratio ──────────────────────────────
        s = p_real / (max_fake_prob + 1e-10)
        results["H3. p(real)/max(p(fake))"] = eval_score(s, labels)

        # ── Multi-signal fusions ──────────────────────────────────────
        ll_fake = -loglik(T_bn, gF)
        lr = loglik(T_bn, gR) - loglik(T_bn, gF)

        # Triple fusion: LL + cosine + softmax
        s = rank_pct(ll_fake) + rank_pct(cos_r) + rank_pct(p_real)
        results["I1. rank3: -LL(fake)+cos(real)+p(real)"] = eval_score(s, labels)

        s = rank_pct(lr) + rank_pct(cos_r) + rank_pct(p_real)
        results["I2. rank3: LR+cos(real)+p(real)"] = eval_score(s, labels)

        # LL + kNN fusion
        knn_f10 = knn_score(T_bn, F_bn, k=10)
        s = rank_pct(ll_fake) + rank_pct(knn_f10)
        results["I3. rank: -LL(fake)+kNN(fake,k=10)"] = eval_score(s, labels)

        s = rank_pct(knn_f10) + rank_pct(p_real)
        results["I4. rank: kNN(fake,k=10)+p(real)"] = eval_score(s, labels)

        # ── Contrastive head if available ─────────────────────────────
        if T_ct is not None:
            gF_ct = fit_gaussian(F_ct)
            gR_ct = fit_gaussian(R_ct)

            # LR on contrastive + LR on bonafide, weighted
            lr_bn = loglik(T_bn, gR) - loglik(T_bn, gF)
            lr_ct = loglik(T_ct, gR_ct) - loglik(T_ct, gF_ct)
            for w in [0.3, 0.5, 0.7]:
                s = rank_pct(lr_bn) * (1-w) + rank_pct(lr_ct) * w
                results[f"J1. weighted: LR_bn*{1-w:.1f}+LR_ct*{w:.1f}"] = eval_score(s, labels)

            # 4-signal fusion
            s = rank_pct(lr_bn) + rank_pct(lr_ct) + rank_pct(cos_r) + rank_pct(p_real)
            results["J2. rank4: LR_bn+LR_ct+cos+p(real)"] = eval_score(s, labels)

        # ── Print sorted by real_acc ──────────────────────────────────
        print(f"\n  {'Method':55s}  {'AUC':>6s}  {'EER':>7s}  {'Real%':>7s}  {'Fake%':>7s}  {'F1_R':>5s}  {'F1_F':>5s}")
        print("  " + "-" * 120)
        for name, r in sorted(results.items(), key=lambda x: -x[1]["real_acc"]):
            print_result(name, r)

    print("\nDone!")


if __name__ == "__main__":
    main()
