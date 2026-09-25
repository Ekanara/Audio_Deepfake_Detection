"""
Try different scoring methods on a single checkpoint.
Methods:
  1. baseline (-LL_fake)        — Gaussian on fake ref, score = -loglik
  2. LR (LL_real - LL_fake)     — likelihood ratio
  3. cosine_real                 — cosine similarity to real centroid
  4. cosine_fake                 — negative cosine sim to fake centroid
  5. mahal_real                  — negative Mahalanobis distance to real
  6. mahal_fake                  — Mahalanobis distance to fake (higher = more real)
  7. ccl_centers                 — distance to CCL learned centers
  8. softmax_p_real              — p(real) from softmax head
  9. knn_real                    — mean distance to k-nearest real refs
  10. fusion                     — weighted combo of best methods
"""
import os, sys, json, gc
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from beats.model_beat import model_beat
from utils.training_utils import CenterContrastiveLoss

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


def mahalanobis(X, g):
    """Mahalanobis distance (not squared) to Gaussian center."""
    mu, cf, lower, logdet = g
    diff = X.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    return np.sqrt(np.einsum("ij,ij->i", diff, solved))


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
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv_label = {v: k for k, v in ds.label.items()}
    from torch.utils.data import DataLoader
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
    lbl_str = np.array([inv_label[i] for i in lbl_int])
    return bn, sm, lbl_int, lbl_str


def eval_score(name, scores, binary_labels):
    """Print AUC, EER, per-class acc for a scoring method."""
    if len(np.unique(binary_labels)) < 2:
        return None
    auc = roc_auc_score(binary_labels, scores)
    fpr, tpr, thrs = roc_curve(binary_labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    thr = thrs[eer_idx]
    preds = (scores >= thr).astype(int)
    fake_acc = (preds[binary_labels == 0] == 0).mean() * 100
    real_acc = (preds[binary_labels == 1] == 1).mean() * 100
    bal = (fake_acc + real_acc) / 2
    print(f"  {name:30s}  AUC={auc:.4f}  EER={eer*100:5.1f}%  fake={fake_acc:5.1f}%  real={real_acc:5.1f}%  bal={bal:5.1f}%")
    return {"auc": auc, "eer": eer, "bal": bal, "scores": scores, "thr": thr}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoint/Beats_journal/retrain_cclW8_ceW3_ceRW8_LR5e-08_8ep/epoch-06_bal0.9653.ckpt"
    track2_json = "data/label/beats/test_track2.json"
    ref_fake_json = "data/label/beats/Event_train_stage1_fakeonly.json"
    ref_real_json = "data/label/beats/Event_train_stage1_realonly.json"

    print(f"Checkpoint: {ckpt_path}")
    print(f"Test set: {track2_json}")
    print("=" * 100)

    # Load model
    pipeline, ccl_head = load_checkpoint(ckpt_path, device)

    # Collect features
    print("\n  Collecting test features...")
    test_bn, test_sm, test_lint, test_lstr = collect_features(pipeline, track2_json, device)
    print(f"  Test: {test_bn.shape[0]} samples, {test_bn.shape[1]}-dim")

    print("  Collecting fake reference features...")
    ref_fake_bn, ref_fake_sm, _, _ = collect_features(pipeline, ref_fake_json, device)
    print(f"  Fake ref: {ref_fake_bn.shape[0]} samples")

    print("  Collecting real reference features...")
    ref_real_bn, ref_real_sm, _, _ = collect_features(pipeline, ref_real_json, device)
    print(f"  Real ref: {ref_real_bn.shape[0]} samples")

    # Binary labels: 1=real, 0=fake
    binary = (test_lstr == "real").astype(int)
    print(f"\n  Binary: {binary.sum()} real, {(1-binary).sum()} fake")

    # Fit Gaussians
    gF = fit_gaussian(ref_fake_bn)
    gR = fit_gaussian(ref_real_bn)

    print(f"\n{'='*100}")
    print(f"  {'Method':30s}  {'AUC':>6s}  {'EER':>6s}  {'Fake%':>6s}  {'Real%':>6s}  {'Bal%':>6s}")
    print(f"{'='*100}")

    results = {}

    # 1. baseline (-LL_fake)
    ll_fake = loglik(test_bn, gF)
    r = eval_score("1. baseline (-LL_fake)", -ll_fake, binary)
    if r: results["baseline"] = r

    # 2. LR (LL_real - LL_fake)
    ll_real = loglik(test_bn, gR)
    r = eval_score("2. LR (LL_real - LL_fake)", ll_real - ll_fake, binary)
    if r: results["LR"] = r

    # 3. LL_real only
    r = eval_score("3. LL_real only", ll_real, binary)
    if r: results["LL_real"] = r

    # 4. Cosine similarity to real centroid
    real_centroid = ref_real_bn.mean(axis=0)
    cos_real = test_bn @ real_centroid / (np.linalg.norm(test_bn, axis=1) * np.linalg.norm(real_centroid) + 1e-8)
    r = eval_score("4. cosine_to_real_centroid", cos_real, binary)
    if r: results["cos_real"] = r

    # 5. Cosine similarity to fake centroid (negated)
    fake_centroid = ref_fake_bn.mean(axis=0)
    cos_fake = test_bn @ fake_centroid / (np.linalg.norm(test_bn, axis=1) * np.linalg.norm(fake_centroid) + 1e-8)
    r = eval_score("5. -cosine_to_fake_centroid", -cos_fake, binary)
    if r: results["neg_cos_fake"] = r

    # 6. Cosine ratio (cos_real - cos_fake)
    r = eval_score("6. cosine_ratio (real-fake)", cos_real - cos_fake, binary)
    if r: results["cos_ratio"] = r

    # 7. Mahalanobis to fake (higher = farther from fake = more real)
    mah_fake = mahalanobis(test_bn, gF)
    r = eval_score("7. mahal_dist_to_fake", mah_fake, binary)
    if r: results["mahal_fake"] = r

    # 8. Mahalanobis to real (lower = closer to real; negate for scoring)
    mah_real = mahalanobis(test_bn, gR)
    r = eval_score("8. -mahal_dist_to_real", -mah_real, binary)
    if r: results["neg_mahal_real"] = r

    # 9. Mahalanobis ratio (mahal_fake - mahal_real)
    r = eval_score("9. mahal_ratio (fake-real)", mah_fake - mah_real, binary)
    if r: results["mahal_ratio"] = r

    # 10. CCL center distances
    ccl_centers = ccl_head.centers.detach().cpu().numpy()  # shape [2, 527]
    print(f"\n  CCL centers shape: {ccl_centers.shape}")
    ccl_fake_center = ccl_centers[0]  # class 0 = fake
    ccl_real_center = ccl_centers[1]  # class 1 = real
    dist_ccl_fake = np.linalg.norm(test_bn - ccl_fake_center, axis=1)
    dist_ccl_real = np.linalg.norm(test_bn - ccl_real_center, axis=1)
    r = eval_score("10a. CCL dist_to_fake", dist_ccl_fake, binary)
    if r: results["ccl_fake"] = r
    r = eval_score("10b. CCL -dist_to_real", -dist_ccl_real, binary)
    if r: results["ccl_neg_real"] = r
    r = eval_score("10c. CCL ratio (fake-real)", dist_ccl_fake - dist_ccl_real, binary)
    if r: results["ccl_ratio"] = r

    # 11. Softmax p(real) — class 4
    p_real = test_sm[:, 4]
    r = eval_score("11. softmax p(real)", p_real, binary)
    if r: results["softmax_preal"] = r

    # 12. Softmax 1-max(p_fake)
    p_fake_max = test_sm[:, :4].max(axis=1)
    r = eval_score("12. softmax 1-max(p_fake)", 1 - p_fake_max, binary)
    if r: results["softmax_neg_pfake"] = r

    # 13. KNN to real (mean dist to 5 nearest real refs)
    print("\n  Computing KNN (k=5)...")
    knn_real = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(ref_real_bn)
    dists_real, _ = knn_real.kneighbors(test_bn)
    mean_knn_real = dists_real.mean(axis=1)
    r = eval_score("13. -KNN5_dist_to_real", -mean_knn_real, binary)
    if r: results["knn_real"] = r

    # 14. KNN to fake
    knn_fake = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(ref_fake_bn)
    dists_fake, _ = knn_fake.kneighbors(test_bn)
    mean_knn_fake = dists_fake.mean(axis=1)
    r = eval_score("14. KNN5_dist_to_fake", mean_knn_fake, binary)
    if r: results["knn_fake"] = r

    # 15. KNN ratio
    r = eval_score("15. KNN5 ratio (fake-real)", mean_knn_fake - mean_knn_real, binary)
    if r: results["knn_ratio"] = r

    # === Fusion: try combining top methods ===
    print(f"\n{'='*100}")
    print("  === Score Fusion ===")
    print(f"{'='*100}")

    # Normalize scores to [0,1] range for fusion
    def norm01(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-12)

    s_base = norm01(-ll_fake)
    s_lr = norm01(ll_real - ll_fake)
    s_ccl = norm01(dist_ccl_fake - dist_ccl_real)
    s_sm = norm01(p_real)
    s_knn = norm01(mean_knn_fake - mean_knn_real)
    s_mahal = norm01(mah_fake - mah_real)

    # Various combos
    eval_score("F1. baseline + LR", s_base + s_lr, binary)
    eval_score("F2. baseline + CCL_ratio", s_base + s_ccl, binary)
    eval_score("F3. baseline + softmax", s_base + s_sm, binary)
    eval_score("F4. baseline + KNN_ratio", s_base + s_knn, binary)
    eval_score("F5. LR + CCL_ratio", s_lr + s_ccl, binary)
    eval_score("F6. LR + softmax", s_lr + s_sm, binary)
    eval_score("F7. baseline+LR+CCL", s_base + s_lr + s_ccl, binary)
    eval_score("F8. baseline+LR+softmax", s_base + s_lr + s_sm, binary)
    eval_score("F9. all5", s_base + s_lr + s_ccl + s_sm + s_knn, binary)
    eval_score("F10. baseline+CCL+softmax", s_base + s_ccl + s_sm, binary)
    eval_score("F11. 2*base+LR+CCL+softmax", 2*s_base + s_lr + s_ccl + s_sm, binary)
    eval_score("F12. base+mahal_ratio", s_base + s_mahal, binary)
    eval_score("F13. base+LR+mahal", s_base + s_lr + s_mahal, binary)

    print(f"\n{'='*100}")
    print("  Done!")


if __name__ == "__main__":
    main()
