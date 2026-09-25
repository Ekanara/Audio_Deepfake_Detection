"""
Sweep: Use CCL logits / ArcFace logits as scoring signals.
The model was trained with CCL (CenterContrastiveLoss) which learns real/fake centers.
The CCL logits might be a better scoring signal than raw Gaussian LL.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from torch.utils.data import DataLoader
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl
from utils.training_utils import CenterContrastiveLoss, ArcFaceLoss


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
    return {"auc": auc, "eer": eer, "real_acc": real_acc, "fake_acc": fake_acc}


def print_result(name, r):
    print(f"  {name:55s}  AUC={r['auc']:.4f}  EER={r['eer']*100:5.2f}%  "
          f"Real%={r['real_acc']*100:5.2f}%  Fake%={r['fake_acc']*100:5.2f}%")


@torch.no_grad()
def collect_all_features(pipeline, ccl_head, arc_head, json_file, device):
    """Collect embeddings + CCL/ArcFace logits."""
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv_label = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)

    all_bn, all_sm, all_ccl, all_arc, all_lbl = [], [], [], [], []
    pipeline.eval()
    ccl_head.eval()
    arc_head.eval()

    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        bn = outputs[0].float()
        sm = outputs[1].float()

        # CCL logits: distance to real/fake centers → softmax
        dummy_labels = torch.zeros(bn.size(0), dtype=torch.long, device=device)
        _, ccl_logits = ccl_head(bn, dummy_labels)
        ccl_probs = F.softmax(ccl_logits, dim=1)  # [B, 2]: [fake_prob, real_prob]

        # ArcFace logits: angular similarity to class prototypes
        with torch.no_grad():
            arc_head.weight.data = F.normalize(arc_head.weight.data, p=2, dim=1)
        x_norm = F.normalize(bn.float(), p=2, dim=1)
        arc_cos = x_norm @ arc_head.weight.T  # [B, 5]
        arc_probs = F.softmax(arc_cos * 30, dim=1)  # scale by s=30

        all_bn.append(bn.cpu().numpy())
        all_sm.append(sm.cpu().numpy())
        all_ccl.append(ccl_probs.cpu().numpy())
        all_arc.append(arc_probs.cpu().numpy())
        if "label" in batch:
            all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs

    bn = np.concatenate(all_bn)
    sm = np.concatenate(all_sm)
    ccl = np.concatenate(all_ccl)
    arc = np.concatenate(all_arc)
    lbl = np.array(all_lbl)
    return bn, sm, ccl, arc, lbl, inv_label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"
    ref_fake_json = "data/label/beats/Event_train_stage1_fakeonly.json"
    ref_real_json = "data/label/beats/Event_train_stage1_realonly.json"

    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "Event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
    }

    print("Loading model + CCL/ArcFace heads...")
    pipeline, ccl_head, arc_head = load_pipeline_and_ccl(
        checkpoint_path=ckpt_path, device=device, mode="beats",
        num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

    # Collect reference features
    print("Collecting reference features...")
    F_bn, F_sm, F_ccl, F_arc, _, _ = collect_all_features(pipeline, ccl_head, arc_head, ref_fake_json, device)
    R_bn, R_sm, R_ccl, R_arc, _, _ = collect_all_features(pipeline, ccl_head, arc_head, ref_real_json, device)
    gF = fit_gaussian(F_bn)
    gR = fit_gaussian(R_bn)
    print(f"  Fake: {F_bn.shape[0]}, Real: {R_bn.shape[0]}")

    sep = "=" * 110
    for ds_name, ds_path in test_sets.items():
        print(f"\n{sep}")
        print(f"  {ds_name}")
        print(sep)

        T_bn, T_sm, T_ccl, T_arc, T_lbl, inv = collect_all_features(
            pipeline, ccl_head, arc_head, ds_path, device)
        real_idx = {v: k for k, v in inv.items()}["real"]
        labels = (T_lbl == real_idx).astype(int)

        results = {}

        # ── Baselines ─────────────────────────────────────────────────
        ll_fake = -loglik(T_bn, gF)
        lr = loglik(T_bn, gR) - loglik(T_bn, gF)
        p_real = T_sm[:, 4]

        results["A1. baseline: -LL(fake)"] = eval_score(ll_fake, labels)
        results["A2. LR: LL(real)-LL(fake)"] = eval_score(lr, labels)
        results["A3. softmax p(real)"] = eval_score(p_real, labels)

        # ── CCL logits ────────────────────────────────────────────────
        ccl_preal = T_ccl[:, 1]  # CCL real probability
        results["B1. CCL p(real)"] = eval_score(ccl_preal, labels)

        # ── ArcFace logits ────────────────────────────────────────────
        arc_preal = T_arc[:, 4]  # ArcFace real probability (class 4 = real)
        results["B2. ArcFace p(real)"] = eval_score(arc_preal, labels)

        # ── CCL + LL fusions ──────────────────────────────────────────
        s = rank_pct(ll_fake) + rank_pct(ccl_preal)
        results["C1. rank: -LL(fake) + CCL_p(real)"] = eval_score(s, labels)

        s = rank_pct(lr) + rank_pct(ccl_preal)
        results["C2. rank: LR + CCL_p(real)"] = eval_score(s, labels)

        s = rank_pct(ll_fake) + rank_pct(ccl_preal) + rank_pct(p_real)
        results["C3. rank3: -LL(fake)+CCL+SM"] = eval_score(s, labels)

        s = rank_pct(lr) + rank_pct(ccl_preal) + rank_pct(p_real)
        results["C4. rank3: LR+CCL+SM"] = eval_score(s, labels)

        # ── ArcFace fusions ───────────────────────────────────────────
        s = rank_pct(ll_fake) + rank_pct(arc_preal)
        results["D1. rank: -LL(fake) + ArcFace_p(real)"] = eval_score(s, labels)

        s = rank_pct(lr) + rank_pct(arc_preal)
        results["D2. rank: LR + ArcFace_p(real)"] = eval_score(s, labels)

        # ── Weighted fusions ──────────────────────────────────────────
        for alpha in [0.3, 0.5, 0.7]:
            s = (1-alpha) * rank_pct(ll_fake) + alpha * rank_pct(ccl_preal)
            results[f"E1. weighted a={alpha}: -LL(fake)+CCL"] = eval_score(s, labels)

        for alpha in [0.3, 0.5, 0.7]:
            s = (1-alpha) * rank_pct(lr) + alpha * rank_pct(ccl_preal)
            results[f"E2. weighted a={alpha}: LR+CCL"] = eval_score(s, labels)

        # ── 4-signal fusion ───────────────────────────────────────────
        s = rank_pct(ll_fake) + rank_pct(ccl_preal) + rank_pct(p_real) + rank_pct(arc_preal)
        results["F1. rank4: LL+CCL+SM+ArcFace"] = eval_score(s, labels)

        s = rank_pct(lr) + rank_pct(ccl_preal) + rank_pct(p_real) + rank_pct(arc_preal)
        results["F2. rank4: LR+CCL+SM+ArcFace"] = eval_score(s, labels)

        # ── Print sorted by real_acc ──────────────────────────────────
        print(f"\n  {'Method':55s}  {'AUC':>6s}  {'EER':>7s}  {'Real%':>7s}  {'Fake%':>7s}")
        print("  " + "-" * 90)
        for name, r in sorted(results.items(), key=lambda x: -x[1]["real_acc"]):
            print_result(name, r)

    print("\nDone!")


if __name__ == "__main__":
    main()
