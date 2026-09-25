"""Quick verify: LR inference improvement on LL_fake_only checkpoint."""
import os, sys
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve
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


@torch.no_grad()
def collect_features(pipeline, json_file, device):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv_label = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        audio = batch["audio"].to(device, dtype=torch.float32)
        outputs = pipeline.forward_pipeline(audio)
        all_bn.append(outputs[0].float().cpu().numpy())
        if "label" in batch:
            all_lbl.extend(batch["label"].cpu().numpy())
        del audio, outputs
    bn = np.concatenate(all_bn)
    lbl = np.array(all_lbl)
    return bn, lbl, inv_label


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"
    ref_fake = "data/label/beats/Event_train_stage1_fakeonly.json"
    ref_real = "data/label/beats/Event_train_stage1_realonly.json"

    test_sets = {
        "test_track2": "data/label/beats/test_track2.json",
        "Event_test": "data/label/beats/old/audio_labels_beats_event_test.json",
        "TUTASC19_test": "data/label/beats/TUTASC19_test.json",
    }

    print("Loading model...")
    pipeline, _, _ = load_pipeline_and_ccl(
        checkpoint_path=ckpt, device=device, mode="beats",
        num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

    print("Collecting reference features...")
    F_bn, _, _ = collect_features(pipeline, ref_fake, device)
    R_bn, _, _ = collect_features(pipeline, ref_real, device)
    gF = fit_gaussian(F_bn)
    gR = fit_gaussian(R_bn)
    print(f"  Fake: {F_bn.shape[0]}, Real: {R_bn.shape[0]}")

    print(f"\n{'Test Set':20s} {'Method':20s} {'AUC':>7s} {'EER':>8s} {'Real%':>8s} {'Fake%':>8s} {'ΔReal':>8s}")
    print("-" * 85)

    for ds_name, ds_path in test_sets.items():
        T_bn, T_lbl, inv = collect_features(pipeline, ds_path, device)
        real_idx = {v: k for k, v in inv.items()}["real"]
        labels = (T_lbl == real_idx).astype(int)

        # Baseline
        s1 = -loglik(T_bn, gF)
        r1 = eval_score(s1, labels)
        print(f"{ds_name:20s} {'baseline -LL(fake)':20s} {r1['auc']:7.4f} {r1['eer']*100:7.2f}% {r1['real_acc']*100:7.2f}% {r1['fake_acc']*100:7.2f}% {'':>8s}")

        # LR
        s2 = loglik(T_bn, gR) - loglik(T_bn, gF)
        r2 = eval_score(s2, labels)
        delta = (r2['real_acc'] - r1['real_acc']) * 100
        sign = "+" if delta >= 0 else ""
        print(f"{'':20s} {'LR: LL(R)-LL(F)':20s} {r2['auc']:7.4f} {r2['eer']*100:7.2f}% {r2['real_acc']*100:7.2f}% {r2['fake_acc']*100:7.2f}% {sign}{delta:7.2f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
