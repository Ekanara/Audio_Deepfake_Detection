"""Evaluate real-reference scoring methods on test_track2 to get full metrics."""
import os, sys, numpy as np, torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from argparse import Namespace
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

device = "cuda"

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
    mu, cf, lower, _ = g
    diff = X.astype(np.float64) - mu
    solved = cho_solve((cf, lower), diff.T).T
    return np.sqrt(np.einsum("ij,ij->i", diff, solved))

@torch.no_grad()
def collect(pipeline, json_file):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=json_file, transformation=None, args=ds_args)
    inv = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=0, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        a = batch["audio"].to(device, dtype=torch.float32)
        o = pipeline.forward_pipeline(a)
        all_bn.append(o[0].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del a, o
    return np.concatenate(all_bn), np.array(all_lbl), inv

def eval_method(name, scores, binary):
    auc = roc_auc_score(binary, scores)
    fpr, tpr, thrs = roc_curve(binary, scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    thr = thrs[eer_idx]
    preds = (scores >= thr).astype(int)
    acc = accuracy_score(binary, preds)
    f1r = f1_score(binary, preds, pos_label=1)
    f1f = f1_score(binary, preds, pos_label=0)
    f1m = (f1r + f1f) / 2
    real_acc = (preds[binary == 1] == 1).mean()
    fake_acc = (preds[binary == 0] == 0).mean()
    print(f"  {name:40s}  AUC={auc:.4f}  Acc={acc*100:.2f}%  F1m={f1m:.4f}  F1r={f1r:.4f}  F1f={f1f:.4f}  Real={real_acc*100:.2f}%  Fake={fake_acc*100:.2f}%")

ckpt = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"
print("Loading checkpoint...")
pipeline, _, _ = load_pipeline_and_ccl(
    checkpoint_path=ckpt, device=device, mode="beats",
    num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

print("Collecting fake reference...")
ref_fake_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")
print("Collecting real reference...")
ref_real_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_realonly.json")
gF = fit_gaussian(ref_fake_bn)
gR = fit_gaussian(ref_real_bn)
real_centroid = ref_real_bn.mean(axis=0)

test_sets = {
    "test_track2": "data/label/beats/test_track2.json",
    "Event_test": "data/label/beats/Event_test_5class.json",
    "TUTASC19_test": "data/label/beats/TUTASC19_test_5class.json",
}

for ds_name, ds_path in test_sets.items():
    print(f"\nCollecting {ds_name}...")
    test_bn, test_lbl, inv = collect(pipeline, ds_path)
    real_idx = {v: k for k, v in inv.items()}["real"]
    binary = (test_lbl == real_idx).astype(int)

    ll_fake = loglik(test_bn, gF)
    ll_real = loglik(test_bn, gR)
    cos_real = test_bn @ real_centroid / (np.linalg.norm(test_bn, axis=1) * np.linalg.norm(real_centroid) + 1e-8)

    print(f"\n{'='*130}")
    print(f"  {ds_name} ({len(binary)} samples, {binary.sum()} real, {(1-binary).sum()} fake)")
    print(f"{'='*130}")
    maha_fake = mahalanobis(test_bn, gF)
    maha_real = mahalanobis(test_bn, gR)

    eval_method("-Log-Likelihood(fake) [BASELINE]", -ll_fake, binary)
    eval_method("Log-Likelihood(real) - Log-Likelihood(fake)", ll_real - ll_fake, binary)
    eval_method("Log-Likelihood(real) only", ll_real, binary)
    eval_method("Mahalanobis to fake reference", maha_fake, binary)
    eval_method("Mahalanobis to real reference", -maha_real, binary)
    eval_method("Cosine similarity to real centroid", cos_real, binary)
    del test_bn, test_lbl

print("\nDone!")
