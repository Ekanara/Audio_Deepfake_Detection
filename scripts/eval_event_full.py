"""Evaluate Event_full checkpoints on test_track2 + Foley_Sound."""
import os, sys, numpy as np, torch
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, accuracy_score
from argparse import Namespace
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from beats.model_beat import model_beat

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

def load_model(p):
    ckpt = torch.load(p, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    pipeline = model_beat(num_label=5, three_loss=True, feature_layer="predictor").to(device)
    ps = {k[9:]: v for k, v in sd.items() if k.startswith("pipeline.")}
    cur = pipeline.state_dict()
    pipeline.load_state_dict({k: v for k, v in ps.items() if k in cur and v.shape == cur[k].shape}, strict=False)
    return pipeline

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

def eval_checkpoint(ckpt_path, ref_fake_json, test_sets, label):
    print(f"\n{'='*80}")
    print(f"  {label}: {os.path.basename(ckpt_path)}")
    print(f"{'='*80}")

    pipeline = load_model(ckpt_path)
    ref_bn, _, _ = collect(pipeline, ref_fake_json)
    gF = fit_gaussian(ref_bn)
    del ref_bn

    print(f"  {'Test Set':15s} {'AUC':>7s} {'EER':>7s} {'Acc':>7s} {'F1r':>7s} {'F1f':>7s} {'F1m':>7s} {'Real%':>7s} {'Fake%':>7s}")
    print("  " + "-" * 75)

    for ds_name, ds_path in test_sets.items():
        bn, lbl, inv = collect(pipeline, ds_path)
        real_idx = {v: k for k, v in inv.items()}["real"]
        binary = (lbl == real_idx).astype(int)
        score = -loglik(bn, gF)
        auc = roc_auc_score(binary, score)
        fpr, tpr, thrs = roc_curve(binary, score, pos_label=1)
        fnr = 1.0 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
        thr = thrs[eer_idx]
        preds = (score >= thr).astype(int)
        acc = accuracy_score(binary, preds)
        f1r = f1_score(binary, preds, pos_label=1)
        f1f = f1_score(binary, preds, pos_label=0)
        f1m = (f1r + f1f) / 2
        real_acc = (preds[binary == 1] == 1).mean()
        fake_acc = (preds[binary == 0] == 0).mean()
        print(f"  {ds_name:15s} {auc:7.4f} {eer*100:6.2f}% {acc*100:6.2f}% {f1r:7.4f} {f1f:7.4f} {f1m:7.4f} {real_acc*100:6.2f}% {fake_acc*100:6.2f}%")

    del pipeline
    torch.cuda.empty_cache()

ref_fake = "data/label/beats/Event_full_fakeonly.json"
test_sets = {
    "test_track2": "data/label/beats/test_track2.json",
    "Foley_Sound": "data/label/beats/Foley_Sound_test.json",
}

base = "checkpoint/Beats_journal/Event_full_scratch"

# Eval last 3 stage1 checkpoints + all stage2 checkpoints
checkpoints = []
for ep in [12, 13, 14]:
    p = os.path.join(base, "stage1", f"S1_ep{ep:02d}.ckpt")
    if os.path.exists(p):
        checkpoints.append((p, f"Stage1 ep{ep}"))

for ep in range(8):
    p = os.path.join(base, "stage2", f"S2_ep{ep:02d}.ckpt")
    if os.path.exists(p):
        checkpoints.append((p, f"Stage2 ep{ep}"))

for path, label in checkpoints:
    eval_checkpoint(path, ref_fake, test_sets, label)

print("\nDone!")
