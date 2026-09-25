"""Plot actual LL(real) and -LL(fake) score distributions for real vs fake samples."""
import os, sys, numpy as np, torch
from scipy.linalg import cho_factor, cho_solve
from argparse import Namespace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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

ckpt = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"
print("Loading checkpoint...")
pipeline, _, _ = load_pipeline_and_ccl(
    checkpoint_path=ckpt, device=device, mode="beats",
    num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

print("Collecting references...")
ref_fake_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")
ref_real_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_realonly.json")
gF = fit_gaussian(ref_fake_bn)
gR = fit_gaussian(ref_real_bn)

test_sets = {
    "test_track2": "data/label/beats/test_track2.json",
    "Event_test": "data/label/beats/Event_test_5class.json",
    "TUTASC19_test": "data/label/beats/TUTASC19_test_5class.json",
}

os.makedirs("inference_outputs/score_distributions", exist_ok=True)

for ds_name, ds_path in test_sets.items():
    print(f"\nCollecting {ds_name}...")
    test_bn, test_lbl, inv = collect(pipeline, ds_path)
    real_idx = {v: k for k, v in inv.items()}["real"]
    is_real = test_lbl == real_idx

    ll_fake = loglik(test_bn, gF)
    ll_real = loglik(test_bn, gR)
    maha_fake = mahalanobis(test_bn, gF)
    maha_real = mahalanobis(test_bn, gR)

    methods = [
        ("-Log-Likelihood(fake) [baseline]", -ll_fake),
        ("Log-Likelihood(real) only", ll_real),
        ("Mahalanobis to fake reference", maha_fake),
        ("Mahalanobis to real reference", -maha_real),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Score distributions — {ds_name}", fontsize=16, fontweight="bold")

    for ax, (method_name, scores) in zip(axes.flat, methods):
        real_scores = scores[is_real]
        fake_scores = scores[~is_real]

        lo = min(np.percentile(real_scores, 1), np.percentile(fake_scores, 1))
        hi = max(np.percentile(real_scores, 99), np.percentile(fake_scores, 99))
        bins = np.linspace(lo, hi, 80)

        ax.hist(fake_scores, bins=bins, alpha=0.5, color="#E24B4A", label=f"Fake (n={len(fake_scores)})", density=True)
        ax.hist(real_scores, bins=bins, alpha=0.5, color="#1D9E75", label=f"Real (n={len(real_scores)})", density=True)
        ax.set_title(method_name, fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylabel("Density")
        ax.axvline(x=np.median(real_scores), color="#1D9E75", linestyle="--", alpha=0.6, linewidth=1)
        ax.axvline(x=np.median(fake_scores), color="#E24B4A", linestyle="--", alpha=0.6, linewidth=1)

    plt.tight_layout()
    out_path = f"inference_outputs/score_distributions/{ds_name}_score_dist.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

print("\nDone!")
