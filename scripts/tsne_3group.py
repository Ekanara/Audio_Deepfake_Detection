"""t-SNE: fake training vs real test (challenge) vs fake test (challenge)
using LL_fake_only checkpoint embeddings."""
import os, sys, numpy as np, torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from argparse import Namespace
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from base_dataset import BeatsDataset
from inference.inference_ccl import load_pipeline_and_ccl

COLORS = {
    "Fake (train)": "#e74c3c",
    "Real (test_track2)": "#2ecc71",
    "Fake (test_track2)": "#3498db",
}
ORDER = ["Fake (train)", "Real (test_track2)", "Fake (test_track2)"]

@torch.no_grad()
def collect(pipeline, json_file, device):
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

device = "cuda"
ckpt = "checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt"

print("Loading checkpoint...")
pipeline, _, _ = load_pipeline_and_ccl(
    checkpoint_path=ckpt, device=device, mode="beats",
    num_classes=5, embed_dim=527, truncate_layers=0, beats_feature="predictor")

print("Collecting fake training...")
fake_train_bn, _, _ = collect(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json", device)

print("Collecting test_track2...")
test_bn, test_lbl, test_inv = collect(pipeline, "data/label/beats/test_track2.json", device)
real_idx = {v: k for k, v in test_inv.items()}["real"]
test_is_real = test_lbl == real_idx

test_real_bn = test_bn[test_is_real]
test_fake_bn = test_bn[~test_is_real]

print(f"Fake train: {len(fake_train_bn)}, Test real: {len(test_real_bn)}, Test fake: {len(test_fake_bn)}")

max_per_group = 400
np.random.seed(42)

def subsample(arr, n):
    if len(arr) <= n:
        return arr
    idx = np.random.choice(len(arr), n, replace=False)
    return arr[idx]

ft = subsample(fake_train_bn, max_per_group)
tr = subsample(test_real_bn, max_per_group)
tf = subsample(test_fake_bn, max_per_group)

all_embs = np.concatenate([ft, tr, tf])
all_groups = (["Fake (train)"] * len(ft) +
              ["Real (test_track2)"] * len(tr) +
              ["Fake (test_track2)"] * len(tf))
all_groups = np.array(all_groups)

print(f"Running t-SNE on {len(all_embs)} samples...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42,
            max_iter=1000, learning_rate="auto", init="pca")
embs_2d = tsne.fit_transform(all_embs)

fig, ax = plt.subplots(figsize=(10, 8))
for grp in ORDER:
    m = all_groups == grp
    ax.scatter(embs_2d[m, 0], embs_2d[m, 1],
               c=COLORS[grp], label=grp,
               alpha=0.6, s=12, edgecolors="none")

ax.legend(fontsize=11, markerscale=3, loc="best")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE: Fake train vs Test track2 (real & fake)")
ax.grid(True, alpha=0.3)

os.makedirs("inference_outputs/tsne", exist_ok=True)
out = "inference_outputs/tsne/tsne_fake_train_vs_test_track2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
print("Done!")
