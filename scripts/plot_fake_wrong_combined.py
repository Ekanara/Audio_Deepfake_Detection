"""Bar charts: fake wrong predictions from Combined Event+TUTASC19 model on Event_test."""
import os, sys, json, numpy as np, torch, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_curve
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
def collect_feats(pipeline, jf):
    ds_args = Namespace(num_label=5, three_loss=True, audio_aug=False,
                        audio_mixup=False, audio_aug_prob=0, audio_mixup_prob=0)
    ds = BeatsDataset(json_file=jf, transformation=None, args=ds_args)
    inv = {v: k for k, v in ds.label.items()}
    dl = DataLoader(ds, batch_size=512, num_workers=4, shuffle=False, pin_memory=True)
    all_bn, all_lbl = [], []
    pipeline.eval()
    for batch in dl:
        a = batch["audio"].to(device, dtype=torch.float32)
        o = pipeline.forward_pipeline(a)
        all_bn.append(o[0].float().cpu().numpy())
        all_lbl.extend(batch["label"].cpu().numpy())
        del a, o
    return np.concatenate(all_bn), np.array(all_lbl), inv

# Load metadata
meta = pd.read_csv("data/test/test_set/test_metadata.csv")
meta["wavename_base"] = meta["wavename"].str.replace(".wav", "", regex=False)
meta["gen_combo"] = meta["faketype"] + "/" + meta["generator"]

# Load Combined model
ckpt = "checkpoint/Beats_journal/combined_Event_TUTASC19_scratch/stage2/S2r_ep02.ckpt"
print(f"Loading: {ckpt}")
pipeline = load_model(ckpt)

# Use combined reference
ref_fake_bn, _, _ = collect_feats(pipeline, "data/label/beats/combined_Event_TUTASC19_train_fakeonly.json")

with open("data/label/beats/Event_test_5class.json") as f:
    entries = json.load(f)
paths = [e["audio"] for e in entries]

test_bn, test_lbl, inv = collect_feats(pipeline, "data/label/beats/Event_test_5class.json")
real_idx = {v: k for k, v in inv.items()}["real"]
binary = (test_lbl == real_idx).astype(int)

gF = fit_gaussian(ref_fake_bn)
scores = -loglik(test_bn, gF)
fpr, tpr, thrs = roc_curve(binary, scores, pos_label=1)
fnr = 1.0 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
thr = thrs[eer_idx]
preds = (scores >= thr).astype(int)

wavenames = [os.path.basename(p).replace(".wav", "") for p in paths]
df = pd.DataFrame({"wavename_base": wavenames, "is_real": binary, "pred": preds})
df = df.merge(meta, on="wavename_base", how="left")

# Fake only, predicted wrong
fake_wrong = df[(df["is_real"] == 0) & (df["pred"] == 1)]
fake_total = df[df["is_real"] == 0]

total_wrong = len(fake_wrong)
total_fake = len(fake_total)
print(f"Total fake: {total_fake}, Fake wrong: {total_wrong}")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# === 7 generators (faketype/generator combo) ===
combo_wrong = fake_wrong.groupby("gen_combo").size()
combo_total = fake_total.groupby("gen_combo").size()

combos = []
for c in combo_wrong.index:
    if "real" in c or "none" in c:
        continue
    combos.append((c, combo_wrong[c], combo_total.get(c, 0)))
combos.sort(key=lambda x: -x[1])

names_gen = [c[0] for c in combos]
wrongs_gen = [c[1] for c in combos]

print("\nFake wrong by generation method (7):")
for n, w, t in combos:
    print(f"  {n:20s}  {w:5d} / {t:5d}  ({w/t*100:.1f}%)")

colors7 = ['#f06050', '#f09040', '#f5d040', '#50d080', '#5ab0e8', '#b070c8', '#40c8a8']

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.bar(names_gen, wrongs_gen, color=colors7, edgecolor='white', linewidth=1.5, width=0.65)
ax.set_ylabel('Count', fontsize=12, labelpad=10)
ax.tick_params(axis='x', labelsize=10, rotation=20)
ax.tick_params(axis='y', labelsize=11)
ax.set_ylim(0, max(wrongs_gen) * 1.1)
ax.grid(axis='y', alpha=0.2, linestyle='--')
plt.tight_layout()
fig.savefig('output/combined_analysis/Combined Fake Wrong by Generator.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: Combined Fake Wrong by Generator.png")

# === By Dataset ===
ds_wrong = fake_wrong.groupby("source dataset").size().sort_values(ascending=False)
ds_total = fake_total.groupby("source dataset").size()

datasets = list(ds_wrong.index)
wrongs_ds = list(ds_wrong.values)

print("\nFake wrong by Dataset:")
for d in datasets:
    w = ds_wrong[d]
    t = ds_total[d]
    print(f"  {d:20s}  {w:5d} / {t:5d}  ({w/t*100:.1f}%)")

colors_ds = ['#f06050', '#f09040', '#f5d040', '#50d080', '#5ab0e8', '#b070c8', '#40c8a8'][:len(datasets)]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(datasets, wrongs_ds, color=colors_ds, edgecolor='white', linewidth=1.5, width=0.65)
ax.set_ylabel('Count', fontsize=12, labelpad=10)
ax.tick_params(axis='x', labelsize=10, rotation=15)
ax.tick_params(axis='y', labelsize=11)
ax.set_ylim(0, max(wrongs_ds) * 1.1)
ax.grid(axis='y', alpha=0.2, linestyle='--')
plt.tight_layout()
fig.savefig('output/combined_analysis/Combined Fake Wrong by Dataset.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: Combined Fake Wrong by Dataset.png")
