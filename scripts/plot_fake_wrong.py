"""Bar charts: only FAKE samples predicted wrongly (fake predicted as real)."""
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

# Load model + inference
pipeline = load_model("checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt")
ref_fake_bn, _, _ = collect_feats(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")

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

# Build dataframe
wavenames = [os.path.basename(p).replace(".wav", "") for p in paths]
df = pd.DataFrame({
    "wavename_base": wavenames,
    "is_real": binary,
    "pred": preds,
})
df = df.merge(meta, on="wavename_base", how="left")

# Filter: FAKE only, predicted WRONG (fake predicted as real)
fake_wrong = df[(df["is_real"] == 0) & (df["pred"] == 1)]
fake_total = df[df["is_real"] == 0]

print(f"Total fake: {len(fake_total)}, Fake predicted wrong: {len(fake_wrong)}")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# === Chart 1: Fake wrong by Generator ===
gen_wrong = fake_wrong.groupby("generator").size().sort_values(ascending=False)
gen_total = fake_total.groupby("generator").size()

generators = list(gen_wrong.index)
wrong_counts = list(gen_wrong.values)
colors1 = ['#e74c3c', '#e67e22', '#f1c40f', '#3498db', '#2ecc71'][:len(generators)]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(generators, wrong_counts, color=colors1, edgecolor='white', linewidth=1.5, width=0.65)
ax.set_ylabel('Fake Predicted as Real', fontsize=13, fontweight='bold', labelpad=10)
ax.tick_params(axis='both', labelsize=11)
ax.set_ylim(0, max(wrong_counts) * 1.1)
ax.grid(axis='y', alpha=0.2, linestyle='--')
plt.tight_layout()
fig.savefig('output/combined_analysis/Fake Wrong by Generator.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: Fake Wrong by Generator.png")

# Print counts
print("\nFake wrong by Generator:")
for g in generators:
    w = gen_wrong[g]
    t = gen_total[g]
    print(f"  {g:15s}  {w:5d} / {t:5d}  ({w/t*100:.1f}%)")

# === Chart 2: Fake wrong by Dataset ===
ds_wrong = fake_wrong.groupby("source dataset").size().sort_values(ascending=False)
ds_total = fake_total.groupby("source dataset").size()

datasets = list(ds_wrong.index)
wrong_ds = list(ds_wrong.values)
colors2 = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c'][:len(datasets)]

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(datasets, wrong_ds, color=colors2, edgecolor='white', linewidth=1.5, width=0.65)
ax.set_ylabel('Fake Predicted as Real', fontsize=13, fontweight='bold', labelpad=10)
ax.tick_params(axis='both', labelsize=11, rotation=15)
ax.set_ylim(0, max(wrong_ds) * 1.1)
ax.grid(axis='y', alpha=0.2, linestyle='--')
plt.tight_layout()
fig.savefig('output/combined_analysis/Fake Wrong by Dataset.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: Fake Wrong by Dataset.png")

print("\nFake wrong by Dataset:")
for d in datasets:
    w = ds_wrong[d]
    t = ds_total[d]
    print(f"  {d:20s}  {w:5d} / {t:5d}  ({w/t*100:.1f}%)")
