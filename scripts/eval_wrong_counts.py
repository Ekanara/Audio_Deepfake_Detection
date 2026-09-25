"""Count wrong predictions by generator and dataset."""
import os, sys, json, numpy as np, torch, pandas as pd
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

# Load model
print("Loading LL_fake_only...")
pipeline = load_model("checkpoint/Beats_journal/Beats_Event_2stage_scratch_stage2_LR1.5e-7_ce3_realW4_8epoch/sample-03.ckpt")

# Collect features
ref_fake_bn, _, _ = collect_feats(pipeline, "data/label/beats/Event_train_stage1_fakeonly.json")
ref_real_bn, _, _ = collect_feats(pipeline, "data/label/beats/Event_train_stage1_realonly.json")

with open("data/label/beats/Event_test_5class.json") as f:
    entries = json.load(f)
paths = [e["audio"] for e in entries]

test_bn, test_lbl, inv = collect_feats(pipeline, "data/label/beats/Event_test_5class.json")
real_idx = {v: k for k, v in inv.items()}["real"]
binary = (test_lbl == real_idx).astype(int)

# Score + threshold
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
    "gt_label": [inv[i] for i in test_lbl],
    "is_real": binary,
    "pred": preds,
    "wrong": (preds != binary).astype(int),
})
df = df.merge(meta, on="wavename_base", how="left")

total_wrong = df["wrong"].sum()
total = len(df)
print(f"\nTong: {total} samples | Sai: {total_wrong} | Dung: {total - total_wrong}")

# === BY GENERATOR ===
print(f"\n{'='*50}")
print("  SAI THEO GENERATOR")
print(f"{'='*50}")
fmt = "{:15s} {:>6s} {:>6s} {:>6s} {:>7s}"
print(fmt.format("Generator", "Tong", "Sai", "Dung", "Acc%"))
print("-" * 45)
for gen in sorted(df["generator"].dropna().unique()):
    m = df["generator"] == gen
    n = m.sum()
    wrong = df.loc[m, "wrong"].sum()
    acc = (n - wrong) / n * 100
    print(fmt.format(gen, str(n), str(wrong), str(n - wrong), f"{acc:.1f}%"))

# === BY DATASET ===
print(f"\n{'='*55}")
print("  SAI THEO DATASET")
print(f"{'='*55}")
fmt2 = "{:20s} {:>6s} {:>6s} {:>6s} {:>7s}"
print(fmt2.format("Dataset", "Tong", "Sai", "Dung", "Acc%"))
print("-" * 50)
for ds in sorted(df["source dataset"].dropna().unique()):
    m = df["source dataset"] == ds
    n = m.sum()
    wrong = df.loc[m, "wrong"].sum()
    acc = (n - wrong) / n * 100
    print(fmt2.format(ds, str(n), str(wrong), str(n - wrong), f"{acc:.1f}%"))

# === BY GENERATOR x DATASET (sorted by wrong count descending) ===
print(f"\n{'='*65}")
print("  SAI THEO GENERATOR x DATASET (sorted by so sai)")
print(f"{'='*65}")
fmt3 = "{:15s} {:20s} {:>6s} {:>6s} {:>7s}"
print(fmt3.format("Generator", "Dataset", "Tong", "Sai", "Acc%"))
print("-" * 60)
rows = []
for (gen, ds), grp in df.dropna(subset=["generator"]).groupby(["generator", "source dataset"]):
    n = len(grp)
    wrong = grp["wrong"].sum()
    rows.append((gen, ds, n, wrong))
rows.sort(key=lambda x: -x[3])
for gen, ds, n, wrong in rows:
    acc = (n - wrong) / n * 100
    print(fmt3.format(gen, ds, str(n), str(wrong), f"{acc:.1f}%"))

# === REAL ONLY - by dataset ===
print(f"\n{'='*55}")
print("  REAL SAMPLES - SAI THEO DATASET")
print(f"{'='*55}")
real_df = df[df["is_real"] == 1]
print(fmt2.format("Dataset", "Tong", "Sai", "Dung", "Acc%"))
print("-" * 50)
for ds in sorted(real_df["source dataset"].dropna().unique()):
    m = real_df["source dataset"] == ds
    n = m.sum()
    wrong = real_df.loc[m, "wrong"].sum()
    acc = (n - wrong) / n * 100
    print(fmt2.format(ds, str(n), str(wrong), str(n - wrong), f"{acc:.1f}%"))

print("\nDone!")
